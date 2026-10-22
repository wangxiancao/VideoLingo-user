#include "mel_frontend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

class ExecutionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::string cuda_error(cudaError_t status, const char* expression) {
  std::ostringstream out;
  out << expression << " failed: " << cudaGetErrorString(status);
  return out.str();
}

#define CUDA_CHECK(expr)                                                         \
  do {                                                                           \
    const cudaError_t status_ = (expr);                                          \
    if (status_ != cudaSuccess) throw ExecutionError(cuda_error(status_, #expr)); \
  } while (false)

// Host-side mel filter bank, reproducing transformers audio_utils.mel_filter_bank
// with mel_scale="slaney", norm="slaney": slaney Hz<->mel piecewise mapping,
// 128 triangles between 130 mel-space endpoints, area normalization.
std::vector<double> build_mel_filterbank() {
  auto hertz_to_mel = [](double hertz) {
    const double kMinLogHertz = 1000.0;
    const double kMinLogMel = 15.0;
    const double kLogstep = 27.0 / std::log(6.4);
    if (hertz < kMinLogHertz) return 3.0 * hertz / 200.0;
    return kMinLogMel + std::log(hertz / kMinLogHertz) * kLogstep;
  };
  auto mel_to_hertz = [](double mels) {
    const double kMinLogHertz = 1000.0;
    const double kMinLogMel = 15.0;
    const double kLogstep = std::log(6.4) / 27.0;
    const double linear = 200.0 * mels / 3.0;
    if (mels < kMinLogMel) return linear;
    return kMinLogHertz * std::exp(kLogstep * (mels - kMinLogMel));
  };

  const int num_mel = kMelBins;
  const int num_freq = kFrequencyBins;
  const double fmin = 0.0, fmax = kSampleRate / 2.0;

  std::vector<double> mel_points(num_mel + 2);
  const double mel_min = hertz_to_mel(fmin), mel_max = hertz_to_mel(fmax);
  for (int i = 0; i < num_mel + 2; ++i)
    mel_points[i] = mel_min + (mel_max - mel_min) * i / double(num_mel + 1);
  std::vector<double> filter_freqs(num_mel + 2);
  for (int i = 0; i < num_mel + 2; ++i) filter_freqs[i] = mel_to_hertz(mel_points[i]);

  std::vector<double> fft_freqs(num_freq);
  for (int i = 0; i < num_freq; ++i) fft_freqs[i] = fmax * i / double(num_freq - 1);

  // Triangular filters: min(rising, falling) slope clipped at zero, then the
  // slaney area normalization 2 / (right - left). This mirrors the reference's
  // diff-based slopes: rising = (f - left) / (mid - left),
  // falling = (right - f) / (right - mid).
  std::vector<double> filters(static_cast<size_t>(num_freq) * num_mel, 0.0);
  for (int m = 0; m < num_mel; ++m) {
    const double left = filter_freqs[m], mid = filter_freqs[m + 1], right = filter_freqs[m + 2];
    const double enorm = 2.0 / (right - left);
    for (int k = 0; k < num_freq; ++k) {
      const double rising = (fft_freqs[k] - left) / (mid - left);
      const double falling = (right - fft_freqs[k]) / (right - mid);
      filters[static_cast<size_t>(k) * num_mel + m] =
          std::max(0.0, std::min(rising, falling)) * enorm;
    }
  }
  return filters;
}

// window[n] * cos(2*pi*k*n/400) and window[n] * sin(...) for every
// (frequency bin k, tap n): precomputed on the host in double precision.
void build_windowed_trig_tables(std::vector<float>* wcos, std::vector<float>* wsin) {
  const int n_fft = kFftLength;
  std::vector<double> window(n_fft);
  for (int n = 0; n < n_fft; ++n)
    window[n] = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / n_fft));  // periodic Hann
  wcos->resize(static_cast<size_t>(kFrequencyBins) * n_fft);
  wsin->resize(static_cast<size_t>(kFrequencyBins) * n_fft);
  for (int k = 0; k < kFrequencyBins; ++k) {
    for (int n = 0; n < n_fft; ++n) {
      const double angle = 2.0 * M_PI * (static_cast<double>(k) * n / n_fft);
      (*wcos)[static_cast<size_t>(k) * n_fft + n] =
          static_cast<float>(window[n] * std::cos(angle));
      (*wsin)[static_cast<size_t>(k) * n_fft + n] =
          static_cast<float>(window[n] * std::sin(angle));
    }
  }
}


// Per-chunk log/normalize on device. Stage A: log10 + per-(chunk,bin) max;
// stage B: chunk-level max reduction; stage C: clamp to chunk max - 8 and
// (x+4)/4, dropping the last frame.
__global__ void mel_log_max_kernel(const float* __restrict__ mel,
                                   float* __restrict__ logs, float* __restrict__ bin_max,
                                   int chunks) {
  const int chunk = blockIdx.z;
  const int bin = blockIdx.y;
  const int tid = threadIdx.x;
  if (chunk >= chunks || bin >= kMelBins) return;
  const float* src = mel + (static_cast<size_t>(chunk) * kMelBins + bin) * kFrameCount;
  float* dst = logs + (static_cast<size_t>(chunk) * kMelBins + bin) * kOutputFrames;
  __shared__ float s_max[256];
  float local_max = -1e30f;
  for (int f = tid; f < kOutputFrames; f += blockDim.x) {
    const float v = log10f(fmaxf(src[f], 1e-10f));
    dst[f] = v;
    local_max = fmaxf(local_max, v);
  }
  s_max[tid] = local_max;
  __syncthreads();
  float maximum = -1e30f;
  for (int i = 0; i < blockDim.x; ++i) maximum = fmaxf(maximum, s_max[i]);
  if (tid == 0) bin_max[chunk * kMelBins + bin] = maximum;
}

__global__ void mel_chunk_max_kernel(const float* __restrict__ bin_max,
                                     float* __restrict__ chunk_max, int chunks) {
  const int chunk = blockIdx.x;
  if (chunk >= chunks) return;
  float maximum = -1e30f;
  for (int b = 0; b < kMelBins; ++b)
    maximum = fmaxf(maximum, bin_max[chunk * kMelBins + b]);
  chunk_max[chunk] = maximum;
}

__global__ void mel_normalize_kernel(const float* __restrict__ logs,
                                     const float* __restrict__ chunk_max,
                                     float* __restrict__ out, int chunks) {
  const int chunk = blockIdx.z;
  const int bin = blockIdx.y;
  const int tid = threadIdx.x;
  if (chunk >= chunks || bin >= kMelBins) return;
  const float* src = logs + (static_cast<size_t>(chunk) * kMelBins + bin) * kOutputFrames;
  float* dst = out + (static_cast<size_t>(chunk) * kMelBins + bin) * kOutputFrames;
  const float floor_log = chunk_max[chunk] - 8.0f;
  for (int f = tid; f < kOutputFrames; f += blockDim.x)
    dst[f] = (fmaxf(src[f], floor_log) + 4.0f) / 4.0f;
}

// np.pad(x, (200, 200), mode="reflect") indexing: j < 0 -> x[-j], j >= N -> x[2N-2-j].
__device__ int reflect_index(int j, int n) {
  if (j < 0) return -j;
  if (j >= n) return 2 * n - 2 - j;
  return j;
}

// One thread per (frequency bin, frame): DFT tap loop in float32 with
// host-precomputed windowed trig tables. power = |X|^2.
__global__ void stft_power_kernel(const float* __restrict__ pcm, int pcm_length,
                                  const float* __restrict__ wcos,
                                  const float* __restrict__ wsin,
                                  float* __restrict__ power, int chunks) {
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;   // 0..200
  const int frame = blockIdx.y * blockDim.y + threadIdx.y; // 0..3000
  const int chunk = blockIdx.z;                            // 0..chunks-1
  if (bin >= kFrequencyBins || frame >= kFrameCount || chunk >= chunks) return;
  pcm += static_cast<size_t>(chunk) * kChunkSamples;
  power += static_cast<size_t>(chunk) * kFrequencyBins * kFrameCount;
  const int start = frame * kHopLength - kCenterPad;
  float re = 0.0f, im = 0.0f;
  const float* cos_row = wcos + static_cast<size_t>(bin) * kFftLength;
  const float* sin_row = wsin + static_cast<size_t>(bin) * kFftLength;
  for (int n = 0; n < kFftLength; ++n) {
    const float sample = pcm[reflect_index(start + n, pcm_length)];
    re += sample * cos_row[n];
    im += sample * sin_row[n];
  }
  power[static_cast<size_t>(bin) * kFrameCount + frame] = re * re + im * im;
}

// One thread per (mel bin, frame): mel = filterbank^T @ power.
__global__ void mel_matmul_kernel(const float* __restrict__ filters,
                                  const float* __restrict__ power,
                                  float* __restrict__ mel, int chunks) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;   // 0..127
  const int frame = blockIdx.y * blockDim.y + threadIdx.y; // 0..3000
  const int chunk = blockIdx.z;
  if (m >= kMelBins || frame >= kFrameCount || chunk >= chunks) return;
  power += static_cast<size_t>(chunk) * kFrequencyBins * kFrameCount;
  mel += static_cast<size_t>(chunk) * kMelBins * kFrameCount;
  float sum = 0.0f;
  for (int k = 0; k < kFrequencyBins; ++k)
    sum += filters[static_cast<size_t>(k) * kMelBins + m] *
           power[static_cast<size_t>(k) * kFrameCount + frame];
  mel[static_cast<size_t>(m) * kFrameCount + frame] = sum;
}

}  // namespace

MelOutput compute_mel(const std::vector<float>& pcm) {
  return compute_mel_batched(pcm, 1);
}

MelOutput compute_mel_batched(const std::vector<float>& pcm, int chunks) {
  if (chunks <= 0 || pcm.size() != static_cast<size_t>(chunks) * kChunkSamples)
    throw ExecutionError("compute_mel_batched expects chunks * 480000 padded samples");

  const std::vector<double> filters_host = build_mel_filterbank();
  std::vector<float> wcos, wsin;
  build_windowed_trig_tables(&wcos, &wsin);

  const size_t power_count = static_cast<size_t>(kFrequencyBins) * kFrameCount;
  const size_t mel_count = static_cast<size_t>(kMelBins) * kFrameCount;
  std::vector<float> filters_gpu_host(filters_host.size());
  for (size_t i = 0; i < filters_host.size(); ++i)
    filters_gpu_host[i] = static_cast<float>(filters_host[i]);

  const size_t power_total = power_count * chunks;
  const size_t mel_total = mel_count * chunks;
  const size_t out_count = static_cast<size_t>(kMelBins) * kOutputFrames;
  MelOutput out;
  out.input_samples = kChunkSamples;
  float* d_pcm = nullptr;
  float* d_wcos = nullptr;
  float* d_wsin = nullptr;
  float* d_power = nullptr;
  float* d_filters = nullptr;
  float* d_mel = nullptr;
  std::vector<float> mel_gpu(mel_total);
  try {
    CUDA_CHECK(cudaMalloc(&d_pcm, pcm.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wcos, wcos.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wsin, wsin.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_power, power_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filters, filters_gpu_host.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mel, mel_total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pcm, pcm.data(), pcm.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wcos, wcos.data(), wcos.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wsin, wsin.data(), wsin.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filters, filters_gpu_host.data(),
                          filters_gpu_host.size() * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 threads(16, 16, 1);
    const dim3 stft_blocks((kFrequencyBins + 15) / 16, (kFrameCount + 15) / 16, chunks);
    stft_power_kernel<<<stft_blocks, threads>>>(d_pcm, kChunkSamples, d_wcos, d_wsin,
                                                d_power, chunks);
    CUDA_CHECK(cudaGetLastError());
    const dim3 mel_blocks((kMelBins + 15) / 16, (kFrameCount + 15) / 16, chunks);
    mel_matmul_kernel<<<mel_blocks, threads>>>(d_filters, d_power, d_mel, chunks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float* d_out = nullptr;
    float* d_logs = nullptr;
    float* d_bin_max = nullptr;
    float* d_chunk_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, out_count * chunks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logs, out_count * chunks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bin_max, static_cast<size_t>(chunks) * kMelBins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_chunk_max, static_cast<size_t>(chunks) * sizeof(float)));
    mel_log_max_kernel<<<dim3(1, kMelBins, chunks), 256>>>(d_mel, d_logs, d_bin_max, chunks);
    CUDA_CHECK(cudaGetLastError());
    mel_chunk_max_kernel<<<chunks, 1>>>(d_bin_max, d_chunk_max, chunks);
    CUDA_CHECK(cudaGetLastError());
    mel_normalize_kernel<<<dim3(1, kMelBins, chunks), 256>>>(d_logs, d_chunk_max, d_out,
                                                             chunks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_logs); cudaFree(d_bin_max); cudaFree(d_chunk_max);
    out.features.resize(out_count * chunks);
    CUDA_CHECK(cudaMemcpy(out.features.data(), d_out, out_count * chunks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    cudaFree(d_pcm); cudaFree(d_wcos); cudaFree(d_wsin);
    cudaFree(d_power); cudaFree(d_filters); cudaFree(d_mel);
    return out;
  } catch (...) {
    cudaFree(d_pcm); cudaFree(d_wcos); cudaFree(d_wsin);
    cudaFree(d_power); cudaFree(d_filters); cudaFree(d_mel);
    throw;
  }
  cudaFree(d_pcm); cudaFree(d_wcos); cudaFree(d_wsin);
  cudaFree(d_power); cudaFree(d_filters); cudaFree(d_mel);

  return out;
}
