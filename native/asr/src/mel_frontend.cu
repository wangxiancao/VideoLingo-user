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
                                  float* __restrict__ power) {
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;   // 0..200
  const int frame = blockIdx.y * blockDim.y + threadIdx.y; // 0..3000
  if (bin >= kFrequencyBins || frame >= kFrameCount) return;
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
                                  float* __restrict__ mel) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;   // 0..127
  const int frame = blockIdx.y * blockDim.y + threadIdx.y; // 0..3000
  if (m >= kMelBins || frame >= kFrameCount) return;
  float sum = 0.0f;
  for (int k = 0; k < kFrequencyBins; ++k)
    sum += filters[static_cast<size_t>(k) * kMelBins + m] *
           power[static_cast<size_t>(k) * kFrameCount + frame];
  mel[static_cast<size_t>(m) * kFrameCount + frame] = sum;
}

}  // namespace

MelOutput compute_mel(const std::vector<float>& pcm) {
  if (pcm.size() != static_cast<size_t>(kChunkSamples))
    throw ExecutionError("compute_mel expects exactly 480000 padded samples");

  const std::vector<double> filters_host = build_mel_filterbank();
  std::vector<float> wcos, wsin;
  build_windowed_trig_tables(&wcos, &wsin);

  const size_t power_count = static_cast<size_t>(kFrequencyBins) * kFrameCount;
  const size_t mel_count = static_cast<size_t>(kMelBins) * kFrameCount;
  std::vector<float> filters_gpu_host(filters_host.size());
  for (size_t i = 0; i < filters_host.size(); ++i)
    filters_gpu_host[i] = static_cast<float>(filters_host[i]);

  float* d_pcm = nullptr;
  float* d_wcos = nullptr;
  float* d_wsin = nullptr;
  float* d_power = nullptr;
  float* d_filters = nullptr;
  float* d_mel = nullptr;
  std::vector<float> mel_gpu(mel_count);
  try {
    CUDA_CHECK(cudaMalloc(&d_pcm, pcm.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wcos, wcos.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wsin, wsin.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_power, power_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filters, filters_gpu_host.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mel, mel_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pcm, pcm.data(), pcm.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wcos, wcos.data(), wcos.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wsin, wsin.data(), wsin.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filters, filters_gpu_host.data(),
                          filters_gpu_host.size() * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 threads(16, 16);
    const dim3 stft_blocks((kFrequencyBins + 15) / 16, (kFrameCount + 15) / 16);
    stft_power_kernel<<<stft_blocks, threads>>>(d_pcm, kChunkSamples, d_wcos, d_wsin, d_power);
    CUDA_CHECK(cudaGetLastError());
    const dim3 mel_blocks((kMelBins + 15) / 16, (kFrameCount + 15) / 16);
    mel_matmul_kernel<<<mel_blocks, threads>>>(d_filters, d_power, d_mel);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(mel_gpu.data(), d_mel, mel_count * sizeof(float), cudaMemcpyDeviceToHost));
  } catch (...) {
    cudaFree(d_pcm); cudaFree(d_wcos); cudaFree(d_wsin);
    cudaFree(d_power); cudaFree(d_filters); cudaFree(d_mel);
    throw;
  }
  cudaFree(d_pcm); cudaFree(d_wcos); cudaFree(d_wsin);
  cudaFree(d_power); cudaFree(d_filters); cudaFree(d_mel);

  // Post-processing on the host in double precision, mirroring the reference:
  // floor -> log10 -> drop last frame -> clamp to global max - 8 -> (x+4)/4.
  MelOutput out;
  out.input_samples = kChunkSamples;
  out.features.resize(static_cast<size_t>(kMelBins) * kOutputFrames);
  double global_max = -std::numeric_limits<double>::infinity();
  std::vector<double> log_spec(static_cast<size_t>(kMelBins) * kOutputFrames);
  for (int m = 0; m < kMelBins; ++m) {
    for (int f = 0; f < kOutputFrames; ++f) {
      const double value = std::max(1e-10, static_cast<double>(mel_gpu[static_cast<size_t>(m) * kFrameCount + f]));
      const double logged = std::log10(value);
      log_spec[static_cast<size_t>(m) * kOutputFrames + f] = logged;
      global_max = std::max(global_max, logged);
    }
  }
  const double floor_log = global_max - 8.0;
  for (size_t i = 0; i < log_spec.size(); ++i)
    out.features[i] = static_cast<float>((std::max(log_spec[i], floor_log) + 4.0) / 4.0);
  return out;
}
