#include "engine.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "model_loader.h"

namespace {

// Fixed official GLM-ASR-Nano-2512 architecture (validated against config.json
// by the loader's inventory check).
constexpr int kMelBins = 128;
constexpr int kMelFrames = 3000;
constexpr int kFrames = 1500;
constexpr int kHidden = 1280;
constexpr int kLayers = 32;
constexpr int kHeads = 20;
constexpr int kHeadDim = 64;
constexpr int kRotaryDim = 32;   // partial_rotary_factor 0.5 * 64
constexpr int kIntermediate = 5120;
constexpr int kProjMid = 4096;
constexpr int kTextHidden = 2048;
constexpr int kTokens = kFrames / 4;  // 375
constexpr float kLnEps = 1e-5f;

using ExecutionError = ModelExecutionError;

std::string cuda_error(cudaError_t status, const char* expression) {
  std::ostringstream out;
  out << expression << " failed: " << cudaGetErrorString(status);
  return out.str();
}

std::string cublas_error(cublasStatus_t status, const char* expression) {
  std::ostringstream out;
  out << expression << " failed with cuBLAS status " << static_cast<int>(status);
  return out.str();
}

#define CUDA_CHECK(expr)                                                         \
  do {                                                                           \
    const cudaError_t status_ = (expr);                                          \
    if (status_ != cudaSuccess) throw ExecutionError(cuda_error(status_, #expr)); \
  } while (false)

#define CUBLAS_CHECK(expr)                                                       \
  do {                                                                           \
    const cublasStatus_t status_ = (expr);                                       \
    if (status_ != CUBLAS_STATUS_SUCCESS) throw ExecutionError(cublas_error(status_, #expr)); \
  } while (false)

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) {
    CUDA_CHECK(cudaMalloc(&data_, bytes));
  }
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_) { other.data_ = nullptr; }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    reset();
    data_ = other.data_;
    other.data_ = nullptr;
    return *this;
  }
  void reset() {
    if (data_) cudaFree(data_);
    data_ = nullptr;
  }
  half* get() const { return reinterpret_cast<half*>(data_); }
  size_t size() const { return bytes_; }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
};

class CublasHandle {
 public:
  CublasHandle() {
    CUBLAS_CHECK(cublasCreate(&handle_));
    // Keep FP16 GEMMs on the tensor-core path with predictable numerics.
    CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
  }
  ~CublasHandle() {
    if (handle_) cublasDestroy(handle_);
  }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

__device__ __forceinline__ float gelu(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// out[c, t] = gelu(bias[c] + sum_{cin,k} W[c,cin,k] * in[cin, t - 1 + k])
__global__ void conv1_gelu_kernel(const half* __restrict__ in, const half* __restrict__ w,
                                  const half* __restrict__ bias, half* __restrict__ out,
                                  int in_channels, int frames, int chunks) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;   // 0..1279
  const int t = blockIdx.y * blockDim.y + threadIdx.y;   // 0..2999
  const int chunk = blockIdx.z;
  if (c >= kHidden || t >= frames || chunk >= chunks) return;
  in += static_cast<size_t>(chunk) * in_channels * frames;
  out += static_cast<size_t>(chunk) * kHidden * frames;
  float acc = __half2float(bias[c]);
  for (int cin = 0; cin < in_channels; ++cin) {
    const half* w_row = w + (static_cast<size_t>(c) * in_channels + cin) * 3;
    const float w0 = __half2float(w_row[0]);
    const float w1 = __half2float(w_row[1]);
    const float w2 = __half2float(w_row[2]);
    const size_t base = static_cast<size_t>(cin) * frames + t;
    if (t >= 1) acc += w0 * __half2float(in[base - 1]);
    acc += w1 * __half2float(in[base]);
    if (t + 1 < frames) acc += w2 * __half2float(in[base + 1]);
  }
  out[static_cast<size_t>(c) * frames + t] = __float2half(gelu(acc));
}

// Stride-2 conv writing the transposed layout hidden[t, c] directly.
__global__ void conv2_gelu_kernel(const half* __restrict__ in, const half* __restrict__ w,
                                  const half* __restrict__ bias, half* __restrict__ hidden,
                                  int in_channels, int in_frames, int chunks) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;   // 0..1279
  const int t = blockIdx.y * blockDim.y + threadIdx.y;   // 0..1499
  const int chunk = blockIdx.z;
  if (c >= kHidden || t >= kFrames || chunk >= chunks) return;
  in += static_cast<size_t>(chunk) * in_channels * in_frames;
  hidden += static_cast<size_t>(chunk) * kFrames * kHidden;
  const int center = 2 * t;  // window [2t-1, 2t, 2t+1] in the conv1 output
  float acc = __half2float(bias[c]);
  for (int cin = 0; cin < in_channels; ++cin) {
    const half* w_row = w + (static_cast<size_t>(c) * in_channels + cin) * 3;
    for (int k = 0; k < 3; ++k) {
      const int idx = center - 1 + k;
      if (idx >= 0 && idx < in_frames)
        acc += __half2float(w_row[k]) * __half2float(in[static_cast<size_t>(cin) * in_frames + idx]);
    }
  }
  hidden[static_cast<size_t>(t) * kHidden + c] = __float2half(gelu(acc));
}

// Standard LayerNorm over the last dimension with weight and bias, fp32 math.
__global__ void layer_norm_kernel(half* __restrict__ x, const half* __restrict__ weight,
                                  const half* __restrict__ bias, int rows, int cols) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float* sums = shared;           // [blockDim.x]
  float* squares = shared + blockDim.x;
  half* row_ptr = x + static_cast<size_t>(row) * cols;
  float local_sum = 0.0f, local_sq = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = __half2float(row_ptr[i]);
    local_sum += v;
    local_sq += v * v;
  }
  sums[threadIdx.x] = local_sum;
  squares[threadIdx.x] = local_sq;
  __syncthreads();
  float mean = 0.0f, sq_mean = 0.0f;
  for (int i = 0; i < blockDim.x; ++i) {
    mean += sums[i];
    sq_mean += squares[i];
  }
  mean /= cols;
  sq_mean /= cols;
  const float var = sq_mean - mean * mean;
  const float inv_std = rsqrtf(var + kLnEps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float normalized = (__half2float(row_ptr[i]) - mean) * inv_std;
    row_ptr[i] = __float2half(normalized * __half2float(weight[i]) + __half2float(bias[i]));
  }
}

__global__ void gelu_kernel(half* __restrict__ x, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  x[i] = __float2half(gelu(__half2float(x[i])));
}

__global__ void add_bias_kernel(half* __restrict__ x, const half* __restrict__ bias,
                                size_t rows, int cols) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  x[i] = __float2half(__half2float(x[i]) + __half2float(bias[i % cols]));
}

__global__ void residual_add_kernel(half* __restrict__ x, const half* __restrict__ delta,
                                    size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  x[i] = __float2half(__half2float(x[i]) + __half2float(delta[i]));
}

// Partial RoPE: per (row, frequency), rotate the first kRotaryDim dims of
// every head. The reference's cat(freqs, freqs) makes dims d and d+16 share
// one frequency, so the table has kRotaryDim columns and 16 distinct values.
__global__ void rope_kernel(half* __restrict__ x, const float* __restrict__ cos_t,
                            const float* __restrict__ sin_t, int rows, int frames) {
  const int row = blockIdx.y;
  const int freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || freq >= kRotaryDim / 2) return;
  const int position = row % frames;
  const float cos_v = cos_t[static_cast<size_t>(position) * kRotaryDim + freq];
  const float sin_v = sin_t[static_cast<size_t>(position) * kRotaryDim + freq];
  half* row_ptr = x + static_cast<size_t>(row) * (kHeads * kHeadDim);
  for (int h = 0; h < kHeads; ++h) {
    half* head_ptr = row_ptr + static_cast<size_t>(h) * kHeadDim;
    const float x1 = __half2float(head_ptr[freq]);
    const float x2 = __half2float(head_ptr[freq + kRotaryDim / 2]);
    head_ptr[freq] = __float2half(x1 * cos_v - x2 * sin_v);
    head_ptr[freq + kRotaryDim / 2] = __float2half(x2 * cos_v + x1 * sin_v);
  }
}




// ---------------------------------------------------------------------------
// Flash-style fused attention (wmma tensor cores, online softmax). Reads
// head-major Q/K/V [batch, seq, dim] (batch = chunks*heads), writes head-major
// output; never materializes the score matrix in HBM. Optional causal mask
// for the decoder prefill. Scale = 1/sqrt(dim) applied inside.
// One block per (query tile of 64, batch); 4 warps, 16 query rows per warp.
// ---------------------------------------------------------------------------
#include <mma.h>

constexpr int kFlashTile = 64;

__global__ void flash_attention_kernel(const half* __restrict__ q,  // [b, s, d]
                                       const half* __restrict__ k,
                                       const half* __restrict__ v,
                                       half* __restrict__ out, int seq, int dim,
                                       int query_tiles, bool causal) {
  using namespace nvcuda;
  const int tile = blockIdx.x;        // query tile
  const int b = blockIdx.y;           // (chunk, head) batch index
  if (tile >= query_tiles) return;
  const float scale = 1.0f / sqrtf(static_cast<float>(dim));

  extern __shared__ char smem_raw[];
  half* sQ = reinterpret_cast<half*>(smem_raw);                  // [64, 64]
  half* sK = sQ + kFlashTile * kFlashTile;                       // [64, 64]
  half* sV = sK + kFlashTile * kFlashTile;                       // [64, 64]
  float* sS = reinterpret_cast<float*>(sV + kFlashTile * kFlashTile);  // [64, 64]
  float* sO = sS + kFlashTile * kFlashTile;                      // [64, 64]
  half* sP = reinterpret_cast<half*>(sO + kFlashTile * kFlashTile);    // [64, 64]
  float* sM = reinterpret_cast<float*>(sP + kFlashTile * kFlashTile);  // [64]
  float* sL = sM + kFlashTile;                                   // [64]

  const int q0 = tile * kFlashTile;
  const int tid = threadIdx.x;

  // Load the Q tile (zero-pad rows beyond seq).
  for (int i = tid; i < kFlashTile * kFlashTile; i += blockDim.x) {
    const int r = i / kFlashTile, c = i % kFlashTile;
    const int row = q0 + r;
    sQ[i] = (row < seq)
                ? q[(static_cast<size_t>(b) * seq + row) * dim + c]
                : __float2half(0.0f);
  }
  for (int r = tid; r < kFlashTile; r += blockDim.x) {
    sM[r] = -1e30f;
    sL[r] = 0.0f;
    for (int c = 0; c < kFlashTile; ++c) sO[r * kFlashTile + c] = 0.0f;
  }
  __syncthreads();

  const int key_tiles = (seq + kFlashTile - 1) / kFlashTile;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int row_base = warp * 16;  // this warp's 16 query rows

  for (int kt = 0; kt < key_tiles; ++kt) {
    if (causal && kt * kFlashTile > q0 + kFlashTile - 1) break;  // fully masked
    for (int i = tid; i < kFlashTile * kFlashTile; i += blockDim.x) {
      const int r = i / kFlashTile, c = i % kFlashTile;
      const int key = kt * kFlashTile + r;
      sK[i] = (key < seq) ? k[(static_cast<size_t>(b) * seq + key) * dim + c]
                          : __float2half(0.0f);
      sV[i] = (key < seq) ? v[(static_cast<size_t>(b) * seq + key) * dim + c]
                          : __float2half(0.0f);
    }
    __syncthreads();

    // S = Q * K^T via wmma; each warp computes rows [row_base, +16).
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      for (int n0 = 0; n0 < kFlashTile; n0 += 16) {
        wmma::fill_fragment(acc, 0.0f);
        for (int k0 = 0; k0 < kFlashTile; k0 += 16) {
          wmma::load_matrix_sync(a, sQ + (row_base)*kFlashTile + k0, kFlashTile);
          // B: col-major load of sK[n0.., k0..] gives K^T[k0 dims, n0 keys]
          wmma::load_matrix_sync(bf, sK + n0 * kFlashTile + k0, kFlashTile);
          wmma::mma_sync(acc, a, bf, acc);
        }
        wmma::store_matrix_sync(sS + row_base * kFlashTile + n0, acc, kFlashTile,
                                wmma::mem_row_major);
      }
    }
    __syncthreads();

    // Online softmax: lane r of each warp owns query row row_base + r.
    // Applies scale, range/causal masks, running max/sum and O rescale.
    {
      const int row = row_base + lane;
      if (lane < 16) {  // one lane per row; this warp owns exactly 16 rows
        float tile_max = -1e30f;
        for (int c = 0; c < kFlashTile; ++c) {
          float value = sS[row * kFlashTile + c] * scale;
          if (kt * kFlashTile + c >= seq) value = -1e30f;
          if (causal && kt * kFlashTile + c > q0 + row) value = -1e30f;
          sS[row * kFlashTile + c] = value;
          tile_max = fmaxf(tile_max, value);
        }
        const float m_old = sM[row];
        const float m_new = fmaxf(m_old, tile_max);
        const float correction = (sL[row] > 0.0f) ? __expf(m_old - m_new) : 0.0f;
        float l_new = sL[row] * correction;
        for (int c = 0; c < kFlashTile; ++c) {
          const float p = __expf(sS[row * kFlashTile + c] - m_new);
          sP[row * kFlashTile + c] = __float2half(p);
          l_new += p;
        }
        for (int c = 0; c < kFlashTile; ++c)
          sO[row * kFlashTile + c] *= correction;
        sM[row] = m_new;
        sL[row] = l_new;
      }
    }
    __syncthreads();

    // O += P * V via wmma (P row-major A, V row-major B).
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      for (int n0 = 0; n0 < kFlashTile; n0 += 16) {
        // load current O as accumulator, then accumulate P*V
        wmma::load_matrix_sync(acc, sO + row_base * kFlashTile + n0, kFlashTile,
                               wmma::mem_row_major);
        for (int k0 = 0; k0 < kFlashTile; k0 += 16) {
          wmma::load_matrix_sync(a, sP + row_base * kFlashTile + k0, kFlashTile);
          wmma::load_matrix_sync(bf, sV + k0 * kFlashTile + n0, kFlashTile);
          wmma::mma_sync(acc, a, bf, acc);
        }
        wmma::store_matrix_sync(sO + row_base * kFlashTile + n0, acc, kFlashTile,
                                wmma::mem_row_major);
      }
    }
    __syncthreads();
  }

  // Normalize by l and store.
  for (int i = tid; i < kFlashTile * kFlashTile; i += blockDim.x) {
    const int r = i / kFlashTile, c = i % kFlashTile;
    const int row = q0 + r;
    if (row < seq) {
      const float l = sL[r];
      out[(static_cast<size_t>(b) * seq + row) * dim + c] =
          __float2half(l > 0.0f ? sO[r * kFlashTile + c] / l : 0.0f);
    }
  }
}

// im2col for the conv frontend. conv1 gathers [chunk,t] rows of 384 taps
// from mel [chunks,128,3000]; conv2 gathers [chunk,t] rows of 3840 taps from
// the (t,c)-layout conv1 output. Boundary taps stay zero (padding 1).
__global__ void im2col_conv1_kernel(const half* __restrict__ in, half* __restrict__ out,
                                    int chunks) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;   // 0..2999
  const int index = blockIdx.y * blockDim.y + threadIdx.y;  // chunk*128 + cin
  if (t >= kMelFrames || index >= chunks * kMelBins) return;
  const int chunk = index / kMelBins;
  const int cin = index % kMelBins;
  half* row = out + (static_cast<size_t>(chunk) * kMelFrames + t) * (kMelBins * 3);
  const half* src = in + static_cast<size_t>(chunk) * kMelBins * kMelFrames +
                    static_cast<size_t>(cin) * kMelFrames;
  row[cin * 3 + 0] = t >= 1 ? src[t - 1] : __float2half(0.0f);
  row[cin * 3 + 1] = src[t];
  row[cin * 3 + 2] = t + 1 < kMelFrames ? src[t + 1] : __float2half(0.0f);
}

__global__ void im2col_conv2_kernel(const half* __restrict__ in, half* __restrict__ out,
                                    int chunks) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;   // 0..1499
  const int index = blockIdx.y * blockDim.y + threadIdx.y;  // chunk*1280 + cin
  if (t >= kFrames || index >= chunks * kHidden) return;
  const int chunk = index / kHidden;
  const int cin = index % kHidden;
  half* row = out + (static_cast<size_t>(chunk) * kFrames + t) * (kHidden * 3);
  const half* src = in + static_cast<size_t>(chunk) * kMelFrames * kHidden;
  for (int k = 0; k < 3; ++k) {
    const int idx = 2 * t - 1 + k;
    row[cin * 3 + k] =
        (idx >= 0 && idx < kMelFrames)
            ? src[static_cast<size_t>(idx) * kHidden + cin]
            : __float2half(0.0f);
  }
}

// Head-major re-layout: [chunks,1500,20,64] -> [chunks,20,1500,64] (and the
// inverse for the attention output) so one uniform-stride batched GEMM covers
// every (chunk, head) pair.
__global__ void scatter_heads_kernel(const half* __restrict__ in, half* __restrict__ out,
                                     int rows, int chunks) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int hidden = kHeads * kHeadDim;
  const size_t total = static_cast<size_t>(rows) * hidden;
  if (i >= total) return;
  const int col = static_cast<int>(i % hidden);
  const int row = static_cast<int>(i / hidden);
  const int head = col / kHeadDim;
  const int dim = col % kHeadDim;
  const int frames = rows / chunks;
  const int chunk = row / frames;
  const int frame = row % frames;
  out[((static_cast<size_t>(chunk) * kHeads + head) * frames + frame) * kHeadDim + dim] =
      in[i];
}

__global__ void gather_heads_kernel(const half* __restrict__ in, half* __restrict__ out,
                                    int rows, int chunks) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int hidden = kHeads * kHeadDim;
  const size_t total = static_cast<size_t>(rows) * hidden;
  if (i >= total) return;
  const int col = static_cast<int>(i % hidden);
  const int row = static_cast<int>(i / hidden);
  const int head = col / kHeadDim;
  const int dim = col % kHeadDim;
  const int frames = rows / chunks;
  const int chunk = row / frames;
  const int frame = row % frames;
  out[i] =
      in[((static_cast<size_t>(chunk) * kHeads + head) * frames + frame) * kHeadDim + dim];
}

// Softmax over one contiguous query row of the row-major score buffer
// S(i, j) at [head * L * L + i * L + j]; the scores GEMM computes S^T via
// swapped q/k operands so each query row lands contiguous in memory.
__global__ void softmax_rows_kernel(half* __restrict__ scores, int heads, int length) {
  const int query = blockIdx.x;
  const int head = blockIdx.y;
  if (query >= length || head >= heads) return;
  half* row = scores + static_cast<size_t>(head) * length * length +
              static_cast<size_t>(query) * length;
  __shared__ float shared[256];
  float local_max = -1e30f;
  for (int j = threadIdx.x; j < length; j += blockDim.x)
    local_max = fmaxf(local_max, __half2float(row[j]));
  shared[threadIdx.x] = local_max;
  __syncthreads();
  float maximum = -1e30f;
  for (int i = 0; i < blockDim.x; ++i) maximum = fmaxf(maximum, shared[i]);
  float local_sum = 0.0f;
  for (int j = threadIdx.x; j < length; j += blockDim.x)
    local_sum += __expf(__half2float(row[j]) - maximum);
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  float total = 0.0f;
  for (int i = 0; i < blockDim.x; ++i) total += shared[i];
  const float inv = 1.0f / total;
  for (int j = threadIdx.x; j < length; j += blockDim.x)
    row[j] = __float2half(__expf(__half2float(row[j]) - maximum) * inv);
}

const float kOneF32 = 1.0f;
const float kZeroF32 = 0.0f;
const half kZeroHalf = __float2half(0.0f);

// y = x * W^T (+ bias), row-major [m,k] x [n,k]^T -> [m,n], FP16 in/out.
void linear(CublasHandle& handle, bool acc16, const half* w, const half* bias,
            const half* x, half* y, int m, int n, int k) {
  const half alpha16 = __float2half(1.0f);
  const half beta16 = __float2half(0.0f);
  const float alpha32 = 1.0f, beta32 = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
      acc16 ? static_cast<const void*>(&alpha16) : static_cast<const void*>(&alpha32),
      w, CUDA_R_16F, k, x, CUDA_R_16F, k,
      acc16 ? static_cast<const void*>(&beta16) : static_cast<const void*>(&beta32),
      y, CUDA_R_16F, n,
      acc16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  if (bias) {
    const size_t count = static_cast<size_t>(m) * n;
    add_bias_kernel<<<static_cast<unsigned>((count + 255) / 256), 256>>>(y, bias, m, n);
    CUDA_CHECK(cudaGetLastError());
  }
}

void to_host_float(const half* device, size_t count, float* out) {
  std::vector<half> host(count);
  CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(half), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) out[i] = __half2float(host[i]);
}

std::vector<float> build_rope_table(bool cosine) {
  std::vector<float> table(static_cast<size_t>(kFrames) * kRotaryDim);
  const int half_rot = kRotaryDim / 2;
  for (int i = 0; i < kFrames; ++i) {
    for (int d = 0; d < kRotaryDim; ++d) {
      const double freq = 1.0 / std::pow(10000.0,
          static_cast<double>((d % half_rot) * 2) / kRotaryDim);
      const double angle = freq * i;
      table[static_cast<size_t>(i) * kRotaryDim + d] =
          static_cast<float>(cosine ? std::cos(angle) : std::sin(angle));
    }
  }
  return table;
}

}  // namespace

DeviceWeights::DeviceWeights(DeviceWeights&& other) noexcept
    : arena_(other.arena_), bytes_(other.bytes_), entries_(std::move(other.entries_)) {
  other.arena_ = nullptr;
  other.bytes_ = 0;
}

DeviceWeights& DeviceWeights::operator=(DeviceWeights&& other) noexcept {
  if (this != &other) {
    if (arena_) cudaFree(arena_);
    arena_ = other.arena_;
    bytes_ = other.bytes_;
    entries_ = std::move(other.entries_);
    other.arena_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

DeviceWeights::~DeviceWeights() {
  if (arena_) cudaFree(arena_);
}

const DeviceWeights::Entry* DeviceWeights::find(const std::string& name) const {
  const auto it = std::lower_bound(
      entries_.begin(), entries_.end(), name,
      [](const std::pair<std::string, Entry>& e, const std::string& n) { return e.first < n; });
  if (it == entries_.end() || it->first != name) return nullptr;
  return &it->second;
}

EncodeOutput run_encode(const std::vector<float>& mel_features, DeviceWeights* weights,
                        const EncodeOptions& options, uint16_t* projector_device_out) {
  if (mel_features.size() != static_cast<size_t>(kMelBins) * kMelFrames)
    throw ExecutionError("run_encode expects mel features [128,3000]");
  if (!weights->loaded()) throw ExecutionError("run_encode requires loaded weights");

  EncodeOutput out;
  CublasHandle cublas;

  const size_t mel_count = mel_features.size();
  const size_t hidden_count = static_cast<size_t>(kFrames) * kHidden;
  const size_t mlp_count = static_cast<size_t>(kFrames) * kIntermediate;
  const size_t scores_count = static_cast<size_t>(kHeads) * kFrames * kFrames;

  DeviceBuffer d_mel(mel_count * sizeof(half));
  {
    std::vector<half> host(mel_count);
    for (size_t i = 0; i < mel_count; ++i) host[i] = __float2half(mel_features[i]);
    CUDA_CHECK(cudaMemcpy(d_mel.get(), host.data(), mel_count * sizeof(half),
                          cudaMemcpyHostToDevice));
  }
  DeviceBuffer d_conv(kHidden * kMelFrames * sizeof(half));
  DeviceBuffer d_hidden(hidden_count * sizeof(half));
  DeviceBuffer d_ln(hidden_count * sizeof(half));
  DeviceBuffer d_q(hidden_count * sizeof(half));
  DeviceBuffer d_k(hidden_count * sizeof(half));
  DeviceBuffer d_v(hidden_count * sizeof(half));
  DeviceBuffer d_attn_out(hidden_count * sizeof(half));
  DeviceBuffer d_scores(scores_count * sizeof(half));
  DeviceBuffer d_scratch(hidden_count * sizeof(half));  // GEMM output scratch
  DeviceBuffer d_mlp(mlp_count * sizeof(half));
  DeviceBuffer d_proj(static_cast<size_t>(kTokens) * kProjMid * sizeof(half));
  DeviceBuffer d_proj_out(static_cast<size_t>(kTokens) * kTextHidden * sizeof(half));
  DeviceBuffer d_cos(build_rope_table(true).size() * sizeof(float));
  DeviceBuffer d_sin(build_rope_table(false).size() * sizeof(float));
  {
    const std::vector<float> cos_t = build_rope_table(true);
    const std::vector<float> sin_t = build_rope_table(false);
    CUDA_CHECK(cudaMemcpy(d_cos.get(), cos_t.data(), cos_t.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin.get(), sin_t.data(), sin_t.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  const auto weight = [&](const std::string& name) -> const half* {
    const DeviceWeights::Entry* entry = weights->find(name);
    if (!entry) throw ExecutionError("missing resident tensor: " + name);
    return reinterpret_cast<const half*>(entry->ptr);
  };

  // Conv frontend.
  {
    const dim3 threads(16, 16);
    const dim3 blocks1((kHidden + 15) / 16, (kMelFrames + 15) / 16, 1);
    conv1_gelu_kernel<<<blocks1, threads>>>(d_mel.get(), weight("audio_tower.conv1.weight"),
                                            weight("audio_tower.conv1.bias"), d_conv.get(),
                                            kMelBins, kMelFrames, 1);
    CUDA_CHECK(cudaGetLastError());
    const dim3 blocks2((kHidden + 15) / 16, (kFrames + 15) / 16, 1);
    conv2_gelu_kernel<<<blocks2, threads>>>(d_conv.get(), weight("audio_tower.conv2.weight"),
                                            weight("audio_tower.conv2.bias"), d_hidden.get(),
                                            kHidden, kMelFrames, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    out.conv_hidden.resize(hidden_count);
    to_host_float(d_hidden.get(), hidden_count, out.conv_hidden.data());
  }

  // 32 pre-LN encoder layers with partial-RoPE attention and GELU MLP.
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  for (int layer = 0; layer < kLayers; ++layer) {
    const std::string prefix = "audio_tower.layers." + std::to_string(layer) + ".";
    CUDA_CHECK(cudaMemcpy(d_ln.get(), d_hidden.get(), hidden_count * sizeof(half),
                          cudaMemcpyDeviceToDevice));
    layer_norm_kernel<<<kFrames, 256, 2 * 256 * sizeof(float)>>>(
        d_ln.get(), weight(prefix + "input_layernorm.weight"),
        weight(prefix + "input_layernorm.bias"), kFrames, kHidden);
    CUDA_CHECK(cudaGetLastError());

    linear(cublas, options.fp16_accumulate, weight(prefix + "self_attn.q_proj.weight"),
           weight(prefix + "self_attn.q_proj.bias"), d_ln.get(), d_q.get(), kFrames, kHidden,
           kHidden);
    linear(cublas, options.fp16_accumulate, weight(prefix + "self_attn.k_proj.weight"), nullptr,
           d_ln.get(), d_k.get(), kFrames, kHidden, kHidden);
    linear(cublas, options.fp16_accumulate, weight(prefix + "self_attn.v_proj.weight"),
           weight(prefix + "self_attn.v_proj.bias"), d_ln.get(), d_v.get(), kFrames, kHidden,
           kHidden);

    {
      const dim3 blocks((kRotaryDim / 2 + 63) / 64, kFrames);
      rope_kernel<<<blocks, 64>>>(d_q.get(), reinterpret_cast<const float*>(d_cos.get()),
                                  reinterpret_cast<const float*>(d_sin.get()), kFrames, kFrames);
      CUDA_CHECK(cudaGetLastError());
      rope_kernel<<<blocks, 64>>>(d_k.get(), reinterpret_cast<const float*>(d_cos.get()),
                                  reinterpret_cast<const float*>(d_sin.get()), kFrames, kFrames);
      CUDA_CHECK(cudaGetLastError());
    }

    if (layer == 0 && getenv("M0_DEBUG_LAYER0")) {
      const std::string dir = getenv("M0_DEBUG_LAYER0");
      const auto dump = [&](const char* name, const half* ptr, size_t count) {
        std::vector<float> host(count);
        to_host_float(ptr, count, host.data());
        std::ofstream out(dir + "/" + name, std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()),
                  static_cast<std::streamsize>(host.size() * sizeof(float)));
      };
      CUDA_CHECK(cudaDeviceSynchronize());
      dump("dbg_q.f32", d_q.get(), hidden_count);
      dump("dbg_k.f32", d_k.get(), hidden_count);
      dump("dbg_v.f32", d_v.get(), hidden_count);
    }

    // scores[head] = (1/sqrt(d)) * q_head * k_head^T; the col-major result
    // holds S(i, j) at [i + j * kFrames].
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas.get(), CUBLAS_OP_T, CUBLAS_OP_N, kFrames, kFrames, kHeadDim,
        static_cast<const void*>(&attn_scale), d_k.get(), CUDA_R_16F, kHeads * kHeadDim,
        kHeadDim, d_q.get(), CUDA_R_16F, kHeads * kHeadDim, kHeadDim,
        static_cast<const void*>(&kZeroF32), d_scores.get(), CUDA_R_16F, kFrames,
        static_cast<long long>(kFrames) * kFrames, kHeads, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
    if (layer == 0 && getenv("M0_DEBUG_LAYER0")) {
      const std::string dir = getenv("M0_DEBUG_LAYER0");
      const auto dump = [&](const char* name, const half* ptr, size_t count) {
        std::vector<float> host(count);
        to_host_float(ptr, count, host.data());
        std::ofstream out(dir + "/" + name, std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()),
                  static_cast<std::streamsize>(host.size() * sizeof(float)));
      };
      CUDA_CHECK(cudaDeviceSynchronize());
      dump("dbg_scores_pre.f32", d_scores.get(), scores_count);
    }
    softmax_rows_kernel<<<dim3(kFrames, kHeads), 256>>>(d_scores.get(), kHeads, kFrames);
    CUDA_CHECK(cudaGetLastError());

    // attn[head] = softmax(S) * v_head, written back into the concat layout.
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, kFrames, kFrames,
        static_cast<const void*>(&kOneF32), d_v.get(), CUDA_R_16F, kHeads * kHeadDim,
        kHeadDim, d_scores.get(), CUDA_R_16F, kFrames,
        static_cast<long long>(kFrames) * kFrames,
        static_cast<const void*>(&kZeroHalf), d_attn_out.get(), CUDA_R_16F, kHeads * kHeadDim,
        kHeadDim, kHeads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    if (layer == 0 && getenv("M0_DEBUG_LAYER0")) {
      const std::string dir = getenv("M0_DEBUG_LAYER0");
      const auto dump = [&](const char* name, const half* ptr, size_t count) {
        std::vector<float> host(count);
        to_host_float(ptr, count, host.data());
        std::ofstream out(dir + "/" + name, std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()),
                  static_cast<std::streamsize>(host.size() * sizeof(float)));
      };
      CUDA_CHECK(cudaDeviceSynchronize());
      dump("dbg_scores_post.f32", d_scores.get(), scores_count);
      dump("dbg_attn_pv.f32", d_attn_out.get(), hidden_count);
    }
    linear(cublas, options.fp16_accumulate, weight(prefix + "self_attn.o_proj.weight"),
           weight(prefix + "self_attn.o_proj.bias"), d_attn_out.get(), d_scratch.get(),
           kFrames, kHidden, kHidden);
    CUDA_CHECK(cudaMemcpy(d_attn_out.get(), d_scratch.get(), hidden_count * sizeof(half),
                          cudaMemcpyDeviceToDevice));
    residual_add_kernel<<<static_cast<unsigned>((hidden_count + 255) / 256), 256>>>(
        d_hidden.get(), d_attn_out.get(), hidden_count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(d_ln.get(), d_hidden.get(), hidden_count * sizeof(half),
                          cudaMemcpyDeviceToDevice));
    layer_norm_kernel<<<kFrames, 256, 2 * 256 * sizeof(float)>>>(
        d_ln.get(), weight(prefix + "post_attention_layernorm.weight"),
        weight(prefix + "post_attention_layernorm.bias"), kFrames, kHidden);
    CUDA_CHECK(cudaGetLastError());
    linear(cublas, options.fp16_accumulate, weight(prefix + "mlp.fc1.weight"),
           weight(prefix + "mlp.fc1.bias"), d_ln.get(), d_mlp.get(), kFrames, kIntermediate,
           kHidden);
    gelu_kernel<<<static_cast<unsigned>((mlp_count + 255) / 256), 256>>>(d_mlp.get(), mlp_count);
    CUDA_CHECK(cudaGetLastError());
    linear(cublas, options.fp16_accumulate, weight(prefix + "mlp.fc2.weight"),
           weight(prefix + "mlp.fc2.bias"), d_mlp.get(), d_attn_out.get(), kFrames, kHidden,
           kIntermediate);
    residual_add_kernel<<<static_cast<unsigned>((hidden_count + 255) / 256), 256>>>(
        d_hidden.get(), d_attn_out.get(), hidden_count);
    CUDA_CHECK(cudaGetLastError());

    out.layer_hidden.resize(hidden_count * (layer + 1));
    to_host_float(d_hidden.get(), hidden_count,
                  out.layer_hidden.data() + hidden_count * layer);
  }

  // Final LayerNorm and the frame-merge projector ([1500,1280] -> [375,5120]
  // is a contiguous reshape, no data movement).
  layer_norm_kernel<<<kFrames, 256, 2 * 256 * sizeof(float)>>>(
      d_hidden.get(), weight("audio_tower.norm.weight"),
      weight("audio_tower.norm.bias"), kFrames, kHidden);
  CUDA_CHECK(cudaGetLastError());
  out.final_norm.resize(hidden_count);
  to_host_float(d_hidden.get(), hidden_count, out.final_norm.data());

  linear(cublas, options.fp16_accumulate, weight("multi_modal_projector.linear_1.weight"),
         weight("multi_modal_projector.linear_1.bias"), d_hidden.get(), d_proj.get(), kTokens,
         kProjMid, kHidden * 4);
  gelu_kernel<<<static_cast<unsigned>((static_cast<size_t>(kTokens) * kProjMid + 255) / 256),
               256>>>(d_proj.get(), static_cast<size_t>(kTokens) * kProjMid);
  CUDA_CHECK(cudaGetLastError());
  linear(cublas, options.fp16_accumulate, weight("multi_modal_projector.linear_2.weight"),
         weight("multi_modal_projector.linear_2.bias"), d_proj.get(), d_proj_out.get(), kTokens,
         kTextHidden, kProjMid);
  out.projector.resize(static_cast<size_t>(kTokens) * kTextHidden);
  if (projector_device_out) {
    CUDA_CHECK(cudaMemcpy(projector_device_out, d_proj_out.get(),
                          static_cast<size_t>(kTokens) * kTextHidden * sizeof(half),
                          cudaMemcpyDeviceToDevice));
  }
  to_host_float(d_proj_out.get(), static_cast<size_t>(kTokens) * kTextHidden, out.projector.data());
  CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

// ---------------------------------------------------------------------------
// Batched audio tower + projector: mel [chunks,128,3000] -> projector
// [chunks*375, 2048] device FP16. Row-wise stages run at m = chunks*1500;
// attention loops chunks with a single-chunk score buffer to keep VRAM flat.
// EncodeOutput (optional) receives chunk-0 stage dumps for gating.
// ---------------------------------------------------------------------------

EncodeOutput run_encode_batched(const std::vector<float>& mel_features, int chunks,
                                DeviceWeights* weights, const EncodeOptions& options,
                                uint16_t* projector_device_out) {
  constexpr int kMaxBatchChunks = 8;
  if (chunks <= 0 || chunks > kMaxBatchChunks)
    throw ExecutionError("run_encode_batched supports 1..8 chunks");
  if (mel_features.size() != static_cast<size_t>(chunks) * kMelBins * kMelFrames)
    throw ExecutionError("run_encode_batched expects mel [chunks,128,3000]");
  if (!weights->loaded()) throw ExecutionError("run_encode_batched requires loaded weights");

  EncodeOutput out;
  CublasHandle cublas;
  const int rows = chunks * kFrames;
  const size_t mel_count = static_cast<size_t>(chunks) * kMelBins * kMelFrames;
  const size_t hidden_count = static_cast<size_t>(rows) * kHidden;
  const size_t mlp_count = static_cast<size_t>(rows) * kIntermediate;
  const size_t chunk_hidden = static_cast<size_t>(kFrames) * kHidden;
  const size_t score_count = static_cast<size_t>(kHeads) * kFrames * kFrames;

  // Persistent workspace: sized once for the 8-chunk group ceiling and reused
  // across calls so repeated groups do not pay cudaMalloc/free per invocation.
  constexpr int kMaxGroup = 8;
  struct Workspace {
    DeviceBuffer mel, conv, hidden, ln, q, k, v, attn, scratch, scores, q_hm, k_hm, v_hm,
        attn_hm, mlp, proj, proj_out, cos, sin, im2col_1, im2col_2;
    bool ready = false;
  };
  static Workspace ws;
  const size_t max_rows = static_cast<size_t>(kMaxGroup) * kFrames;
  const size_t max_hidden = max_rows * kHidden;
  if (!ws.ready) {
    ws.mel = DeviceBuffer(static_cast<size_t>(kMaxGroup) * kMelBins * kMelFrames * sizeof(half));
    ws.conv = DeviceBuffer(static_cast<size_t>(kMaxGroup) * kHidden * kMelFrames * sizeof(half));
    ws.hidden = DeviceBuffer(max_hidden * sizeof(half));
    ws.ln = DeviceBuffer(max_hidden * sizeof(half));
    ws.q = DeviceBuffer(max_hidden * sizeof(half));
    ws.k = DeviceBuffer(max_hidden * sizeof(half));
    ws.v = DeviceBuffer(max_hidden * sizeof(half));
    ws.attn = DeviceBuffer(max_hidden * sizeof(half));
    ws.scratch = DeviceBuffer(max_hidden * sizeof(half));
    ws.scores = DeviceBuffer(static_cast<size_t>(kMaxGroup) *
                             static_cast<size_t>(kHeads) * kFrames * kFrames * sizeof(half));
    ws.q_hm = DeviceBuffer(max_hidden * sizeof(half));
    ws.k_hm = DeviceBuffer(max_hidden * sizeof(half));
    ws.v_hm = DeviceBuffer(max_hidden * sizeof(half));
    ws.attn_hm = DeviceBuffer(max_hidden * sizeof(half));
    ws.mlp = DeviceBuffer(static_cast<size_t>(kMaxGroup) * kFrames * kIntermediate * sizeof(half));
    ws.proj = DeviceBuffer(static_cast<size_t>(kMaxGroup) * kTokens * kProjMid * sizeof(half));
    ws.im2col_1 = DeviceBuffer(static_cast<size_t>(kMaxGroup) * kMelFrames * kMelBins * 3 *
                               sizeof(half));
    ws.im2col_2 =
        DeviceBuffer(static_cast<size_t>(kMaxGroup) * kFrames * kHidden * 3 * sizeof(half));
    ws.proj_out =
        DeviceBuffer(static_cast<size_t>(kMaxGroup) * kTokens * kTextHidden * sizeof(half));
    ws.cos = DeviceBuffer(build_rope_table(true).size() * sizeof(float));
    ws.sin = DeviceBuffer(build_rope_table(false).size() * sizeof(float));
    const std::vector<float> cos_t = build_rope_table(true);
    const std::vector<float> sin_t = build_rope_table(false);
    CUDA_CHECK(cudaMemcpy(ws.cos.get(), cos_t.data(), cos_t.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws.sin.get(), sin_t.data(), sin_t.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    ws.ready = true;
  }
  DeviceBuffer& d_mel = ws.mel;
  DeviceBuffer& d_conv = ws.conv;
  DeviceBuffer& d_hidden = ws.hidden;
  DeviceBuffer& d_ln = ws.ln;
  DeviceBuffer& d_q = ws.q;
  DeviceBuffer& d_k = ws.k;
  DeviceBuffer& d_v = ws.v;
  DeviceBuffer& d_attn = ws.attn;
  DeviceBuffer& d_scratch = ws.scratch;
  DeviceBuffer& d_scores = ws.scores;
  DeviceBuffer& d_q_hm = ws.q_hm;
  DeviceBuffer& d_k_hm = ws.k_hm;
  DeviceBuffer& d_v_hm = ws.v_hm;
  DeviceBuffer& d_attn_hm = ws.attn_hm;
  DeviceBuffer& d_mlp = ws.mlp;
  DeviceBuffer& d_proj = ws.proj;
  DeviceBuffer& d_proj_out = ws.proj_out;
  DeviceBuffer& d_cos = ws.cos;
  DeviceBuffer& d_sin = ws.sin;
  DeviceBuffer& d_im2col_1 = ws.im2col_1;
  DeviceBuffer& d_im2col_2 = ws.im2col_2;
  {
    std::vector<half> host(mel_count);
    for (size_t i = 0; i < mel_count; ++i) host[i] = __float2half(mel_features[i]);
    CUDA_CHECK(cudaMemcpy(d_mel.get(), host.data(), mel_count * sizeof(half),
                          cudaMemcpyHostToDevice));
  }

  const bool stage_timing = getenv("M2_STAGE_TIMING") != nullptr;
  float t_gemm = 0, t_attn = 0, t_elt = 0, t_cpy = 0;
  cudaEvent_t ev_a, ev_b;
  if (stage_timing) {
    cudaEventCreate(&ev_a);
    cudaEventCreate(&ev_b);
  }
  const auto tick = [&]() { if (stage_timing) cudaEventRecord(ev_a); };
  const auto tock = [&](float& bucket) {
    if (stage_timing) {
      cudaEventRecord(ev_b);
      cudaEventSynchronize(ev_b);
      float ms = 0;
      cudaEventElapsedTime(&ms, ev_a, ev_b);
      bucket += ms;
    }
  };
  struct StageTimings {
    bool on = false;
    float gemm = 0, attn = 0, elt = 0, cpy = 0;
  } st;
  st.on = stage_timing;
  const auto weight = [&](const std::string& name) -> const half* {
    const DeviceWeights::Entry* entry = weights->find(name);
    if (!entry) throw ExecutionError("missing resident tensor: " + name);
    return reinterpret_cast<const half*>(entry->ptr);
  };
  const auto linear_rows = [&](const half* w, const half* bias, const half* x, half* y,
                               int m, int n, int k) {
    const half alpha16 = __float2half(1.0f), beta16 = __float2half(0.0f);
    const float alpha32 = 1.0f, beta32 = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        options.fp16_accumulate ? static_cast<const void*>(&alpha16)
                                : static_cast<const void*>(&alpha32),
        w, CUDA_R_16F, k, x, CUDA_R_16F, k,
        options.fp16_accumulate ? static_cast<const void*>(&beta16)
                                : static_cast<const void*>(&beta32),
        y, CUDA_R_16F, n,
        options.fp16_accumulate ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
    if (bias) {
      const size_t count = static_cast<size_t>(m) * n;
      add_bias_kernel<<<static_cast<unsigned>((count + 255) / 256), 256>>>(y, bias, m, n);
      CUDA_CHECK(cudaGetLastError());
    }
  };

  {
    // Conv frontend via im2col + cuBLAS GEMM: the (t,c) GEMM output layout is
    // exactly what conv2's gather expects and what the encoder hidden needs.
    {
      const dim3 threads(16, 16);
      im2col_conv1_kernel<<<dim3((kMelFrames + 15) / 16, (chunks * kMelBins + 15) / 16),
                            threads>>>(d_mel.get(), d_im2col_1.get(), chunks);
      CUDA_CHECK(cudaGetLastError());
      linear_rows(weight("audio_tower.conv1.weight"), weight("audio_tower.conv1.bias"),
                  d_im2col_1.get(), d_conv.get(), chunks * kMelFrames, kHidden,
                  kMelBins * 3);
      const size_t conv1_count = static_cast<size_t>(chunks) * kMelFrames * kHidden;
      gelu_kernel<<<static_cast<unsigned>((conv1_count + 255) / 256), 256>>>(
          d_conv.get(), conv1_count);
      CUDA_CHECK(cudaGetLastError());
      im2col_conv2_kernel<<<dim3((kFrames + 15) / 16, (chunks * kHidden + 15) / 16), threads>>>(
          d_conv.get(), d_im2col_2.get(), chunks);
      CUDA_CHECK(cudaGetLastError());
      linear_rows(weight("audio_tower.conv2.weight"), weight("audio_tower.conv2.bias"),
                  d_im2col_2.get(), d_hidden.get(), chunks * kFrames, kHidden, kHidden * 3);
      const size_t conv2_count = static_cast<size_t>(chunks) * kFrames * kHidden;
      gelu_kernel<<<static_cast<unsigned>((conv2_count + 255) / 256), 256>>>(
          d_hidden.get(), conv2_count);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (options.dump_stages) {
      out.conv_hidden.resize(chunk_hidden);
      to_host_float(d_hidden.get(), chunk_hidden, out.conv_hidden.data());
    }
  }

  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  for (int layer = 0; layer < kLayers; ++layer) {
    const std::string prefix = "audio_tower.layers." + std::to_string(layer) + ".";
    tick();
    CUDA_CHECK(cudaMemcpy(d_ln.get(), d_hidden.get(), hidden_count * sizeof(half),
                          cudaMemcpyDeviceToDevice));
    tock(t_cpy);
    tick();
    layer_norm_kernel<<<rows, 256, 2 * 256 * sizeof(float)>>>(
        d_ln.get(), weight(prefix + "input_layernorm.weight"),
        weight(prefix + "input_layernorm.bias"), rows, kHidden);
    CUDA_CHECK(cudaGetLastError());
    tock(t_elt);
    tick();
    linear_rows(weight(prefix + "self_attn.q_proj.weight"),
                weight(prefix + "self_attn.q_proj.bias"), d_ln.get(), d_q.get(), rows, kHidden,
                kHidden);
    linear_rows(weight(prefix + "self_attn.k_proj.weight"), nullptr, d_ln.get(), d_k.get(),
                rows, kHidden, kHidden);
    linear_rows(weight(prefix + "self_attn.v_proj.weight"),
                weight(prefix + "self_attn.v_proj.bias"), d_ln.get(), d_v.get(), rows, kHidden,
                kHidden);
    tock(t_gemm);
    {
      tick();
      const dim3 blocks((kRotaryDim / 2 + 63) / 64, rows);
      rope_kernel<<<blocks, 64>>>(d_q.get(), reinterpret_cast<const float*>(d_cos.get()),
                                  reinterpret_cast<const float*>(d_sin.get()), rows, kFrames);
      CUDA_CHECK(cudaGetLastError());
      rope_kernel<<<blocks, 64>>>(d_k.get(), reinterpret_cast<const float*>(d_cos.get()),
                                  reinterpret_cast<const float*>(d_sin.get()), rows, kFrames);
      CUDA_CHECK(cudaGetLastError());
      tock(t_elt);
    }
    tick();
    {
      // head-major re-layout, then ONE uniform-stride batched call per stage
      // across every (chunk, head) pair in the group.
      const size_t total = hidden_count;
      scatter_heads_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(
          d_q.get(), d_q_hm.get(), rows, chunks);
      scatter_heads_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(
          d_k.get(), d_k_hm.get(), rows, chunks);
      scatter_heads_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(
          d_v.get(), d_v_hm.get(), rows, chunks);
      CUDA_CHECK(cudaGetLastError());
      const int batch = chunks * kHeads;
      const long long head_stride = static_cast<long long>(kFrames) * kHeadDim;
      const long long score_stride = static_cast<long long>(kFrames) * kFrames;
      // NOTE: the wmma flash_attention_kernel below is numerically correct but
      // currently slower than the materialized path (1 block/SM occupancy);
      // it stays in the tree for a tuning pass (M2 backlog).
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          cublas.get(), CUBLAS_OP_T, CUBLAS_OP_N, kFrames, kFrames, kHeadDim,
          static_cast<const void*>(&attn_scale), d_k_hm.get(), CUDA_R_16F, kHeadDim,
          head_stride, d_q_hm.get(), CUDA_R_16F, kHeadDim, head_stride,
          static_cast<const void*>(&kZeroF32), d_scores.get(), CUDA_R_16F, kFrames,
          score_stride, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      softmax_rows_kernel<<<dim3(kFrames, batch), 256>>>(d_scores.get(), batch, kFrames);
      CUDA_CHECK(cudaGetLastError());
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, kFrames, kFrames,
          static_cast<const void*>(&kOneF32), d_v_hm.get(), CUDA_R_16F, kHeadDim,
          head_stride, d_scores.get(), CUDA_R_16F, kFrames, score_stride,
          static_cast<const void*>(&kZeroHalf), d_attn_hm.get(), CUDA_R_16F, kHeadDim,
          head_stride, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      gather_heads_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(
          d_attn_hm.get(), d_attn.get(), rows, chunks);
      CUDA_CHECK(cudaGetLastError());
    }
    tock(t_attn);
    tick();
    linear_rows(weight(prefix + "self_attn.o_proj.weight"),
                weight(prefix + "self_attn.o_proj.bias"), d_attn.get(), d_scratch.get(), rows,
                kHidden, kHidden);
    tock(t_gemm);
    tick();
    CUDA_CHECK(cudaMemcpy(d_attn.get(), d_scratch.get(), hidden_count * sizeof(half),
                          cudaMemcpyDeviceToDevice));
    tock(t_cpy);
    tick();
    residual_add_kernel<<<static_cast<unsigned>((hidden_count + 255) / 256), 256>>>(
        d_hidden.get(), d_attn.get(), hidden_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_ln.get(), d_hidden.get(), hidden_count * sizeof(half),
                          cudaMemcpyDeviceToDevice));
    layer_norm_kernel<<<rows, 256, 2 * 256 * sizeof(float)>>>(
        d_ln.get(), weight(prefix + "post_attention_layernorm.weight"),
        weight(prefix + "post_attention_layernorm.bias"), rows, kHidden);
    CUDA_CHECK(cudaGetLastError());
    tock(t_elt);
    tick();
    linear_rows(weight(prefix + "mlp.fc1.weight"), weight(prefix + "mlp.fc1.bias"), d_ln.get(),
                d_mlp.get(), rows, kIntermediate, kHidden);
    tock(t_gemm);
    tick();
    gelu_kernel<<<static_cast<unsigned>((mlp_count + 255) / 256), 256>>>(d_mlp.get(),
                                                                         mlp_count);
    CUDA_CHECK(cudaGetLastError());
    tock(t_elt);
    tick();
    linear_rows(weight(prefix + "mlp.fc2.weight"), weight(prefix + "mlp.fc2.bias"),
                d_mlp.get(), d_attn.get(), rows, kHidden, kIntermediate);
    tock(t_gemm);
    tick();
    residual_add_kernel<<<static_cast<unsigned>((hidden_count + 255) / 256), 256>>>(
        d_hidden.get(), d_attn.get(), hidden_count);
    CUDA_CHECK(cudaGetLastError());
    tock(t_elt);

    if (options.dump_stages) {
      out.layer_hidden.resize(chunk_hidden * (layer + 1));
      to_host_float(d_hidden.get(), chunk_hidden,
                    out.layer_hidden.data() + chunk_hidden * layer);
    }
  }

  layer_norm_kernel<<<rows, 256, 2 * 256 * sizeof(float)>>>(
      d_hidden.get(), weight("audio_tower.norm.weight"), weight("audio_tower.norm.bias"), rows,
      kHidden);
  CUDA_CHECK(cudaGetLastError());
  if (options.dump_stages) {
    out.final_norm.resize(chunk_hidden);
    to_host_float(d_hidden.get(), chunk_hidden, out.final_norm.data());
  }

  linear_rows(weight("multi_modal_projector.linear_1.weight"),
              weight("multi_modal_projector.linear_1.bias"), d_hidden.get(), d_proj.get(),
              chunks * kTokens, kProjMid, kHidden * 4);
  const size_t proj_count = static_cast<size_t>(chunks) * kTokens * kProjMid;
  gelu_kernel<<<static_cast<unsigned>((proj_count + 255) / 256), 256>>>(d_proj.get(),
                                                                        proj_count);
  CUDA_CHECK(cudaGetLastError());
  linear_rows(weight("multi_modal_projector.linear_2.weight"),
              weight("multi_modal_projector.linear_2.bias"), d_proj.get(), d_proj_out.get(),
              chunks * kTokens, kTextHidden, kProjMid);
  if (projector_device_out) {
    CUDA_CHECK(cudaMemcpy(projector_device_out, d_proj_out.get(),
                          static_cast<size_t>(chunks) * kTokens * kTextHidden * sizeof(half),
                          cudaMemcpyDeviceToDevice));
  }
  if (options.dump_stages) {
    out.projector.resize(static_cast<size_t>(kTokens) * kTextHidden);
    to_host_float(d_proj_out.get(), static_cast<size_t>(kTokens) * kTextHidden,
                  out.projector.data());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  if (stage_timing)
    std::cerr << "stage_timing chunks=" << chunks << " gemm_ms=" << t_gemm
              << " attn_ms=" << t_attn << " elt_ms=" << t_elt << " cpy_ms=" << t_cpy
              << " (sum " << (t_gemm + t_attn + t_elt + t_cpy) << ")\n";
  return out;
}
