// M0 Task 4: native greedy decoder for the GLM-ASR text model (Llama-style,
// 28 layers, GQA 16/4, full RoPE, RMSNorm, SiLU MLP, untied lm_head).
//
// One incremental path serves prefill and decode: every prompt or generated
// token runs through all layers with a growing KV cache, so causality holds
// by construction. Token ids come from the caller (the tokenizer stays out
// of the native runtime); audio placeholder positions (id 59260) are fed the
// on-device projector outputs produced by the encoder stage.
//
// Correctness tool: reports generated ids only, no timing claims.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "model_loader.h"

namespace {

constexpr int kTextLayers = 28;
constexpr int kTextHeads = 16;
constexpr int kTextKvHeads = 4;
constexpr int kTextHeadDim = 128;
constexpr int kTextRotaryDim = 128;  // full rotary
constexpr int kAudioTokens = 375;
constexpr int kTextHidden = 2048;
constexpr int kVocab = 59264;
constexpr int kAudioTokenId = 59260;
constexpr float kRmsEps = 1e-5f;
constexpr int kMaxSequence = 1024;

using ExecutionError = ModelExecutionError;

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

// Builds the input embedding row for every sequence position: token
// embeddings, with audio placeholder rows taken from the projector output.
__global__ void gather_inputs_kernel(const int64_t* __restrict__ ids, int count,
                                     const half* __restrict__ embed,  // [vocab, hidden]
                                     const half* __restrict__ audio,  // [375, hidden]
                                     half* __restrict__ out           // [count, hidden]
) {
  const int pos = blockIdx.x;
  if (pos >= count) return;
  const int64_t id = ids[pos];
  const half* source;
  int audio_used = 0;
  for (int i = 0; i < pos; ++i) audio_used += ids[i] == kAudioTokenId ? 1 : 0;
  // The placeholder row uses this position's audio index (audio_used counts
  // placeholders strictly before pos, so this is the audio_used-th segment).
  source = (id == kAudioTokenId) ? audio + static_cast<size_t>(audio_used) * kTextHidden
                                 : embed + static_cast<size_t>(id) * kTextHidden;
  (void)embed;
  for (int d = threadIdx.x; d < kTextHidden; d += blockDim.x)
    out[static_cast<size_t>(pos) * kTextHidden + d] = source[d];
}

// RMSNorm over one row: x / sqrt(mean(x^2) + eps) * weight.
__global__ void rmsnorm_kernel(half* __restrict__ x, const half* __restrict__ weight,
                               int hidden) {
  extern __shared__ float shared[];
  float* partial = shared;               // [blockDim.x]
  float* total = shared + blockDim.x;
  const int row = blockIdx.x;
  half* row_ptr = x + static_cast<size_t>(row) * hidden;
  float local = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = __half2float(row_ptr[i]);
    local += v * v;
  }
  partial[threadIdx.x] = local;
  if (threadIdx.x == 0) *total = 0.0f;
  __syncthreads();
  for (int i = 0; i < blockDim.x; ++i) *total += partial[i];
  __syncthreads();
  const float inv = rsqrtf(*total / hidden + kRmsEps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    row_ptr[i] = __float2half(__half2float(row_ptr[i]) * inv * __half2float(weight[i]));
}

__global__ void silu_mul_kernel(half* __restrict__ gated, const half* __restrict__ up,
                                size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float g = __half2float(gated[i]);
  gated[i] = __float2half(g / (1.0f + __expf(-g)) * __half2float(up[i]));
}

// Full RoPE on one position's q [heads*128] and k [kv_heads*128]: pairs
// (d, d+64) share frequency d, for every head.
__global__ void rope_full_kernel(half* __restrict__ q, half* __restrict__ k,
                                 const float* __restrict__ cos_t,
                                 const float* __restrict__ sin_t, int position) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;  // 0..63 frequency pair
  if (d >= kTextRotaryDim / 2) return;
  const float c = cos_t[position * (kTextRotaryDim / 2) + d];
  const float s = sin_t[position * (kTextRotaryDim / 2) + d];
  for (int h = 0; h < kTextHeads; ++h) {
    half* ptr = q + static_cast<size_t>(h) * kTextHeadDim;
    const float x1 = __half2float(ptr[d]);
    const float x2 = __half2float(ptr[d + kTextRotaryDim / 2]);
    ptr[d] = __float2half(x1 * c - x2 * s);
    ptr[d + kTextRotaryDim / 2] = __float2half(x2 * c + x1 * s);
  }
  for (int h = 0; h < kTextKvHeads; ++h) {
    half* ptr = k + static_cast<size_t>(h) * kTextHeadDim;
    const float x1 = __half2float(ptr[d]);
    const float x2 = __half2float(ptr[d + kTextRotaryDim / 2]);
    ptr[d] = __float2half(x1 * c - x2 * s);
    ptr[d + kTextRotaryDim / 2] = __float2half(x2 * c + x1 * s);
  }
}




// ---- Prefill attention re-layout: group-major so each GQA group's
// (chunk, head-within-group) pairs are contiguous with uniform stride. ----
// q: concat [chunks*L, 2048] -> [4 groups][chunks][4 hq][L, 128]
__global__ void scatter_q_groupmajor_kernel(const half* __restrict__ in,
                                             half* __restrict__ out, int rows, int chunks) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int hidden = kTextHeads * kTextHeadDim;
  const size_t total = static_cast<size_t>(rows) * hidden;
  if (i >= total) return;
  const int col = static_cast<int>(i % hidden);
  const int row = static_cast<int>(i / hidden);
  const int head = col / kTextHeadDim;
  const int dim = col % kTextHeadDim;
  const int frames = rows / chunks;
  const int chunk = row / frames;
  const int frame = row % frames;
  const int group = head / (kTextHeads / kTextKvHeads);
  const int hq = head % (kTextHeads / kTextKvHeads);
  const size_t batch = (static_cast<size_t>(group) * chunks + chunk) *
                           (kTextHeads / kTextKvHeads) + hq;
  out[batch * frames * kTextHeadDim + static_cast<size_t>(frame) * kTextHeadDim + dim] = in[i];
}

// k/v: concat [chunks*L, 512] -> [4 groups][chunks][4 hq-copies][L, 128]:
// each KV head is replicated for the 4 query heads of its group so every
// batched GEMM operand advances with a uniform stride.
__global__ void scatter_kv_groupmajor_kernel(const half* __restrict__ in,
                                              half* __restrict__ out, int rows, int chunks) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int width = kTextKvHeads * kTextHeadDim;
  const size_t total = static_cast<size_t>(rows) * width;
  if (i >= total) return;
  const int col = static_cast<int>(i % width);
  const int row = static_cast<int>(i / width);
  const int frames = rows / chunks;
  const int chunk = row / frames;
  const int frame = row % frames;
  const int group = col / kTextHeadDim;
  const int dim = col % kTextHeadDim;
  const int copies = kTextHeads / kTextKvHeads;
  for (int hq = 0; hq < copies; ++hq) {
    const size_t batch =
        (static_cast<size_t>(group) * chunks + chunk) * copies + hq;
    out[batch * frames * kTextHeadDim + static_cast<size_t>(frame) * kTextHeadDim + dim] =
        in[i];
  }
}

// attn out: [4 groups][chunks][4 hq][L, 128] -> concat [chunks*L, 2048]
__global__ void gather_q_groupmajor_kernel(const half* __restrict__ in,
                                            half* __restrict__ out, int rows, int chunks) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int hidden = kTextHeads * kTextHeadDim;
  const size_t total = static_cast<size_t>(rows) * hidden;
  if (i >= total) return;
  const int col = static_cast<int>(i % hidden);
  const int row = static_cast<int>(i / hidden);
  const int head = col / kTextHeadDim;
  const int dim = col % kTextHeadDim;
  const int frames = rows / chunks;
  const int chunk = row / frames;
  const int frame = row % frames;
  const int group = head / (kTextHeads / kTextKvHeads);
  const int hq = head % (kTextHeads / kTextKvHeads);
  const size_t batch = (static_cast<size_t>(group) * chunks + chunk) *
                           (kTextHeads / kTextKvHeads) + hq;
  out[i] = in[batch * frames * kTextHeadDim + static_cast<size_t>(frame) * kTextHeadDim + dim];
}

// ---- Cross-chunk batched decode kernels ----

// Gathers prompt embeddings for a group of chunks: token rows with audio
// placeholders taken from that chunk's projector output.
__global__ void gather_batch_kernel(const int64_t* __restrict__ ids, int prompt_length,
                                    const half* __restrict__ embed,
                                    const half* __restrict__ audio,  // [chunks,375,2048]
                                    int audio_chunk_stride, half* __restrict__ out,
                                    int group_chunks) {
  const int chunk = blockIdx.y;
  const int pos = blockIdx.x;
  if (pos >= prompt_length || chunk >= group_chunks) return;
  const int64_t id = ids[pos];
  int audio_used = 0;
  for (int i = 0; i < pos; ++i) audio_used += ids[i] == kAudioTokenId ? 1 : 0;
  const half* source =
      (id == kAudioTokenId)
          ? audio + static_cast<size_t>(chunk) * audio_chunk_stride +
                static_cast<size_t>(audio_used) * kTextHidden
          : embed + static_cast<size_t>(id) * kTextHidden;
  half* dst = out + (static_cast<size_t>(chunk) * prompt_length + pos) * kTextHidden;
  for (int d = threadIdx.x; d < kTextHidden; d += blockDim.x) dst[d] = source[d];
}

// RoPE at one shared position for every row (decode lockstep).
__global__ void rope_at_position_kernel(half* __restrict__ q, half* __restrict__ k,
                                        const float* __restrict__ cos_t,
                                        const float* __restrict__ sin_t, int position,
                                        int rows) {
  const int d = threadIdx.x;
  const int row = blockIdx.x;
  if (d >= kTextRotaryDim / 2 || row >= rows) return;
  const float c = cos_t[static_cast<size_t>(position) * (kTextRotaryDim / 2) + d];
  const float s = sin_t[static_cast<size_t>(position) * (kTextRotaryDim / 2) + d];
  half* q_row = q + static_cast<size_t>(row) * kTextHidden;
  for (int h = 0; h < kTextHeads; ++h) {
    half* ptr = q_row + static_cast<size_t>(h) * kTextHeadDim;
    const float x1 = __half2float(ptr[d]);
    const float x2 = __half2float(ptr[d + kTextRotaryDim / 2]);
    ptr[d] = __float2half(x1 * c - x2 * s);
    ptr[d + kTextRotaryDim / 2] = __float2half(x2 * c + x1 * s);
  }
  half* k_row = k + static_cast<size_t>(row) * (kTextKvHeads * kTextHeadDim);
  for (int h = 0; h < kTextKvHeads; ++h) {
    half* ptr = k_row + static_cast<size_t>(h) * kTextHeadDim;
    const float x1 = __half2float(ptr[d]);
    const float x2 = __half2float(ptr[d + kTextRotaryDim / 2]);
    ptr[d] = __float2half(x1 * c - x2 * s);
    ptr[d + kTextRotaryDim / 2] = __float2half(x2 * c + x1 * s);
  }
}

// One block per (chunk, head): single-position attention over that chunk's KV
// cache for one layer; all chunks share the same current length (lockstep).
__global__ void attention_step_batched_kernel(
    const half* __restrict__ q,                  // [chunks, 2048]
    const half* __restrict__ kv_cache,           // [chunks, layers, 2, max, 512]
    half* __restrict__ out,                      // [chunks, 2048]
    int length, int layer, size_t layer_stride, size_t chunk_stride) {
  extern __shared__ float shared[];
  float* scores = shared;          // [kMaxSequence]
  float* sums = shared + kMaxSequence;
  const int index = blockIdx.x;
  const int chunk = index / kTextHeads;
  const int head = index % kTextHeads;
  const int kv_head = head / (kTextHeads / kTextKvHeads);
  const half* q_head =
      q + static_cast<size_t>(chunk) * kTextHidden + static_cast<size_t>(head) * kTextHeadDim;
  const half* k_cache =
      kv_cache + static_cast<size_t>(chunk) * chunk_stride +
      static_cast<size_t>(layer) * layer_stride;
  const half* v_cache = k_cache + static_cast<size_t>(kMaxSequence) * (kTextKvHeads * kTextHeadDim);
  const size_t row_width = kTextKvHeads * kTextHeadDim;
  for (int t = threadIdx.x; t < length; t += blockDim.x) {
    float dot = 0.0f;
    for (int d = 0; d < kTextHeadDim; ++d)
      dot += __half2float(q_head[d]) *
             __half2float(k_cache[static_cast<size_t>(t) * row_width +
                                    static_cast<size_t>(kv_head) * kTextHeadDim + d]);
    scores[t] = dot / 11.3137085f;
  }
  __syncthreads();
  float maximum = -1e30f;
  for (int t = 0; t < length; ++t) maximum = fmaxf(maximum, scores[t]);
  float local = 0.0f;
  for (int t = threadIdx.x; t < length; t += blockDim.x) {
    scores[t] = __expf(scores[t] - maximum);
    local += scores[t];
  }
  sums[threadIdx.x] = local;
  __syncthreads();
  float total = 0.0f;
  for (int i = 0; i < blockDim.x; ++i) total += sums[i];
  __syncthreads();
  const float inv = 1.0f / total;
  for (int t = threadIdx.x; t < length; t += blockDim.x) scores[t] *= inv;
  __syncthreads();
  half* out_head =
      out + static_cast<size_t>(chunk) * kTextHidden + static_cast<size_t>(head) * kTextHeadDim;
  for (int d = threadIdx.x; d < kTextHeadDim; d += blockDim.x) {
    float acc = 0.0f;
    for (int t = 0; t < length; ++t)
      acc += scores[t] *
             __half2float(v_cache[static_cast<size_t>(t) * row_width +
                                    static_cast<size_t>(kv_head) * kTextHeadDim + d]);
    out_head[d] = __float2half(acc);
  }
}

// Argmax of each logits row [chunks, vocab] -> ids [chunks].
__global__ void argmax_rows_kernel(const half* __restrict__ logits, int chunks,
                                   int64_t* __restrict__ ids) {
  const int chunk = blockIdx.x;
  if (chunk >= chunks) return;
  __shared__ float local_best[256];
  __shared__ int local_index[256];
  const half* row = logits + static_cast<size_t>(chunk) * kVocab;
  float best = -1e30f;
  int index = 0;
  for (int j = threadIdx.x; j < kVocab; j += blockDim.x) {
    const float v = __half2float(row[j]);
    if (v > best) {
      best = v;
      index = j;
    }
  }
  local_best[threadIdx.x] = best;
  local_index[threadIdx.x] = index;
  __syncthreads();
  if (threadIdx.x == 0) {
    float overall = -1e30f;
    int winner = 0;
    for (int t = 0; t < blockDim.x; ++t)
      if (local_best[t] > overall) {
        overall = local_best[t];
        winner = local_index[t];
      }
    ids[chunk] = winner;
  }
}

// ---- Batched prefill kernels (sequence-level, causal) ----

// RoPE for every row of q [rows, 16*128] and k [rows, 4*128]; position = row.
__global__ void rope_rows_kernel(half* __restrict__ q, half* __restrict__ k,
                                 const float* __restrict__ cos_t,
                                 const float* __restrict__ sin_t, int rows, int frames) {
  const int d = threadIdx.x;              // 0..63 frequency pair
  const int row = blockIdx.x;
  if (d >= kTextRotaryDim / 2 || row >= rows) return;
  const int row_position = frames > 0 ? row % frames : row;
  const float c = cos_t[static_cast<size_t>(row_position) * (kTextRotaryDim / 2) + d];
  const float s = sin_t[static_cast<size_t>(row_position) * (kTextRotaryDim / 2) + d];
  half* q_row = q + static_cast<size_t>(row) * kTextHidden;
  for (int h = 0; h < kTextHeads; ++h) {
    half* ptr = q_row + static_cast<size_t>(h) * kTextHeadDim;
    const float x1 = __half2float(ptr[d]);
    const float x2 = __half2float(ptr[d + kTextRotaryDim / 2]);
    ptr[d] = __float2half(x1 * c - x2 * s);
    ptr[d + kTextRotaryDim / 2] = __float2half(x2 * c + x1 * s);
  }
  half* k_row = k + static_cast<size_t>(row) * (kTextKvHeads * kTextHeadDim);
  for (int h = 0; h < kTextKvHeads; ++h) {
    half* ptr = k_row + static_cast<size_t>(h) * kTextHeadDim;
    const float x1 = __half2float(ptr[d]);
    const float x2 = __half2float(ptr[d + kTextRotaryDim / 2]);
    ptr[d] = __float2half(x1 * c - x2 * s);
    ptr[d + kTextRotaryDim / 2] = __float2half(x2 * c + x1 * s);
  }
}

// Causal mask on the row-major score buffer S(i, j) at [i*L + j]: j > i -> -inf.
__global__ void causal_mask_kernel(half* __restrict__ scores, int heads, int length) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= length || j >= length) return;
  if (j > i) {
    for (int h = 0; h < heads; ++h)
      scores[static_cast<size_t>(h) * length * length + static_cast<size_t>(i) * length +
             j] = __float2half(-65504.0f);
  }
}

// Row softmax over the contiguous query rows of the row-major score buffer
// S(i, j) at [head * L * L + i * L + j].
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

// Appends one k/v row per chunk into each chunk's own cache at `start`.
__global__ void append_kv_batched_kernel(const half* __restrict__ k, const half* __restrict__ v,
                                         half* __restrict__ k_cache,
                                         half* __restrict__ v_cache, int start, int chunks,
                                         size_t chunk_stride) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int width = kTextKvHeads * kTextHeadDim;
  if (i >= static_cast<size_t>(chunks) * width) return;
  const int chunk = static_cast<int>(i / width);
  const size_t col = i % width;
  k_cache[static_cast<size_t>(chunk) * chunk_stride + static_cast<size_t>(start) * width + col] =
      k[i];
  v_cache[static_cast<size_t>(chunk) * chunk_stride + static_cast<size_t>(start) * width + col] =
      v[i];
}

// Writes the prefilled k/v rows [rows, 512] into a chunk's KV cache slots.
__global__ void append_kv_rows_kernel(const half* __restrict__ k, const half* __restrict__ v,
                                      half* __restrict__ k_cache, half* __restrict__ v_cache,
                                      int start, int rows) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int width = kTextKvHeads * kTextHeadDim;
  if (i >= static_cast<size_t>(rows) * width) return;
  const int row = static_cast<int>(i / width);
  k_cache[static_cast<size_t>(start + row) * width + i % width] = k[i];
  v_cache[static_cast<size_t>(start + row) * width + i % width] = v[i];
}

// Single-position attention over the cache for one query head: scores against
// all cached keys of the matching KV head (GQA group), softmax, weighted sum.
__global__ void attention_step_kernel(const half* __restrict__ q,      // [2048]
                                      const half* __restrict__ k_cache,  // [max,512]
                                      const half* __restrict__ v_cache,
                                      half* __restrict__ out,          // [2048]
                                      int length, int head) {
  extern __shared__ float shared[];
  float* scores = shared;  // [length <= kMaxSequence]
  const int kv_head = head / (kTextHeads / kTextKvHeads);
  const half* q_head = q + static_cast<size_t>(head) * kTextHeadDim;
  // one thread per cached position computes the 128-dim dot product
  for (int t = threadIdx.x; t < length; t += blockDim.x) {
    float dot = 0.0f;
    for (int d = 0; d < kTextHeadDim; ++d)
      dot += __half2float(q_head[d]) *
             __half2float(k_cache[static_cast<size_t>(t) * (kTextKvHeads * kTextHeadDim) +
                                    static_cast<size_t>(kv_head) * kTextHeadDim + d]);
    scores[t] = dot / 11.3137085f;  // 1/sqrt(128)
  }
  __syncthreads();
  float maximum = -1e30f;
  for (int t = 0; t < length; ++t) maximum = fmaxf(maximum, scores[t]);
  float sum = 0.0f;
  for (int t = threadIdx.x; t < length; t += blockDim.x) {
    scores[t] = __expf(scores[t] - maximum);
    sum += scores[t];
  }
  __syncthreads();
  // reduce sum across threads through shared tail
  float* sums = shared + kMaxSequence;
  sums[threadIdx.x] = sum;
  __syncthreads();
  float total = 0.0f;
  for (int i = 0; i < blockDim.x; ++i) total += sums[i];
  __syncthreads();
  for (int t = threadIdx.x; t < length; t += blockDim.x) scores[t] /= total;
  __syncthreads();
  half* out_head = out + static_cast<size_t>(head) * kTextHeadDim;
  for (int d = threadIdx.x; d < kTextHeadDim; d += blockDim.x) {
    float acc = 0.0f;
    for (int t = 0; t < length; ++t)
      acc += scores[t] *
             __half2float(v_cache[static_cast<size_t>(t) * (kTextKvHeads * kTextHeadDim) +
                                    static_cast<size_t>(kv_head) * kTextHeadDim + d]);
    out_head[d] = __float2half(acc);
  }
}

__global__ void residual_add_kernel(half* __restrict__ x, const half* __restrict__ delta,
                                    size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  x[i] = __float2half(__half2float(x[i]) + __half2float(delta[i]));
}

__global__ void append_kv_kernel(const half* __restrict__ k, const half* __restrict__ v,
                                 half* __restrict__ k_cache, half* __restrict__ v_cache,
                                 int position) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = kTextKvHeads * kTextHeadDim;  // 512
  if (i >= width) return;
  k_cache[static_cast<size_t>(position) * width + i] = k[i];
  v_cache[static_cast<size_t>(position) * width + i] = v[i];
}

std::vector<float> build_full_rope_table(bool cosine) {
  const int pairs = kTextRotaryDim / 2;
  std::vector<float> table(static_cast<size_t>(kMaxSequence) * pairs);
  for (int pos = 0; pos < kMaxSequence; ++pos)
    for (int d = 0; d < pairs; ++d) {
      const double freq = 1.0 / std::pow(10000.0, static_cast<double>(2 * d) / kTextRotaryDim);
      const double angle = freq * pos;
      table[static_cast<size_t>(pos) * pairs + d] =
          static_cast<float>(cosine ? std::cos(angle) : std::sin(angle));
    }
  return table;
}

// Frees every device pointer it owns, on every exit path.
class BufferGuard {
 public:
  ~BufferGuard() {
    for (void* p : owned_) cudaFree(p);
  }
  void* alloc(size_t bytes) {
    void* p = nullptr;
    const cudaError_t status = cudaMalloc(&p, bytes);
    if (status != cudaSuccess) throw ExecutionError(cuda_error(status, "cudaMalloc"));
    owned_.push_back(p);
    return p;
  }

 private:
  std::vector<void*> owned_;
};

#define CUBLAS_THROW(expr)                                                      \
  do {                                                                          \
    const cublasStatus_t status_ = (expr);                                      \
    if (status_ != CUBLAS_STATUS_SUCCESS)                                       \
      throw ExecutionError("cuBLAS call failed with status " #expr);            \
  } while (false)

}  // namespace

DecodeResult run_decode(DeviceWeights* weights, const uint16_t* audio_embeds_raw,
                        const std::vector<int64_t>& prompt_ids, const DecodeOptions& options) {
  const half* audio_embeds_device = reinterpret_cast<const half*>(audio_embeds_raw);
  if (!weights->loaded()) throw ExecutionError("run_decode requires loaded weights");
  if (prompt_ids.empty()) throw ExecutionError("run_decode requires prompt ids");
  const size_t prompt_length = prompt_ids.size();
  if (prompt_length + static_cast<size_t>(options.max_new_tokens) + 1 >= kMaxSequence)
    throw ExecutionError("prompt plus budget exceeds the decoder sequence limit");

  cublasHandle_t cublas = nullptr;
  CUBLAS_THROW(cublasCreate(&cublas));
  struct HandleGuard {
    cublasHandle_t handle;
    ~HandleGuard() { if (handle) cublasDestroy(handle); }
  } guard{cublas};

  const auto gemv = [&](const half* w, const half* x, half* y, int n, int k) {
    const half alpha16 = __float2half(1.0f), beta16 = __float2half(0.0f);
    const float alpha32 = 1.0f, beta32 = 0.0f;
    CUBLAS_THROW(cublasGemmEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k,
        options.fp16_accumulate ? static_cast<const void*>(&alpha16)
                                : static_cast<const void*>(&alpha32),
        w, CUDA_R_16F, k, x, CUDA_R_16F, k,
        options.fp16_accumulate ? static_cast<const void*>(&beta16)
                                : static_cast<const void*>(&beta32),
        y, CUDA_R_16F, n,
        options.fp16_accumulate ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
  };
  const auto gemm_rows = [&](const half* w, const half* x, half* y, int m, int n, int k) {
    const half alpha16 = __float2half(1.0f), beta16 = __float2half(0.0f);
    const float alpha32 = 1.0f, beta32 = 0.0f;
    CUBLAS_THROW(cublasGemmEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        options.fp16_accumulate ? static_cast<const void*>(&alpha16)
                                : static_cast<const void*>(&alpha32),
        w, CUDA_R_16F, k, x, CUDA_R_16F, k,
        options.fp16_accumulate ? static_cast<const void*>(&beta16)
                                : static_cast<const void*>(&beta32),
        y, CUDA_R_16F, n,
        options.fp16_accumulate ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
  };
  const auto weight = [&](const std::string& name) -> const half* {
    const DeviceWeights::Entry* entry = weights->find(name);
    if (!entry) throw ExecutionError("missing resident tensor: " + name);
    return reinterpret_cast<const half*>(entry->ptr);
  };

  const size_t hidden_count = kTextHidden;
  const size_t kv_width = kTextKvHeads * kTextHeadDim;
  const size_t mlp_width = 6144;

  half *d_x = nullptr, *d_h = nullptr, *d_ln = nullptr, *d_q = nullptr, *d_k = nullptr,
      *d_v = nullptr, *d_attn = nullptr, *d_mlp_g = nullptr, *d_mlp_u = nullptr,
      *d_logits = nullptr, *d_kv_cache = nullptr, *d_row = nullptr;
  float *d_cos = nullptr, *d_sin = nullptr;
  int64_t* d_prompt_ids = nullptr;
  try {
    CUDA_CHECK(cudaMalloc(&d_prompt_ids, prompt_length * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_x, prompt_length * hidden_count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_row, hidden_count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_h, hidden_count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ln, hidden_count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_q, hidden_count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, kv_width * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, kv_width * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_attn, hidden_count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_mlp_g, mlp_width * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_mlp_u, mlp_width * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_logits, static_cast<size_t>(kVocab) * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_kv_cache,
                          static_cast<size_t>(kTextLayers) * 2 * kMaxSequence * kv_width *
                              sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_cos, static_cast<size_t>(kMaxSequence) * 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, static_cast<size_t>(kMaxSequence) * 64 * sizeof(float)));
    {
      const std::vector<float> cos_t = build_full_rope_table(true);
      const std::vector<float> sin_t = build_full_rope_table(false);
      CUDA_CHECK(cudaMemcpy(d_cos, cos_t.data(), cos_t.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_sin, sin_t.data(), sin_t.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_prompt_ids, prompt_ids.data(), prompt_length * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    gather_inputs_kernel<<<static_cast<unsigned>(prompt_length), 128>>>(
        d_prompt_ids, static_cast<int>(prompt_length),
        weight("language_model.model.embed_tokens.weight"), audio_embeds_device, d_x);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Runs one position through the 28 layers; when want_logits, also applies
    // the final norm + lm_head and returns the argmax token id.
    std::vector<half> logits_host(kVocab);
    const auto forward_position = [&](int position, bool want_logits) -> int64_t {
      for (int layer = 0; layer < kTextLayers; ++layer) {
        const std::string p = "language_model.model.layers." + std::to_string(layer) + ".";
        half* k_cache =
            d_kv_cache + static_cast<size_t>(layer) * 2 * kMaxSequence * kv_width;
        half* v_cache = k_cache + static_cast<size_t>(kMaxSequence) * kv_width;
        CUDA_CHECK(cudaMemcpy(d_ln, d_h, hidden_count * sizeof(half),
                              cudaMemcpyDeviceToDevice));
        rmsnorm_kernel<<<1, 256, 257 * sizeof(float)>>>(d_ln,
                                                        weight(p + "input_layernorm.weight"),
                                                        kTextHidden);
        CUDA_CHECK(cudaGetLastError());
        gemv(weight(p + "self_attn.q_proj.weight"), d_ln, d_q, kTextHidden, kTextHidden);
        gemv(weight(p + "self_attn.k_proj.weight"), d_ln, d_k, static_cast<int>(kv_width),
             kTextHidden);
        gemv(weight(p + "self_attn.v_proj.weight"), d_ln, d_v, static_cast<int>(kv_width),
             kTextHidden);
        rope_full_kernel<<<1, 64>>>(d_q, d_k, d_cos, d_sin, position);
        CUDA_CHECK(cudaGetLastError());
        append_kv_kernel<<<(static_cast<unsigned>(kv_width) + 127) / 128, 128>>>(
            d_k, d_v, k_cache, v_cache, position);
        CUDA_CHECK(cudaGetLastError());
        for (int head = 0; head < kTextHeads; ++head)
          attention_step_kernel<<<1, 128, (kMaxSequence + 128) * sizeof(float)>>>(
              d_q, k_cache, v_cache, d_attn, position + 1, head);
        CUDA_CHECK(cudaGetLastError());
        gemv(weight(p + "self_attn.o_proj.weight"), d_attn, d_ln, kTextHidden, kTextHidden);
        residual_add_kernel<<<(static_cast<unsigned>(hidden_count) + 255) / 256, 256>>>(
            d_h, d_ln, hidden_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_ln, d_h, hidden_count * sizeof(half),
                              cudaMemcpyDeviceToDevice));
        rmsnorm_kernel<<<1, 256, 257 * sizeof(float)>>>(
            d_ln, weight(p + "post_attention_layernorm.weight"), kTextHidden);
        CUDA_CHECK(cudaGetLastError());
        gemv(weight(p + "mlp.gate_proj.weight"), d_ln, d_mlp_g, static_cast<int>(mlp_width),
             kTextHidden);
        gemv(weight(p + "mlp.up_proj.weight"), d_ln, d_mlp_u, static_cast<int>(mlp_width),
             kTextHidden);
        silu_mul_kernel<<<(static_cast<unsigned>(mlp_width) + 255) / 256, 256>>>(
            d_mlp_g, d_mlp_u, mlp_width);
        CUDA_CHECK(cudaGetLastError());
        gemv(weight(p + "mlp.down_proj.weight"), d_mlp_g, d_ln, kTextHidden,
             static_cast<int>(mlp_width));
        residual_add_kernel<<<(static_cast<unsigned>(hidden_count) + 255) / 256, 256>>>(
            d_h, d_ln, hidden_count);
        CUDA_CHECK(cudaGetLastError());
      }
      if (!want_logits) return -1;
      rmsnorm_kernel<<<1, 256, 257 * sizeof(float)>>>(d_h,
                                                      weight("language_model.model.norm.weight"),
                                                      kTextHidden);
      CUDA_CHECK(cudaGetLastError());
      gemv(weight("language_model.lm_head.weight"), d_h, d_logits, kVocab, kTextHidden);
      CUDA_CHECK(cudaMemcpy(logits_host.data(), d_logits, kVocab * sizeof(half),
                            cudaMemcpyDeviceToHost));
      int best = 0;
      float best_value = -1e30f;
      for (int i = 0; i < kVocab; ++i) {
        const float value = __half2float(logits_host[i]);
        if (value > best_value) {
          best_value = value;
          best = i;
        }
      }
      return static_cast<int64_t>(best);
    };

    DecodeResult result;
    const std::vector<int64_t> eos_ids = {59246, 59253, 59255};

    // Batched prefill: the whole prompt through every layer as sequence-level
    // GEMMs with causal masked attention, writing the KV cache in one pass.
    // GQA (16 query heads on 4 KV heads) runs as 4 uniform-stride batched GEMMs.
    const auto prefill_batched = [&]() -> int64_t {
      const int length = static_cast<int>(prompt_length);
      const size_t seq_hidden = prompt_length * hidden_count;
      const size_t seq_kv = prompt_length * kv_width;
      const size_t seq_mlp = prompt_length * mlp_width;
      BufferGuard seq;
      half* h_seq = static_cast<half*>(seq.alloc(seq_hidden * sizeof(half)));
      half* ln_seq = static_cast<half*>(seq.alloc(seq_hidden * sizeof(half)));
      half* q_seq = static_cast<half*>(seq.alloc(seq_hidden * sizeof(half)));
      half* k_seq = static_cast<half*>(seq.alloc(seq_kv * sizeof(half)));
      half* v_seq = static_cast<half*>(seq.alloc(seq_kv * sizeof(half)));
      half* attn_seq = static_cast<half*>(seq.alloc(seq_hidden * sizeof(half)));
      half* scratch_seq = static_cast<half*>(seq.alloc(seq_hidden * sizeof(half)));
      half* mlp_g_seq = static_cast<half*>(seq.alloc(seq_mlp * sizeof(half)));
      half* mlp_u_seq = static_cast<half*>(seq.alloc(seq_mlp * sizeof(half)));
      half* scores = static_cast<half*>(seq.alloc(
          static_cast<size_t>(kTextHeads) * prompt_length * prompt_length * sizeof(half)));
      CUDA_CHECK(cudaMemcpy(h_seq, d_x, seq_hidden * sizeof(half), cudaMemcpyDeviceToDevice));

      const float attn_scale = 1.0f / 11.3137085f;
      const float one32 = 1.0f, zero32 = 0.0f;
      const half zero32_h = __float2half(0.0f);
      for (int layer = 0; layer < kTextLayers; ++layer) {
        const std::string p = "language_model.model.layers." + std::to_string(layer) + ".";
        half* k_cache =
            d_kv_cache + static_cast<size_t>(layer) * 2 * kMaxSequence * kv_width;
        half* v_cache = k_cache + static_cast<size_t>(kMaxSequence) * kv_width;
        CUDA_CHECK(cudaMemcpy(ln_seq, h_seq, seq_hidden * sizeof(half),
                              cudaMemcpyDeviceToDevice));
        rmsnorm_kernel<<<static_cast<unsigned>(prompt_length), 256, 257 * sizeof(float)>>>(
            ln_seq, weight(p + "input_layernorm.weight"), kTextHidden);
        CUDA_CHECK(cudaGetLastError());
        gemm_rows(weight(p + "self_attn.q_proj.weight"), ln_seq, q_seq, length, kTextHidden,
                  kTextHidden);
        gemm_rows(weight(p + "self_attn.k_proj.weight"), ln_seq, k_seq, length,
                  static_cast<int>(kv_width), kTextHidden);
        gemm_rows(weight(p + "self_attn.v_proj.weight"), ln_seq, v_seq, length,
                  static_cast<int>(kv_width), kTextHidden);
        rope_rows_kernel<<<static_cast<unsigned>(prompt_length), 64>>>(
            q_seq, k_seq, d_cos, d_sin, length, length);
        CUDA_CHECK(cudaGetLastError());
        append_kv_rows_kernel<<<static_cast<unsigned>((seq_kv + 127) / 128), 128>>>(
            k_seq, v_seq, k_cache, v_cache, 0, length);
        CUDA_CHECK(cudaGetLastError());
        for (int group = 0; group < kTextHeads / kTextKvHeads; ++group) {
          const int heads_per_group = kTextHeads / kTextKvHeads;  // 4
          const half* q_group = q_seq + static_cast<size_t>(group) * heads_per_group * kTextHeadDim;
          const half* k_group = k_seq + static_cast<size_t>(group) * kTextHeadDim;
          CUBLAS_THROW(cublasGemmStridedBatchedEx(
              cublas, CUBLAS_OP_T, CUBLAS_OP_N, length, length, kTextHeadDim,
              static_cast<const void*>(&attn_scale), k_group, CUDA_R_16F,
              static_cast<int>(kv_width), 0, q_group, CUDA_R_16F, kTextHidden,
              kTextHeadDim, static_cast<const void*>(&zero32),
              scores + static_cast<size_t>(group) * heads_per_group * prompt_length * prompt_length,
              CUDA_R_16F, length, static_cast<long long>(prompt_length) * prompt_length,
              heads_per_group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
        {
          const dim3 blocks((static_cast<unsigned>(prompt_length) + 15) / 16,
                            (static_cast<unsigned>(prompt_length) + 15) / 16);
          causal_mask_kernel<<<blocks, dim3(16, 16)>>>(scores, kTextHeads, length);
          CUDA_CHECK(cudaGetLastError());
          softmax_rows_kernel<<<dim3(static_cast<unsigned>(prompt_length), kTextHeads), 256>>>(
              scores, kTextHeads, length);
          CUDA_CHECK(cudaGetLastError());
        }
        for (int group = 0; group < kTextHeads / kTextKvHeads; ++group) {
          const int heads_per_group = kTextHeads / kTextKvHeads;
          const half* v_group = v_seq + static_cast<size_t>(group) * kTextHeadDim;
          CUBLAS_THROW(cublasGemmStridedBatchedEx(
              cublas, CUBLAS_OP_N, CUBLAS_OP_N, kTextHeadDim, length, length,
              static_cast<const void*>(&one32), v_group, CUDA_R_16F,
              static_cast<int>(kv_width), 0,
              scores + static_cast<size_t>(group) * heads_per_group * prompt_length * prompt_length,
              CUDA_R_16F, length, static_cast<long long>(prompt_length) * prompt_length,
              static_cast<const void*>(&zero32_h),
              attn_seq + static_cast<size_t>(group) * heads_per_group * kTextHeadDim,
              CUDA_R_16F, kTextHidden, kTextHeadDim, heads_per_group, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT));
        }
        gemm_rows(weight(p + "self_attn.o_proj.weight"), attn_seq, scratch_seq, length,
                  kTextHidden, kTextHidden);
        residual_add_kernel<<<static_cast<unsigned>((seq_hidden + 255) / 256), 256>>>(
            h_seq, scratch_seq, seq_hidden);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(ln_seq, h_seq, seq_hidden * sizeof(half),
                              cudaMemcpyDeviceToDevice));
        rmsnorm_kernel<<<static_cast<unsigned>(prompt_length), 256, 257 * sizeof(float)>>>(
            ln_seq, weight(p + "post_attention_layernorm.weight"), kTextHidden);
        CUDA_CHECK(cudaGetLastError());
        gemm_rows(weight(p + "mlp.gate_proj.weight"), ln_seq, mlp_g_seq, length,
                  static_cast<int>(mlp_width), kTextHidden);
        gemm_rows(weight(p + "mlp.up_proj.weight"), ln_seq, mlp_u_seq, length,
                  static_cast<int>(mlp_width), kTextHidden);
        silu_mul_kernel<<<static_cast<unsigned>((seq_mlp + 255) / 256), 256>>>(
            mlp_g_seq, mlp_u_seq, seq_mlp);
        CUDA_CHECK(cudaGetLastError());
        gemm_rows(weight(p + "mlp.down_proj.weight"), mlp_g_seq, scratch_seq, length,
                  kTextHidden, static_cast<int>(mlp_width));
        residual_add_kernel<<<static_cast<unsigned>((seq_hidden + 255) / 256), 256>>>(
            h_seq, scratch_seq, seq_hidden);
        CUDA_CHECK(cudaGetLastError());
      }
      rmsnorm_kernel<<<static_cast<unsigned>(prompt_length), 256, 257 * sizeof(float)>>>(
          h_seq, weight("language_model.model.norm.weight"), kTextHidden);
      CUDA_CHECK(cudaGetLastError());
      // The first generated token comes from the last prompt row only.
      CUDA_CHECK(cudaMemcpy(d_h, h_seq + (prompt_length - 1) * hidden_count,
                            hidden_count * sizeof(half), cudaMemcpyDeviceToDevice));
      gemv(weight("language_model.lm_head.weight"), d_h, d_logits, kVocab, kTextHidden);
      CUDA_CHECK(cudaMemcpy(logits_host.data(), d_logits, kVocab * sizeof(half),
                            cudaMemcpyDeviceToHost));
      int best = 0;
      float best_value = -1e30f;
      for (int i = 0; i < kVocab; ++i) {
        const float value = __half2float(logits_host[i]);
        if (value > best_value) {
          best_value = value;
          best = i;
        }
      }
      return static_cast<int64_t>(best);
    };

    int64_t first_token;
    if (options.prefill_batched) {
      first_token = prefill_batched();
    } else {
      // M0 sequential path: run every prompt row through the cache; the last
      // row yields the first generated token.
      first_token = -1;
      for (size_t pos = 0; pos < prompt_length; ++pos) {
        CUDA_CHECK(cudaMemcpy(d_h, d_x + pos * hidden_count, hidden_count * sizeof(half),
                              cudaMemcpyDeviceToDevice));
        const bool last = pos + 1 == prompt_length;
        const int64_t next = forward_position(static_cast<int>(pos), last);
        if (last) first_token = next;
      }
    }
    result.generated_ids.push_back(first_token);
    if (std::find(eos_ids.begin(), eos_ids.end(), first_token) != eos_ids.end()) {
      result.stop_reason = 0;
      cudaFree(d_prompt_ids); cudaFree(d_x); cudaFree(d_row); cudaFree(d_h);
      cudaFree(d_ln); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_attn);
      cudaFree(d_mlp_g); cudaFree(d_mlp_u); cudaFree(d_logits); cudaFree(d_kv_cache);
      cudaFree(d_cos); cudaFree(d_sin);
      return result;
    }
    // Decode loop.
    int64_t current = result.generated_ids.back();
    for (int step = 1; step < options.max_new_tokens; ++step) {
      const int64_t id_scratch[1] = {current};
      CUDA_CHECK(cudaMemcpy(d_prompt_ids, id_scratch, sizeof(int64_t),
                            cudaMemcpyHostToDevice));
      gather_inputs_kernel<<<1, 128>>>(d_prompt_ids, 1,
                                       weight("language_model.model.embed_tokens.weight"),
                                       audio_embeds_device, d_row);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(d_h, d_row, hidden_count * sizeof(half),
                            cudaMemcpyDeviceToDevice));
      const int position = static_cast<int>(prompt_length + step - 1);
      current = forward_position(position, true);
      result.generated_ids.push_back(current);
      if (std::find(eos_ids.begin(), eos_ids.end(), current) != eos_ids.end()) {
        result.stop_reason = 0;
        break;
      }
    }
    if (result.stop_reason != 0) result.stop_reason = 1;
    cudaFree(d_prompt_ids); cudaFree(d_x); cudaFree(d_row); cudaFree(d_h);
    cudaFree(d_ln); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_attn);
    cudaFree(d_mlp_g); cudaFree(d_mlp_u); cudaFree(d_logits); cudaFree(d_kv_cache);
    cudaFree(d_cos); cudaFree(d_sin);
    return result;
  } catch (...) {
    cudaFree(d_prompt_ids); cudaFree(d_x); cudaFree(d_row); cudaFree(d_h);
    cudaFree(d_ln); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_attn);
    cudaFree(d_mlp_g); cudaFree(d_mlp_u); cudaFree(d_logits); cudaFree(d_kv_cache);
    cudaFree(d_cos); cudaFree(d_sin);
    throw;
  }
}

// ---------------------------------------------------------------------------
// Cross-chunk batched decode: all chunks share the prompt and step in
// lockstep. Prefill runs in groups of 8 chunks (sequence GEMMs + per-chunk
// causal attention); decode steps run the whole batch at m=chunks with a
// per-(chunk, head) attention kernel and uniform KV length. Finished chunks
// freeze to EOS on the host side (tiny per-step id round-trip).
// ---------------------------------------------------------------------------

DecodeBatchResult run_decode_batched(DeviceWeights* weights,
                                     const uint16_t* audio_embeds_raw, int chunks,
                                     const std::vector<int64_t>& prompt_ids,
                                     const DecodeOptions& options) {
  if (!weights->loaded()) throw ExecutionError("run_decode_batched requires loaded weights");
  if (chunks <= 0 || chunks > 128) throw ExecutionError("run_decode_batched supports 1..128 chunks");
  const size_t prompt_length = prompt_ids.size();
  if (prompt_length + static_cast<size_t>(options.max_new_tokens) + 1 >= kMaxSequence)
    throw ExecutionError("prompt plus budget exceeds the decoder sequence limit");
  const half* audio_embeds = reinterpret_cast<const half*>(audio_embeds_raw);

  cublasHandle_t cublas = nullptr;
  CUBLAS_THROW(cublasCreate(&cublas));
  struct HandleGuard {
    cublasHandle_t handle;
    ~HandleGuard() { if (handle) cublasDestroy(handle); }
  } guard{cublas};

  const auto gemm = [&](const half* w, const half* x, half* y, int m, int n, int k) {
    const half alpha16 = __float2half(1.0f), beta16 = __float2half(0.0f);
    const float alpha32 = 1.0f, beta32 = 0.0f;
    CUBLAS_THROW(cublasGemmEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        options.fp16_accumulate ? static_cast<const void*>(&alpha16)
                                : static_cast<const void*>(&alpha32),
        w, CUDA_R_16F, k, x, CUDA_R_16F, k,
        options.fp16_accumulate ? static_cast<const void*>(&beta16)
                                : static_cast<const void*>(&beta32),
        y, CUDA_R_16F, n,
        options.fp16_accumulate ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
  };
  const auto weight = [&](const std::string& name) -> const half* {
    const DeviceWeights::Entry* entry = weights->find(name);
    if (!entry) throw ExecutionError("missing resident tensor: " + name);
    return reinterpret_cast<const half*>(entry->ptr);
  };

  const int L = static_cast<int>(prompt_length);
  const size_t kv_width = kTextKvHeads * kTextHeadDim;
  const size_t layer_kv_stride = 2ull * kMaxSequence * kv_width;
  const size_t chunk_kv_stride = static_cast<size_t>(kTextLayers) * layer_kv_stride;
  const size_t audio_chunk_stride = static_cast<size_t>(kAudioTokens) * kTextHidden;

  BufferGuard g;
  const int kGroup = 8;  // prefill sub-batch
  const size_t group_rows = static_cast<size_t>(kGroup) * prompt_length;
  half* d_prompt_ids = static_cast<half*>(g.alloc(prompt_length * sizeof(int64_t)));
  half* d_x_group = static_cast<half*>(g.alloc(group_rows * kTextHidden * sizeof(half)));
  half* d_h_group = static_cast<half*>(g.alloc(group_rows * kTextHidden * sizeof(half)));
  half* d_ln_group = static_cast<half*>(g.alloc(group_rows * kTextHidden * sizeof(half)));
  half* d_q_group = static_cast<half*>(g.alloc(group_rows * kTextHidden * sizeof(half)));
  half* d_kv_group = static_cast<half*>(g.alloc(group_rows * kv_width * sizeof(half) * 2));
  half* d_k_group = d_kv_group;
  half* d_v_group = d_kv_group + group_rows * kv_width;
  half* d_attn_group = static_cast<half*>(g.alloc(group_rows * kTextHidden * sizeof(half)));
  half* d_scratch_group = static_cast<half*>(g.alloc(group_rows * kTextHidden * sizeof(half)));
  half* d_mlp_group =
      static_cast<half*>(g.alloc(group_rows * 6144 * sizeof(half) * 2));
  half* d_mlp_g = d_mlp_group;
  half* d_mlp_u = d_mlp_group + group_rows * 6144;
  half* d_scores = static_cast<half*>(g.alloc(
      static_cast<size_t>(kGroup) * kTextHeads * prompt_length * prompt_length *
      sizeof(half)));
  half* d_q_gm = static_cast<half*>(g.alloc(group_rows * kTextHidden * sizeof(half)));
  half* d_k_gm =
      static_cast<half*>(g.alloc(group_rows * kv_width * 4 * sizeof(half)));
  half* d_v_gm =
      static_cast<half*>(g.alloc(group_rows * kv_width * 4 * sizeof(half)));
  half* d_attn_gm = static_cast<half*>(g.alloc(group_rows * kTextHidden * sizeof(half)));
  half* d_kv_cache = static_cast<half*>(g.alloc(static_cast<size_t>(chunks) * chunk_kv_stride *
                                                sizeof(half)));
  float* d_cos = static_cast<float*>(g.alloc(static_cast<size_t>(kMaxSequence) * 64 * sizeof(float)));
  float* d_sin = static_cast<float*>(g.alloc(static_cast<size_t>(kMaxSequence) * 64 * sizeof(float)));
  // lockstep decode buffers (full batch)
  const size_t batch_rows = static_cast<size_t>(chunks);
  half* d_ids_batch =
      static_cast<half*>(g.alloc(batch_rows * kTextHidden * sizeof(half)));  // embed rows
  half* d_h = static_cast<half*>(g.alloc(batch_rows * kTextHidden * sizeof(half)));
  half* d_ln = static_cast<half*>(g.alloc(batch_rows * kTextHidden * sizeof(half)));
  half* d_q = static_cast<half*>(g.alloc(batch_rows * kTextHidden * sizeof(half)));
  half* d_kv = static_cast<half*>(g.alloc(batch_rows * kv_width * sizeof(half) * 2));
  half* d_k_row = d_kv;
  half* d_v_row = d_kv + batch_rows * kv_width;
  half* d_attn = static_cast<half*>(g.alloc(batch_rows * kTextHidden * sizeof(half)));
  half* d_mlp_b = static_cast<half*>(g.alloc(batch_rows * 6144 * sizeof(half) * 2));
  half* d_mlp_bg = d_mlp_b;
  half* d_mlp_bu = d_mlp_b + batch_rows * 6144;
  half* d_logits = static_cast<half*>(g.alloc(batch_rows * kVocab * sizeof(half)));
  int64_t* d_prompt_raw = reinterpret_cast<int64_t*>(d_prompt_ids);
  int64_t* d_current_ids = static_cast<int64_t*>(g.alloc(batch_rows * sizeof(int64_t)));
  int64_t* d_gather_ids = static_cast<int64_t*>(g.alloc(batch_rows * sizeof(int64_t)));

  {
    const std::vector<float> cos_t = build_full_rope_table(true);
    const std::vector<float> sin_t = build_full_rope_table(false);
    CUDA_CHECK(cudaMemcpy(d_cos, cos_t.data(), cos_t.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, sin_t.data(), sin_t.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prompt_raw, prompt_ids.data(), prompt_length * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
  }

  const float attn_scale = 1.0f / 11.3137085f;
  const float one32 = 1.0f, zero32 = 0.0f;
  const half zero_h = __float2half(0.0f);

  // ---- prefill: groups of kGroup chunks ----
  DecodeBatchResult result;
  result.per_chunk_ids.resize(chunks);
  std::vector<int64_t> current(chunks, 0);
  cudaEvent_t prefill_start, prefill_stop, decode_start, decode_stop;
  CUDA_CHECK(cudaEventCreate(&prefill_start));
  CUDA_CHECK(cudaEventCreate(&prefill_stop));
  CUDA_CHECK(cudaEventCreate(&decode_start));
  CUDA_CHECK(cudaEventCreate(&decode_stop));
  CUDA_CHECK(cudaEventRecord(prefill_start));
  for (int base = 0; base < chunks; base += kGroup) {
    const int group = std::min(kGroup, chunks - base);
    const int rows = group * L;
    const size_t seq_hidden = static_cast<size_t>(rows) * kTextHidden;
    const size_t seq_kv = static_cast<size_t>(rows) * kv_width;
    const size_t seq_mlp = static_cast<size_t>(rows) * 6144;
    gather_batch_kernel<<<dim3(L, group), 128>>>(
        d_prompt_raw, L, weight("language_model.model.embed_tokens.weight"),
        audio_embeds + static_cast<size_t>(base) * audio_chunk_stride,
        static_cast<int>(audio_chunk_stride), d_x_group, group);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_h_group, d_x_group, seq_hidden * sizeof(half),
                          cudaMemcpyDeviceToDevice));

    for (int layer = 0; layer < kTextLayers; ++layer) {
      const std::string p = "language_model.model.layers." + std::to_string(layer) + ".";
      half* kv_base = d_kv_cache + static_cast<size_t>(base) * chunk_kv_stride +
                      static_cast<size_t>(layer) * layer_kv_stride;
      CUDA_CHECK(cudaMemcpy(d_ln_group, d_h_group, seq_hidden * sizeof(half),
                            cudaMemcpyDeviceToDevice));
      rmsnorm_kernel<<<rows, 256, 257 * sizeof(float)>>>(
          d_ln_group, weight(p + "input_layernorm.weight"), kTextHidden);
      CUDA_CHECK(cudaGetLastError());
      gemm(weight(p + "self_attn.q_proj.weight"), d_ln_group, d_q_group, rows, kTextHidden,
           kTextHidden);
      gemm(weight(p + "self_attn.k_proj.weight"), d_ln_group, d_k_group, rows,
           static_cast<int>(kv_width), kTextHidden);
      gemm(weight(p + "self_attn.v_proj.weight"), d_ln_group, d_v_group, rows,
           static_cast<int>(kv_width), kTextHidden);
      rope_rows_kernel<<<rows, 64>>>(d_q_group, d_k_group, d_cos, d_sin, rows, L);
      CUDA_CHECK(cudaGetLastError());
      // write this group's K/V rows into each chunk's cache
      for (int c = 0; c < group; ++c) {
        append_kv_rows_kernel<<<static_cast<unsigned>((seq_kv / group + 127) / 128), 128>>>(
            d_k_group + static_cast<size_t>(c) * L * kv_width,
            d_v_group + static_cast<size_t>(c) * L * kv_width,
            kv_base + static_cast<size_t>(c) * chunk_kv_stride,
            kv_base + static_cast<size_t>(c) * chunk_kv_stride +
                static_cast<size_t>(kMaxSequence) * kv_width,
            0, L);
      }
      CUDA_CHECK(cudaGetLastError());
      {
        // group-major re-layout; per layer: 4 uniform-batch scores GEMMs
        // (batch = group chunks * 4), one mask + one softmax, 4 PV GEMMs.
        const size_t q_total = static_cast<size_t>(rows) * kTextHidden;
        const size_t kv_total = static_cast<size_t>(rows) * kv_width;
        scatter_q_groupmajor_kernel<<<static_cast<unsigned>((q_total + 255) / 256), 256>>>(
            d_q_group, d_q_gm, rows, group);
        scatter_kv_groupmajor_kernel<<<static_cast<unsigned>((kv_total + 255) / 256), 256>>>(
            d_k_group, d_k_gm, rows, group);
        scatter_kv_groupmajor_kernel<<<static_cast<unsigned>((kv_total + 255) / 256), 256>>>(
            d_v_group, d_v_gm, rows, group);
        CUDA_CHECK(cudaGetLastError());
        const int per_group = kTextHeads / kTextKvHeads;  // 4
        const long long head_stride = static_cast<long long>(L) * kTextHeadDim;
        const long long score_stride = static_cast<long long>(L) * L;
        for (int grp = 0; grp < kTextKvHeads; ++grp) {
          const int batch = group * per_group;
          const half* q_base = d_q_gm + static_cast<size_t>(grp) * group * per_group * L *
                                             kTextHeadDim;
          const half* kv_base = d_k_gm + static_cast<size_t>(grp) * group *
                                                 (kTextHeads / kTextKvHeads) * L *
                                                 kTextHeadDim;
          CUBLAS_THROW(cublasGemmStridedBatchedEx(
              cublas, CUBLAS_OP_T, CUBLAS_OP_N, L, L, kTextHeadDim,
              static_cast<const void*>(&attn_scale), kv_base, CUDA_R_16F, kTextHeadDim,
              head_stride, q_base, CUDA_R_16F, kTextHeadDim, head_stride,
              static_cast<const void*>(&zero32),
              d_scores + static_cast<size_t>(grp) * group * per_group * L * L, CUDA_R_16F, L,
              score_stride, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
        causal_mask_kernel<<<dim3((L + 15) / 16, (L + 15) / 16), dim3(16, 16)>>>(
            d_scores, kTextHeads * group, L);
        CUDA_CHECK(cudaGetLastError());
        softmax_rows_kernel<<<dim3(L, kTextHeads * group), 256>>>(d_scores, kTextHeads * group,
                                                                  L);
        CUDA_CHECK(cudaGetLastError());
        for (int grp = 0; grp < kTextKvHeads; ++grp) {
          const int per = kTextHeads / kTextKvHeads;
          const int batch = group * per;
          const half* v_base = d_v_gm + static_cast<size_t>(grp) * group *
                                               (kTextHeads / kTextKvHeads) * L *
                                               kTextHeadDim;
          CUBLAS_THROW(cublasGemmStridedBatchedEx(
              cublas, CUBLAS_OP_N, CUBLAS_OP_N, kTextHeadDim, L, L,
              static_cast<const void*>(&one32), v_base, CUDA_R_16F, kTextHeadDim,
              head_stride,
              d_scores + static_cast<size_t>(grp) * group * per * L * L, CUDA_R_16F, L,
              score_stride, static_cast<const void*>(&zero_h),
              d_attn_gm + static_cast<size_t>(grp) * group * per * L * kTextHeadDim,
              CUDA_R_16F, kTextHeadDim, head_stride, batch, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT));
        }
        gather_q_groupmajor_kernel<<<static_cast<unsigned>((q_total + 255) / 256), 256>>>(
            d_attn_gm, d_attn_group, rows, group);
        CUDA_CHECK(cudaGetLastError());
      }
      gemm(weight(p + "self_attn.o_proj.weight"), d_attn_group, d_scratch_group, rows,
           kTextHidden, kTextHidden);
      residual_add_kernel<<<static_cast<unsigned>((seq_hidden + 255) / 256), 256>>>(
          d_h_group, d_scratch_group, seq_hidden);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(d_ln_group, d_h_group, seq_hidden * sizeof(half),
                            cudaMemcpyDeviceToDevice));
      rmsnorm_kernel<<<rows, 256, 257 * sizeof(float)>>>(
          d_ln_group, weight(p + "post_attention_layernorm.weight"), kTextHidden);
      CUDA_CHECK(cudaGetLastError());
      gemm(weight(p + "mlp.gate_proj.weight"), d_ln_group, d_mlp_g, rows, 6144, kTextHidden);
      gemm(weight(p + "mlp.up_proj.weight"), d_ln_group, d_mlp_u, rows, 6144, kTextHidden);
      silu_mul_kernel<<<static_cast<unsigned>((seq_mlp + 255) / 256), 256>>>(d_mlp_g, d_mlp_u,
                                                                             seq_mlp);
      CUDA_CHECK(cudaGetLastError());
      gemm(weight(p + "mlp.down_proj.weight"), d_mlp_g, d_scratch_group, rows, kTextHidden,
           6144);
      residual_add_kernel<<<static_cast<unsigned>((seq_hidden + 255) / 256), 256>>>(
          d_h_group, d_scratch_group, seq_hidden);
      CUDA_CHECK(cudaGetLastError());
    }
    rmsnorm_kernel<<<rows, 256, 257 * sizeof(float)>>>(
        d_h_group, weight("language_model.model.norm.weight"), kTextHidden);
    CUDA_CHECK(cudaGetLastError());
    // last prompt row of each chunk -> logits -> first ids
    for (int c = 0; c < group; ++c) {
      CUDA_CHECK(cudaMemcpy(d_h + static_cast<size_t>(c) * kTextHidden,
                            d_h_group + (static_cast<size_t>(c) + 1) * L * kTextHidden -
                                kTextHidden,
                            kTextHidden * sizeof(half), cudaMemcpyDeviceToDevice));
    }
    gemm(weight("language_model.lm_head.weight"), d_h, d_logits, group, kVocab, kTextHidden);
    argmax_rows_kernel<<<group, 256>>>(d_logits, group, d_current_ids);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int64_t> first(group);
    CUDA_CHECK(cudaMemcpy(first.data(), d_current_ids, group * sizeof(int64_t),
                          cudaMemcpyDeviceToHost));
    for (int c = 0; c < group; ++c) {
      result.per_chunk_ids[base + c].push_back(first[c]);
      current[base + c] = first[c];
    }
  }

  CUDA_CHECK(cudaEventRecord(prefill_stop));
  CUDA_CHECK(cudaEventRecord(decode_start));
  // ---- lockstep decode over the whole batch ----
  const std::vector<int64_t> eos_ids = {59246, 59253, 59255};
  std::vector<char> finished(chunks, 0);
  auto all_finished = [&]() {
    for (char f : finished)
      if (!f) return false;
    return true;
  };
  for (int c = 0; c < chunks; ++c)
    if (std::find(eos_ids.begin(), eos_ids.end(), current[c]) != eos_ids.end())
      finished[c] = 1;
  int steps = 1;
  for (; steps < options.max_new_tokens && !all_finished(); ++steps) {
    const int position = L + steps - 1;
    CUDA_CHECK(cudaMemcpy(d_gather_ids, current.data(), chunks * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    gather_batch_kernel<<<dim3(1, chunks), 128>>>(
        d_gather_ids, 1, weight("language_model.model.embed_tokens.weight"),
        audio_embeds /*unused for non-audio ids*/, 0, d_ids_batch, chunks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_h, d_ids_batch, static_cast<size_t>(chunks) * kTextHidden * sizeof(half),
                          cudaMemcpyDeviceToDevice));
    for (int layer = 0; layer < kTextLayers; ++layer) {
      const std::string p = "language_model.model.layers." + std::to_string(layer) + ".";
      CUDA_CHECK(cudaMemcpy(d_ln, d_h, static_cast<size_t>(chunks) * kTextHidden * sizeof(half),
                            cudaMemcpyDeviceToDevice));
      rmsnorm_kernel<<<chunks, 256, 257 * sizeof(float)>>>(d_ln,
                                                           weight(p + "input_layernorm.weight"),
                                                           kTextHidden);
      CUDA_CHECK(cudaGetLastError());
      gemm(weight(p + "self_attn.q_proj.weight"), d_ln, d_q, chunks, kTextHidden, kTextHidden);
      gemm(weight(p + "self_attn.k_proj.weight"), d_ln, d_k_row, chunks,
           static_cast<int>(kv_width), kTextHidden);
      gemm(weight(p + "self_attn.v_proj.weight"), d_ln, d_v_row, chunks,
           static_cast<int>(kv_width), kTextHidden);
      rope_at_position_kernel<<<chunks, 64>>>(d_q, d_k_row, d_cos, d_sin, position, chunks);
      CUDA_CHECK(cudaGetLastError());
      // append this step's k/v rows into every chunk's cache (row per chunk)
      half* k_layer_base = d_kv_cache + static_cast<size_t>(layer) * layer_kv_stride;
      half* v_layer_base = k_layer_base + static_cast<size_t>(kMaxSequence) * kv_width;
      append_kv_batched_kernel<<<static_cast<unsigned>((static_cast<size_t>(chunks) * kv_width +
                                                        127) / 128),
                                 128>>>(d_k_row, d_v_row, k_layer_base, v_layer_base,
                                        position, chunks, chunk_kv_stride);
      CUDA_CHECK(cudaGetLastError());
      attention_step_batched_kernel<<<chunks * kTextHeads, 128,
                                      (kMaxSequence + 128) * sizeof(float)>>>(
          d_q, d_kv_cache, d_attn, position + 1, layer, layer_kv_stride, chunk_kv_stride);
      CUDA_CHECK(cudaGetLastError());
      gemm(weight(p + "self_attn.o_proj.weight"), d_attn, d_ln, chunks, kTextHidden, kTextHidden);
      residual_add_kernel<<<static_cast<unsigned>((static_cast<size_t>(chunks) * kTextHidden +
                                                   255) / 256),
                            256>>>(d_h, d_ln, static_cast<size_t>(chunks) * kTextHidden);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(d_ln, d_h, static_cast<size_t>(chunks) * kTextHidden * sizeof(half),
                            cudaMemcpyDeviceToDevice));
      rmsnorm_kernel<<<chunks, 256, 257 * sizeof(float)>>>(
          d_ln, weight(p + "post_attention_layernorm.weight"), kTextHidden);
      CUDA_CHECK(cudaGetLastError());
      const size_t batch_mlp = static_cast<size_t>(chunks) * 6144;
      gemm(weight(p + "mlp.gate_proj.weight"), d_ln, d_mlp_bg, chunks, 6144, kTextHidden);
      gemm(weight(p + "mlp.up_proj.weight"), d_ln, d_mlp_bu, chunks, 6144, kTextHidden);
      silu_mul_kernel<<<static_cast<unsigned>((batch_mlp + 255) / 256), 256>>>(d_mlp_bg,
                                                                               d_mlp_bu, batch_mlp);
      CUDA_CHECK(cudaGetLastError());
      gemm(weight(p + "mlp.down_proj.weight"), d_mlp_bg, d_ln, chunks, kTextHidden, 6144);
      residual_add_kernel<<<static_cast<unsigned>((static_cast<size_t>(chunks) * kTextHidden +
                                                   255) / 256),
                            256>>>(d_h, d_ln, static_cast<size_t>(chunks) * kTextHidden);
      CUDA_CHECK(cudaGetLastError());
    }
    rmsnorm_kernel<<<chunks, 256, 257 * sizeof(float)>>>(
        d_h, weight("language_model.model.norm.weight"), kTextHidden);
    CUDA_CHECK(cudaGetLastError());
    gemm(weight("language_model.lm_head.weight"), d_h, d_logits, chunks, kVocab, kTextHidden);
    argmax_rows_kernel<<<chunks, 256>>>(d_logits, chunks, d_current_ids);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int64_t> next(chunks);
    CUDA_CHECK(cudaMemcpy(next.data(), d_current_ids, chunks * sizeof(int64_t),
                          cudaMemcpyDeviceToHost));
    for (int c = 0; c < chunks; ++c) {
      if (!finished[c]) {
        result.per_chunk_ids[c].push_back(next[c]);
        current[c] = next[c];
        if (std::find(eos_ids.begin(), eos_ids.end(), next[c]) != eos_ids.end()) finished[c] = 1;
      }
    }
  }
  CUDA_CHECK(cudaEventRecord(decode_stop));
  CUDA_CHECK(cudaEventSynchronize(prefill_stop));
  CUDA_CHECK(cudaEventSynchronize(decode_stop));
  CUDA_CHECK(cudaEventElapsedTime(&result.prefill_ms, prefill_start, prefill_stop));
  CUDA_CHECK(cudaEventElapsedTime(&result.decode_ms, decode_start, decode_stop));
  cudaEventDestroy(prefill_start);
  cudaEventDestroy(prefill_stop);
  cudaEventDestroy(decode_start);
  cudaEventDestroy(decode_stop);
  result.steps = steps;
  return result;
}
