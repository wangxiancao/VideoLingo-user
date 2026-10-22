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
  const half* k_head = k_cache + static_cast<size_t>(kv_head) * kTextHeadDim;
  const half* v_head = v_cache + static_cast<size_t>(kv_head) * kTextHeadDim;
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
  (void)k_head;
  (void)v_head;
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
    // Prefill: run every prompt row through the cache; the last row yields
    // the first generated token.
    for (size_t pos = 0; pos < prompt_length; ++pos) {
      CUDA_CHECK(cudaMemcpy(d_h, d_x + pos * hidden_count, hidden_count * sizeof(half),
                            cudaMemcpyDeviceToDevice));
      const bool last = pos + 1 == prompt_length;
      const int64_t next = forward_position(static_cast<int>(pos), last);
      if (last) {
        result.generated_ids.push_back(next);
        if (std::find(eos_ids.begin(), eos_ids.end(), next) != eos_ids.end()) {
          result.stop_reason = 0;
          cudaFree(d_prompt_ids); cudaFree(d_x); cudaFree(d_row); cudaFree(d_h);
          cudaFree(d_ln); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_attn);
          cudaFree(d_mlp_g); cudaFree(d_mlp_u); cudaFree(d_logits); cudaFree(d_kv_cache);
          cudaFree(d_cos); cudaFree(d_sin);
          return result;
        }
      }
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
