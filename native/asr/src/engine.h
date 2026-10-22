// M0 Task 3: native GLM-ASR encoder + projector over resident FP16 weights.
//
// Pipeline (single 30 s chunk): mel [128,3000] f32 -> fp16 -> conv1(k3,s1)+gelu
// -> conv2(k3,s2)+gelu -> [1500,1280] -> 32 x pre-LN layers (q/k/v/o with
// partial RoPE, fp32 softmax, no mask) -> final LN -> reshape [375,5120] ->
// projector linear_1(4096)+gelu -> linear_2(2048) -> [375,2048] fp16.
//
// Projection GEMMs follow the selected accumulation mode (fp16 or fp32
// accumulate). Attention score and PV matmuls always accumulate in fp32 for
// numerical stability; production throughput work (FA2, graphs) is M1+.
// Correctness tool: no timing, no ASR RTFx claims.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Resident FP16 weights on the device, loaded from the official checkpoint.
class DeviceWeights {
 public:
  struct Entry {
    uint16_t* ptr = nullptr;         // device FP16 data
    std::vector<int64_t> shape;
    uint64_t elements = 0;
  };
  DeviceWeights() = default;
  DeviceWeights(const DeviceWeights&) = delete;
  DeviceWeights& operator=(const DeviceWeights&) = delete;
  DeviceWeights(DeviceWeights&& other) noexcept;
  DeviceWeights& operator=(DeviceWeights&& other) noexcept;
  ~DeviceWeights();

  bool loaded() const { return arena_ != nullptr; }
  uint64_t device_bytes() const { return bytes_; }
  // Returns nullptr when the tensor is absent.
  const Entry* find(const std::string& name) const;

 private:
  friend DeviceWeights load_weights_resident(const std::string&, uint64_t);
  uint16_t* arena_ = nullptr;
  uint64_t bytes_ = 0;
  std::vector<std::pair<std::string, Entry>> entries_;  // sorted by name
};

// Loads and converts every tensor (BF16 -> FP16 on device) into one arena.
// Throws ModelExecutionError when the estimated peak exceeds max_vram_bytes.
DeviceWeights load_weights_resident(const std::string& model_dir, uint64_t max_vram_bytes);

struct EncodeOutput {
  std::vector<float> conv_hidden;    // [1500,1280] post conv2+gelu, transposed
  std::vector<float> layer_hidden;   // [32][1500,1280] post each encoder layer
  std::vector<float> final_norm;     // [1500,1280]
  std::vector<float> projector;      // [375,2048]
};

struct EncodeOptions {
  bool fp16_accumulate = true;  // projection GEMM accumulation mode
  uint64_t max_vram_bytes = 16384ull * 1024 * 1024;
};

// mel_features: [128,3000] row-major. Runs the full audio tower + projector.
// When projector_device_out is non-null (375*2048 halves, caller-allocated),
// the FP16 projector output is additionally copied there before returning.
EncodeOutput run_encode(const std::vector<float>& mel_features,
                        DeviceWeights* weights, const EncodeOptions& options,
                        uint16_t* projector_device_out = nullptr);

struct DecodeOptions {
  bool fp16_accumulate = true;
  int max_new_tokens = 256;
  bool prefill_batched = true;  // sequence-level prefill vs per-token (M0 path)
};

struct DecodeResult {
  std::vector<int64_t> generated_ids;  // after the prompt
  int64_t stop_reason = 0;             // 0 eos, 1 budget exhausted
};

// audio_embeds_device: [375, 2048] FP16 device buffer from run_encode.
DecodeResult run_decode(DeviceWeights* weights, const uint16_t* audio_embeds_device,
                        const std::vector<int64_t>& prompt_ids, const DecodeOptions& options);
