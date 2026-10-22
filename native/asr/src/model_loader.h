// Interface for the M0 weight/config loader. The loader is a correctness tool:
// it parses the official safetensors checkpoint, converts BF16 weights to FP16
// and reports per-tensor SHA-256 fingerprints of the converted bytes. It never
// measures performance and its output is not ASR RTFx evidence.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Shared error taxonomy so the CLI can map exceptions onto its exit codes:
// invalid model layout / arguments -> exit 2, execution failures -> exit 1.
class ModelInputError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class ModelExecutionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct TensorSpec {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype;  // as written in the file, e.g. "BF16"
  uint64_t begin = 0; // byte offset within the data region
  uint64_t end = 0;

  uint64_t nbytes() const { return end - begin; }
  uint64_t element_count() const;
};

// Model dimensions derived from config.json. Field names follow the official
// config keys (audio_config.*, text_config.*).
struct AsrModelConfig {
  int audio_hidden = 0;
  int audio_intermediate = 0;
  int audio_layers = 0;
  int audio_heads = 0;
  int audio_head_dim = 0;
  int audio_mel_bins = 0;
  int text_hidden = 0;
  int text_intermediate = 0;
  int text_layers = 0;
  int text_heads = 0;
  int text_kv_heads = 0;
  int text_head_dim = 0;
  int vocab_size = 0;
  int projector_mid = 4096; // not exposed in config.json; validated for
                            // internal consistency (linear_2.in == linear_1.out)
};

struct ConvertedTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::string sha256_fp16;
  uint64_t element_count = 0;
  uint64_t overflow_count = 0; // finite BF16 magnitudes that became +/-Inf
  uint64_t nan_count = 0;      // NaN inputs (converted to canonical half NaN)
  double max_abs_value = 0.0;  // of the original BF16 values
};

struct LoadReport {
  std::string model_dir;
  std::string config_sha256;
  uint64_t tensor_count = 0;
  uint64_t total_params = 0;
  uint64_t fp16_bytes = 0;
  uint64_t overflow_count = 0;
  uint64_t nan_count = 0;
  double max_abs_value = 0.0;
  std::vector<std::string> missing_tensors;      // expected but absent
  std::vector<std::string> unexpected_tensors;   // present but not expected
  std::vector<std::string> shape_mismatches;     // "name: expected ... got ..."
  std::vector<ConvertedTensor> tensors;          // sorted by name
  // Device-only fields; zero/empty for the dump-plan pass.
  uint64_t weights_device_bytes = 0;
  uint64_t peak_device_bytes = 0;
  uint64_t vram_budget_bytes = 0;
  std::string device_name;
  std::string compute_capability;
  uint64_t device_total_memory = 0;
};

// Reads <model_dir>/config.json and validates the expected dimension fields.
AsrModelConfig parse_model_config(const std::string& model_dir);

// Parses the safetensors header. Returns tensors sorted by name.
// data_region_offset receives the byte offset where tensor data starts.
std::vector<TensorSpec> read_safetensors_header(const std::string& path,
                                                uint64_t* data_region_offset);

// Compares the checkpoint inventory against the names and shapes implied by
// the official architecture template and config values.
void validate_inventory(const std::vector<TensorSpec>& tensors,
                        const AsrModelConfig& cfg, LoadReport* report);

// CPU-only pass: parses header/config, optionally converts every tensor and
// hashes the FP16 bytes. Never initializes a CUDA context.
LoadReport dump_plan_pass(const std::string& model_dir, bool hash_tensors);

// Device pass: uploads each BF16 tensor, converts on the GPU, keeps all FP16
// weights resident within max_vram_bytes, then hashes downloaded bytes.
LoadReport device_load_pass(const std::string& model_dir, uint64_t max_vram_bytes);
