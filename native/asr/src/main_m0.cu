// M0 weight/config loader CLI. Two modes:
//   --dump-plan [--hash] : header/config inventory (optionally convert + hash
//                          every tensor on the CPU); never touches CUDA.
//   --load               : full device load, on-GPU BF16->FP16 conversion and
//                          hashing within the VRAM budget (default 16 GiB).
// JSON goes to stdout, diagnostics to stderr. Exit status: 0 success,
// 1 execution/numeric failure, 2 invalid arguments or model layout.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine.h"
#include "mel_frontend.h"
#include "model_loader.h"

namespace {

using InvalidInput = ModelInputError;
using ExecutionError = ModelExecutionError;

// Reads raw little-endian f32 mono PCM and zero-pads to the 30 s chunk.
std::vector<float> read_pcm_padded(const std::string& pcm_path, uint64_t samples_requested,
                                   bool samples_given) {
  std::vector<float> pcm(kChunkSamples, 0.0f);
  std::ifstream in(pcm_path, std::ios::binary);
  if (!in) throw InvalidInput("cannot open PCM file: " + pcm_path);
  in.seekg(0, std::ios::end);
  const uint64_t bytes = static_cast<uint64_t>(in.tellg());
  if (bytes % sizeof(float) != 0) throw InvalidInput("PCM file size is not a multiple of 4");
  const uint64_t file_samples = bytes / sizeof(float);
  if (file_samples > static_cast<uint64_t>(kChunkSamples))
    throw InvalidInput("PCM file exceeds the 30 s chunk (480000 samples)");
  if (samples_given && samples_requested > file_samples)
    throw InvalidInput("--samples exceeds the PCM file contents");
  const uint64_t used = samples_given ? samples_requested : file_samples;
  in.seekg(0, std::ios::beg);
  if (used > 0)
    in.read(reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(used * sizeof(float)));
  if (!in) throw ExecutionError("short read from PCM file");
  return pcm;
}

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (unsigned char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  return out;
}

void print_report(const LoadReport& report, bool with_tensors) {
  std::cout << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"tool\": \"native_asr_m0_loader\",\n"
            << "  \"measurement_kind\": \"correctness_only\",\n"
            << "  \"model_dir\": \"" << json_escape(report.model_dir) << "\",\n"
            << "  \"config_sha256\": \"" << report.config_sha256 << "\",\n"
            << "  \"tensor_count\": " << report.tensor_count << ",\n"
            << "  \"total_params\": " << report.total_params << ",\n"
            << "  \"fp16_bytes\": " << report.fp16_bytes << ",\n"
            << "  \"bf16_to_fp16\": {\"overflow_count\": " << report.overflow_count
            << ", \"nan_count\": " << report.nan_count
            << ", \"max_abs_value\": " << report.max_abs_value << "},\n";
  if (!report.device_name.empty()) {
    std::cout << "  \"device\": {\"name\": \"" << json_escape(report.device_name)
              << "\", \"compute_capability\": \"" << report.compute_capability
              << "\", \"total_memory_bytes\": " << report.device_total_memory << "},\n"
              << "  \"vram\": {\"budget_bytes\": " << report.vram_budget_bytes
              << ", \"weights_device_bytes\": " << report.weights_device_bytes
              << ", \"peak_device_bytes\": " << report.peak_device_bytes << "},\n";
  }
  std::cout << "  \"missing_tensors\": [";
  for (size_t i = 0; i < report.missing_tensors.size(); ++i)
    std::cout << (i ? ", " : "") << '"' << json_escape(report.missing_tensors[i]) << '"';
  std::cout << "],\n  \"unexpected_tensors\": [";
  for (size_t i = 0; i < report.unexpected_tensors.size(); ++i)
    std::cout << (i ? ", " : "") << '"' << json_escape(report.unexpected_tensors[i]) << '"';
  std::cout << "],\n  \"shape_mismatches\": [";
  for (size_t i = 0; i < report.shape_mismatches.size(); ++i)
    std::cout << (i ? ", " : "") << '"' << json_escape(report.shape_mismatches[i]) << '"';
  std::cout << "]";
  if (with_tensors && !report.tensors.empty()) {
    std::cout << ",\n  \"tensors\": [\n";
    for (size_t i = 0; i < report.tensors.size(); ++i) {
      const ConvertedTensor& t = report.tensors[i];
      std::cout << "    {\"name\": \"" << json_escape(t.name) << "\", \"sha256_fp16\": \""
                << t.sha256_fp16 << "\", \"elements\": " << t.element_count
                << ", \"overflow_count\": " << t.overflow_count
                << ", \"nan_count\": " << t.nan_count
                << ", \"max_abs_value\": " << t.max_abs_value << "}"
                << (i + 1 < report.tensors.size() ? "," : "") << "\n";
    }
    std::cout << "  ]";
  }
  std::cout << "\n}\n";
}

void usage(std::ostream& out) {
  out << "usage: glm_asr_m0 --model-dir DIR [--dump-plan [--hash]] [--load]\n"
      << "                  [--max-vram-mib N]\n"
      << "       glm_asr_m0 mel --pcm FILE [--samples N] --out FILE\n"
      << "       glm_asr_m0 encode --model-dir DIR --pcm FILE --out-dir DIR\n"
      << "                        [--samples N] [--gemm-acc 16|32]\n"
      << "  --dump-plan : parse config + header only (no CUDA); --hash converts\n"
      << "                and hashes every tensor on the CPU\n"
      << "  --load      : convert on device, keep FP16 weights resident and hash\n"
      << "                the downloaded bytes (default action)\n"
      << "  --max-vram-mib N : device budget for weights + staging (default 16384,\n"
      << "                the 16 GiB product constraint)\n"
      << "  mel         : Whisper-semantics log-mel frontend correctness mode; PCM\n"
      << "                is raw little-endian f32 mono at 16 kHz, zero-padded to\n"
      << "                480000 samples, output is raw f32 [128, 3000]\n";
}

int run_mel(int argc, char** argv) {
  std::string pcm_path, out_path;
  uint64_t samples = 0;
  bool samples_given = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next_value = [&]() -> std::string {
      if (i + 1 >= argc) throw InvalidInput("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--pcm") {
      pcm_path = next_value();
    } else if (arg == "--out") {
      out_path = next_value();
    } else if (arg == "--samples") {
      const std::string value = next_value();
      char* end = nullptr;
      const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
      if (value.empty() || !end || *end != '\0' || parsed == 0 || parsed > kChunkSamples)
        throw InvalidInput("--samples is out of range: " + value);
      samples = parsed;
      samples_given = true;
    } else {
      throw InvalidInput("unknown mel argument: " + arg);
    }
  }
  if (pcm_path.empty() || out_path.empty())
    throw InvalidInput("mel mode requires --pcm and --out");

  const std::vector<float> pcm = read_pcm_padded(pcm_path, samples, samples_given);
  const MelOutput mel = compute_mel(pcm);
  {
    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw InvalidInput("cannot open output file: " + out_path);
    out.write(reinterpret_cast<const char*>(mel.features.data()),
              static_cast<std::streamsize>(mel.features.size() * sizeof(float)));
    if (!out) throw ExecutionError("short write to output file");
  }

  // Correctness summary only; no timing or performance claims.
  float minimum = mel.features.empty() ? 0.0f : mel.features[0];
  float maximum = mel.features.empty() ? 0.0f : mel.features[0];
  uint64_t non_finite = 0;
  for (float value : mel.features) {
    if (!std::isfinite(value)) ++non_finite;
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  std::cout << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"tool\": \"native_asr_m0_mel\",\n"
            << "  \"measurement_kind\": \"correctness_only\",\n"
            << "  \"mel_bins\": " << kMelBins << ",\n"
            << "  \"frames\": " << kOutputFrames << ",\n"
            << "  \"nonfinite\": " << non_finite << ",\n"
            << "  \"min\": " << minimum << ",\n"
            << "  \"max\": " << maximum << "\n"
            << "}\n";
  return 0;
}

}  // namespace

int run_encode(int argc, char** argv) {
  std::string model_dir, pcm_path, out_dir;
  bool fp16_accumulate = true;
  uint64_t samples = 0;
  bool samples_given = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next_value = [&]() -> std::string {
      if (i + 1 >= argc) throw InvalidInput("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--model-dir") {
      model_dir = next_value();
    } else if (arg == "--pcm") {
      pcm_path = next_value();
    } else if (arg == "--out-dir") {
      out_dir = next_value();
    } else if (arg == "--samples") {
      const std::string value = next_value();
      char* end = nullptr;
      const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
      if (value.empty() || !end || *end != '\0' || parsed == 0 || parsed > kChunkSamples)
        throw InvalidInput("--samples is out of range: " + value);
      samples = parsed;
      samples_given = true;
    } else if (arg == "--gemm-acc") {
      const std::string value = next_value();
      if (value == "16") fp16_accumulate = true;
      else if (value == "32") fp16_accumulate = false;
      else throw InvalidInput("--gemm-acc must be 16 or 32");
    } else {
      throw InvalidInput("unknown encode argument: " + arg);
    }
  }
  if (model_dir.empty() || pcm_path.empty() || out_dir.empty())
    throw InvalidInput("encode mode requires --model-dir, --pcm and --out-dir");

  const std::vector<float> pcm = read_pcm_padded(pcm_path, samples, samples_given);
  const MelOutput mel = compute_mel(pcm);
  DeviceWeights weights = load_weights_resident(model_dir, 16384ull * 1024 * 1024);

  EncodeOptions options;
  options.fp16_accumulate = fp16_accumulate;
  const EncodeOutput encoded = run_encode(mel.features, &weights, options);

  const auto write_raw = [&](const std::string& name, const std::vector<float>& data) {
    const std::string path = out_dir + "/" + name;
    std::ofstream out(path, std::ios::binary);
    if (!out) throw InvalidInput("cannot open output file: " + path);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!out) throw ExecutionError("short write to output file: " + path);
  };
  write_raw("mel.f32", mel.features);
  write_raw("conv.f32", encoded.conv_hidden);
  const size_t hidden_count = encoded.conv_hidden.size();
  for (size_t layer = 0; layer < encoded.layer_hidden.size() / hidden_count; ++layer) {
    char name[32];
    std::snprintf(name, sizeof(name), "layer_%02zu.f32", layer);
    write_raw(name,
              std::vector<float>(encoded.layer_hidden.begin() + hidden_count * layer,
                                 encoded.layer_hidden.begin() + hidden_count * (layer + 1)));
  }
  write_raw("norm.f32", encoded.final_norm);
  write_raw("projector.f32", encoded.projector);

  std::cout << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"tool\": \"native_asr_m0_encode\",\n"
            << "  \"measurement_kind\": \"correctness_only\",\n"
            << "  \"gemm_acc\": \"" << (fp16_accumulate ? "fp16" : "fp32") << "\",\n"
            << "  \"out_dir\": \"" << json_escape(out_dir) << "\",\n"
            << "  \"files\": [\"mel.f32\", \"conv.f32\", \"layer_00.f32..layer_31.f32\","
            << " \"norm.f32\", \"projector.f32\"],\n"
            << "  \"projector_shape\": [375, 2048]\n"
            << "}\n";
  return 0;
}

int run_transcribe(int argc, char** argv) {
  std::string model_dir, pcm_path, prompt_path;
  bool fp16_accumulate = true;
  int max_new_tokens = 256;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next_value = [&]() -> std::string {
      if (i + 1 >= argc) throw InvalidInput("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--model-dir") model_dir = next_value();
    else if (arg == "--pcm") pcm_path = next_value();
    else if (arg == "--prompt-ids") prompt_path = next_value();
    else if (arg == "--gemm-acc") {
      const std::string value = next_value();
      if (value == "16") fp16_accumulate = true;
      else if (value == "32") fp16_accumulate = false;
      else throw InvalidInput("--gemm-acc must be 16 or 32");
    } else if (arg == "--max-new-tokens") {
      const std::string value = next_value();
      char* end = nullptr;
      const long parsed = std::strtol(value.c_str(), &end, 10);
      if (value.empty() || !end || *end != '\0' || parsed <= 0 || parsed > 1024)
        throw InvalidInput("--max-new-tokens is out of range: " + value);
      max_new_tokens = static_cast<int>(parsed);
    } else {
      throw InvalidInput("unknown transcribe argument: " + arg);
    }
  }
  if (model_dir.empty() || pcm_path.empty() || prompt_path.empty())
    throw InvalidInput("transcribe mode requires --model-dir, --pcm and --prompt-ids");

  std::vector<int64_t> prompt_ids;
  {
    std::ifstream in(prompt_path);
    if (!in) throw InvalidInput("cannot open prompt ids file: " + prompt_path);
    long value;
    while (in >> value) prompt_ids.push_back(value);
    if (prompt_ids.empty()) throw InvalidInput("prompt ids file is empty");
    for (int64_t id : prompt_ids)
      if (id < 0 || id > 59263) throw InvalidInput("prompt id out of vocab range");
  }

  const std::vector<float> pcm = read_pcm_padded(pcm_path, 0, false);
  const MelOutput mel = compute_mel(pcm);
  DeviceWeights weights = load_weights_resident(model_dir, 16384ull * 1024 * 1024);

  std::vector<uint16_t> audio_device(375 * 2048);
  void* device_audio = nullptr;
  EncodeOptions encode_options;
  encode_options.fp16_accumulate = fp16_accumulate;
  if (cudaMalloc(&device_audio, audio_device.size() * sizeof(uint16_t)) != cudaSuccess)
    throw ExecutionError("cudaMalloc failed for audio embeds");
  run_encode(mel.features, &weights, encode_options,
             static_cast<uint16_t*>(device_audio));

  DecodeOptions decode_options;
  decode_options.fp16_accumulate = fp16_accumulate;
  decode_options.max_new_tokens = max_new_tokens;
  const DecodeResult decoded =
      run_decode(&weights, static_cast<const uint16_t*>(device_audio), prompt_ids,
                 decode_options);
  cudaFree(device_audio);

  std::cout << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"tool\": \"native_asr_m0_transcribe\",\n"
            << "  \"measurement_kind\": \"correctness_only\",\n"
            << "  \"gemm_acc\": \"" << (fp16_accumulate ? "fp16" : "fp32") << "\",\n"
            << "  \"stop_reason\": " << decoded.stop_reason << ",\n"
            << "  \"generated_ids\": [";
  for (size_t i = 0; i < decoded.generated_ids.size(); ++i)
    std::cout << (i ? ", " : "") << decoded.generated_ids[i];
  std::cout << "]\n}\n";
  return 0;
}

int main(int argc, char** argv) {
  try {
    const std::string first = argc >= 2 ? argv[1] : "";
    if (first == "mel") return run_mel(argc, argv);
    if (first == "encode") return run_encode(argc, argv);
    if (first == "transcribe") return run_transcribe(argc, argv);

    std::string model_dir;
    bool dump_plan = false;
    bool hash = false;
    bool load = false;
    uint64_t max_vram_mib = 16384;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto next_value = [&]() -> std::string {
        if (i + 1 >= argc) throw InvalidInput("missing value for " + arg);
        return argv[++i];
      };
      if (arg == "--model-dir") {
        model_dir = next_value();
      } else if (arg == "--dump-plan") {
        dump_plan = true;
      } else if (arg == "--hash") {
        hash = true;
      } else if (arg == "--load") {
        load = true;
      } else if (arg == "--max-vram-mib") {
        const std::string value = next_value();
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
        if (value.empty() || !end || *end != '\0' || parsed == 0 || parsed > (1ull << 22))
          throw InvalidInput("--max-vram-mib is out of range: " + value);
        max_vram_mib = parsed;
      } else if (arg == "--help" || arg == "-h") {
        usage(std::cout);
        return 0;
      } else {
        throw InvalidInput("unknown argument: " + arg);
      }
    }
    if (model_dir.empty()) throw InvalidInput("--model-dir is required");
    if (hash && !dump_plan)
      throw InvalidInput("--hash is only valid together with --dump-plan");
    if (dump_plan && load)
      throw InvalidInput("--dump-plan and --load are mutually exclusive");
    if (!dump_plan && !load) load = true;  // default action

    const LoadReport report = dump_plan ? dump_plan_pass(model_dir, hash)
                                        : device_load_pass(model_dir, max_vram_mib * 1024 * 1024);
    print_report(report, dump_plan ? hash : true);
    if (report.overflow_count > 0 || report.nan_count > 0) {
      std::cerr << "warning: BF16->FP16 conversion produced " << report.overflow_count
                << " overflows and " << report.nan_count << " NaN inputs\n";
    }
    return 0;
  } catch (const InvalidInput& error) {
    std::cerr << "invalid input: " << error.what() << "\n";
    usage(std::cerr);
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
