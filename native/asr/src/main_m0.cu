// M0 weight/config loader CLI. Two modes:
//   --dump-plan [--hash] : header/config inventory (optionally convert + hash
//                          every tensor on the CPU); never touches CUDA.
//   --load               : full device load, on-GPU BF16->FP16 conversion and
//                          hashing within the VRAM budget (default 16 GiB).
// JSON goes to stdout, diagnostics to stderr. Exit status: 0 success,
// 1 execution/numeric failure, 2 invalid arguments or model layout.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_loader.h"

namespace {

using InvalidInput = ModelInputError;
using ExecutionError = ModelExecutionError;

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
      << "  --dump-plan : parse config + header only (no CUDA); --hash converts\n"
      << "                and hashes every tensor on the CPU\n"
      << "  --load      : convert on device, keep FP16 weights resident and hash\n"
      << "                the downloaded bytes (default action)\n"
      << "  --max-vram-mib N : device budget for weights + staging (default 16384,\n"
      << "                the 16 GiB product constraint)\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
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
