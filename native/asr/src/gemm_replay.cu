#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kMaxCount = 1000000;
constexpr int kThreads = 256;
constexpr int kErrorSamples = 64;

class InvalidInput : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class ExecutionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

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

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    const cudaError_t status_ = (expr);                                          \
    if (status_ != cudaSuccess) throw ExecutionError(cuda_error(status_, #expr));\
  } while (false)

#define CUBLAS_CHECK(expr)                                                        \
  do {                                                                           \
    const cublasStatus_t status_ = (expr);                                        \
    if (status_ != CUBLAS_STATUS_SUCCESS)                                         \
      throw ExecutionError(cublas_error(status_, #expr));                         \
  } while (false)

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw InvalidInput("allocation-size overflow in device buffer");
    if (count_ != 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
  }
  ~DeviceBuffer() noexcept {
    if (ptr_) {
      const cudaError_t status = cudaFree(ptr_);
      if (status != cudaSuccess) std::cerr << cuda_error(status, "cudaFree") << '\n';
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)), count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) {
        const cudaError_t status = cudaFree(ptr_);
        if (status != cudaSuccess) std::cerr << cuda_error(status, "cudaFree") << '\n';
      }
      ptr_ = std::exchange(other.ptr_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  std::size_t size() const { return count_; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

class CudaStream {
 public:
  CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~CudaStream() noexcept {
    if (stream_) {
      const cudaError_t status = cudaStreamDestroy(stream_);
      if (status != cudaSuccess) std::cerr << cuda_error(status, "cudaStreamDestroy") << '\n';
    }
  }
  operator cudaStream_t() const { return stream_; }
 private:
  cudaStream_t stream_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
  ~CudaEvent() noexcept {
    if (event_) {
      const cudaError_t status = cudaEventDestroy(event_);
      if (status != cudaSuccess) std::cerr << cuda_error(status, "cudaEventDestroy") << '\n';
    }
  }
  operator cudaEvent_t() const { return event_; }
 private:
  cudaEvent_t event_ = nullptr;
};

class CublasHandle {
 public:
  explicit CublasHandle(cudaStream_t stream) {
    CUBLAS_CHECK(cublasCreate(&handle_));
    const cublasStatus_t status = cublasSetStream(handle_, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(handle_);
      handle_ = nullptr;
      throw ExecutionError(cublas_error(status, "cublasSetStream(handle_, stream)"));
    }
  }
  ~CublasHandle() noexcept {
    if (handle_) {
      const cublasStatus_t status = cublasDestroy(handle_);
      if (status != CUBLAS_STATUS_SUCCESS)
        std::cerr << cublas_error(status, "cublasDestroy") << '\n';
    }
  }
  operator cublasHandle_t() const { return handle_; }
 private:
  cublasHandle_t handle_ = nullptr;
};

enum class Mode { kFp16Acc32, kFp16Acc16, kInt8 };

const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::kFp16Acc32: return "fp16_acc32";
    case Mode::kFp16Acc16: return "fp16_acc16";
    case Mode::kInt8: return "int8";
  }
  return "unknown";
}

Mode parse_mode(const std::string& value) {
  if (value == "fp16_acc32") return Mode::kFp16Acc32;
  if (value == "fp16_acc16") return Mode::kFp16Acc16;
  if (value == "int8") return Mode::kInt8;
  throw InvalidInput("unsupported --mode: " + value);
}

struct Case {
  std::string id;
  int m;
  int n;
  int k;
  int repeats;
};

struct Options {
  bool self_test = false;
  std::string manifest;
  Mode mode = Mode::kFp16Acc32;
  bool mode_set = false;
  int runs = 3;
  int warmups = 2;
  int iterations = 5;
  std::size_t max_buffer_bytes = std::numeric_limits<std::size_t>::max();
};

struct Result {
  Case spec;
  std::vector<double> times_ms;
  double median_ms = 0.0;
  double max_abs_error = 0.0;
  double relative_l2_error = 0.0;
  bool finite = false;
  std::size_t buffer_bytes = 0;
  int checked_elements = 0;
  bool full_check = false;
};

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw InvalidInput(std::string("allocation-size overflow for ") + label);
  return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw InvalidInput(std::string("allocation-size overflow for ") + label);
  return a + b;
}

long long parse_integer(const std::string& text, const std::string& label,
                        long long minimum, long long maximum) {
  if (text.empty()) throw InvalidInput(label + " must be an integer");
  errno = 0;
  char* end = nullptr;
  const long long value = std::strtoll(text.c_str(), &end, 10);
  if (errno == ERANGE || end == text.c_str() || *end != '\0' || value < minimum || value > maximum)
    throw InvalidInput(label + " is out of range: " + text);
  return value;
}

bool valid_id(const std::string& id) {
  if (id.empty()) return false;
  for (unsigned char c : id) {
    const bool ascii_alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                             (c >= '0' && c <= '9');
    if (!(ascii_alnum || c == '_' || c == '-' || c == '.')) return false;
  }
  return true;
}

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const std::size_t tab = line.find('\t', start);
    fields.push_back(line.substr(start, tab == std::string::npos ? tab : tab - start));
    if (tab == std::string::npos) return fields;
    start = tab + 1;
  }
}

std::vector<Case> read_manifest(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw InvalidInput("manifest not found or unreadable: " + path);
  std::string line;
  if (!std::getline(input, line)) throw InvalidInput("manifest is empty");
  if (!line.empty() && line.back() == '\r') line.pop_back();
  if (line.rfind("\xEF\xBB\xBF", 0) == 0) line.erase(0, 3);
  if (line != "id\tm\tn\tk\trepeats")
    throw InvalidInput("manifest header must be: id\\tm\\tn\\tk\\trepeats");

  std::set<std::string> ids;
  std::vector<Case> cases;
  int line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) throw InvalidInput("blank manifest row at line " + std::to_string(line_number));
    const auto fields = split_tabs(line);
    if (fields.size() != 5)
      throw InvalidInput("manifest row must contain exactly 5 columns at line " + std::to_string(line_number));
    if (!valid_id(fields[0]))
      throw InvalidInput("invalid or ambiguous case id at line " + std::to_string(line_number));
    if (!ids.insert(fields[0]).second) throw InvalidInput("duplicate case id: " + fields[0]);
    cases.push_back({fields[0],
                     static_cast<int>(parse_integer(fields[1], "m", 1, INT_MAX)),
                     static_cast<int>(parse_integer(fields[2], "n", 1, INT_MAX)),
                     static_cast<int>(parse_integer(fields[3], "k", 1, INT_MAX)),
                     static_cast<int>(parse_integer(fields[4], "repeats", 1, INT_MAX))});
  }
  if (input.bad()) throw InvalidInput("failed while reading manifest: " + path);
  if (cases.empty()) throw InvalidInput("manifest contains no cases");
  return cases;
}

Options parse_options(int argc, char** argv) {
  Options options;
  std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (!seen.insert(arg).second) throw InvalidInput("duplicate option: " + arg);
    auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw InvalidInput(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--self-test") options.self_test = true;
    else if (arg == "--manifest") options.manifest = value("--manifest");
    else if (arg == "--mode") { options.mode = parse_mode(value("--mode")); options.mode_set = true; }
    else if (arg == "--runs") options.runs = static_cast<int>(parse_integer(value("--runs"), "--runs", 1, kMaxCount));
    else if (arg == "--warmups") options.warmups = static_cast<int>(parse_integer(value("--warmups"), "--warmups", 0, kMaxCount));
    else if (arg == "--iterations") options.iterations = static_cast<int>(parse_integer(value("--iterations"), "--iterations", 1, kMaxCount));
    else if (arg == "--max-buffer-mib") {
      const auto mib = static_cast<std::size_t>(parse_integer(value("--max-buffer-mib"), "--max-buffer-mib", 1, LLONG_MAX));
      options.max_buffer_bytes = checked_mul(mib, 1024ULL * 1024ULL, "--max-buffer-mib");
    } else {
      throw InvalidInput("unknown argument: " + arg);
    }
  }
  if (argc == 1) throw InvalidInput("expected --self-test or --manifest");
  if (options.self_test) {
    if (argc != 2) throw InvalidInput("--self-test cannot be combined with other options");
  } else {
    if (options.manifest.empty()) throw InvalidInput("--manifest is required");
    if (!options.mode_set) throw InvalidInput("--mode is required");
  }
  return options;
}

__host__ __device__ float x_value(std::size_t index, int seed) {
  return static_cast<float>((static_cast<int>((index + seed * 7ULL) % 17ULL) - 8)) / 16.0f;
}

__host__ __device__ float w_value(std::size_t index, int seed) {
  return static_cast<float>((static_cast<int>((index + seed * 5ULL) % 13ULL) - 6)) / 16.0f;
}

__global__ void init_half(half* values, std::size_t count, bool weights, int seed) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < count; i += gridDim.x * static_cast<std::size_t>(blockDim.x)) {
    values[i] = __float2half_rn(weights ? w_value(i, seed) : x_value(i, seed));
  }
}

__global__ void quantize_int8(const half* input, std::int8_t* output, std::size_t count) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < count; i += gridDim.x * static_cast<std::size_t>(blockDim.x)) {
    const int value = __float2int_rn(__half2float(input[i]) * 16.0f);
    output[i] = static_cast<std::int8_t>(max(-127, min(127, value)));
  }
}

__global__ void dequantize_int32(const std::int32_t* input, half* output, std::size_t count) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < count; i += gridDim.x * static_cast<std::size_t>(blockDim.x)) {
    output[i] = __float2half_rn(static_cast<float>(input[i]) / 256.0f);
  }
}

int blocks_for(std::size_t count) {
  return static_cast<int>(std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
}

std::size_t required_bytes(const Case& spec, Mode mode) {
  const std::size_t x = checked_mul(static_cast<std::size_t>(spec.m), spec.k, "X elements");
  const std::size_t w = checked_mul(static_cast<std::size_t>(spec.n), spec.k, "W elements");
  const std::size_t y = checked_mul(static_cast<std::size_t>(spec.m), spec.n, "output elements");
  std::size_t total = checked_mul(checked_add(checked_add(x, w, "FP16 inputs"), y, "FP16 output"), sizeof(half), "FP16 buffers");
  if (mode == Mode::kInt8) {
    // Deterministic quantized inputs are bounded by 8 and 6, so 48*k is the
    // largest possible dot-product magnitude.
    if (spec.k > INT_MAX / 48)
      throw InvalidInput("k can overflow the deterministic INT32 accumulator");
    total = checked_add(total, x, "quantized activation");
    total = checked_add(total, w, "quantized weight");
    total = checked_add(total, checked_mul(y, sizeof(std::int32_t), "INT32 output"), "INT8 buffers");
  }
  return total;
}

void launch_fp16(cublasHandle_t handle, Mode mode, const Case& spec,
                 const half* x, const half* w, half* y) {
  if (mode == Mode::kFp16Acc32) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              spec.n, spec.m, spec.k, &alpha,
                              w, CUDA_R_16F, spec.k, x, CUDA_R_16F, spec.k,
                              &beta, y, CUDA_R_16F, spec.n,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              spec.n, spec.m, spec.k, &alpha,
                              w, CUDA_R_16F, spec.k, x, CUDA_R_16F, spec.k,
                              &beta, y, CUDA_R_16F, spec.n,
                              CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}

void launch_int8(cublasHandle_t handle, cudaStream_t stream, const Case& spec,
                 const half* x, std::int8_t* x8, const std::int8_t* w8,
                 std::int32_t* y32, half* y) {
  const std::size_t x_count = static_cast<std::size_t>(spec.m) * spec.k;
  const std::size_t y_count = static_cast<std::size_t>(spec.m) * spec.n;
  quantize_int8<<<blocks_for(x_count), kThreads, 0, stream>>>(x, x8, x_count);
  CUDA_CHECK(cudaGetLastError());
  const std::int32_t alpha = 1, beta = 0;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            spec.n, spec.m, spec.k, &alpha,
                            w8, CUDA_R_8I, spec.k, x8, CUDA_R_8I, spec.k,
                            &beta, y32, CUDA_R_32I, spec.n,
                            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  dequantize_int32<<<blocks_for(y_count), kThreads, 0, stream>>>(y32, y, y_count);
  CUDA_CHECK(cudaGetLastError());
}

double reference_value(const Case& spec, Mode mode, int row, int column, int seed) {
  if (mode == Mode::kInt8) {
    std::int32_t integer_sum = 0;
    for (int p = 0; p < spec.k; ++p) {
      const float x = x_value(static_cast<std::size_t>(row) * spec.k + p, seed);
      const float w = w_value(static_cast<std::size_t>(column) * spec.k + p, seed);
      const int x8 = static_cast<int>(std::nearbyint(x * 16.0f));
      const int w8 = static_cast<int>(std::nearbyint(w * 16.0f));
      integer_sum += x8 * w8;
    }
    return static_cast<float>(integer_sum) / 256.0f;
  }
  float sum = 0.0f;
  for (int p = 0; p < spec.k; ++p) {
    const float x = x_value(static_cast<std::size_t>(row) * spec.k + p, seed);
    const float w = w_value(static_cast<std::size_t>(column) * spec.k + p, seed);
    sum = std::fma(x, w, sum);
  }
  return sum;
}

void check_numeric(Result& result, Mode mode, const half* output, int seed) {
  const std::size_t count = static_cast<std::size_t>(result.spec.m) * result.spec.n;
  result.full_check = count <= 4096;
  result.checked_elements = static_cast<int>(result.full_check ? count : std::min<std::size_t>(count, kErrorSamples));
  double error_squared = 0.0, reference_squared = 0.0;
  bool finite = true;
  for (int sample = 0; sample < result.checked_elements; ++sample) {
    const std::size_t index = result.full_check
        ? static_cast<std::size_t>(sample)
        : (static_cast<std::size_t>(sample) * (count - 1)) / (result.checked_elements - 1);
    half actual_half;
    CUDA_CHECK(cudaMemcpy(&actual_half, output + index, sizeof(half), cudaMemcpyDeviceToHost));
    const double actual = __half2float(actual_half);
    const int row = static_cast<int>(index / result.spec.n);
    const int column = static_cast<int>(index % result.spec.n);
    const double expected = reference_value(result.spec, mode, row, column, seed);
    const double error = actual - expected;
    finite = finite && std::isfinite(actual);
    result.max_abs_error = std::max(result.max_abs_error, std::abs(error));
    error_squared += error * error;
    reference_squared += expected * expected;
  }
  result.relative_l2_error = std::sqrt(error_squared / std::max(reference_squared, 1e-30));
  result.finite = finite;
  const double tolerance = mode == Mode::kFp16Acc16 ? 0.12 : 0.02;
  if (!finite || result.relative_l2_error > tolerance) {
    std::ostringstream message;
    message << "numeric sanity check failed for " << result.spec.id
            << " in " << mode_name(mode) << ": relative_l2_error="
            << result.relative_l2_error << " tolerance=" << tolerance;
    throw ExecutionError(message.str());
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

Result execute_case(const Case& spec, Mode mode, int runs, int warmups,
                    int iterations, std::size_t max_buffer_bytes, int seed = 0) {
  Result result;
  result.spec = spec;
  result.buffer_bytes = required_bytes(spec, mode);
  if (result.buffer_bytes > max_buffer_bytes) {
    std::ostringstream message;
    message << "case " << spec.id << " requires " << result.buffer_bytes
            << " buffer bytes, exceeding --max-buffer-mib cap of " << max_buffer_bytes;
    throw InvalidInput(message.str());
  }

  const std::size_t x_count = static_cast<std::size_t>(spec.m) * spec.k;
  const std::size_t w_count = static_cast<std::size_t>(spec.n) * spec.k;
  const std::size_t y_count = static_cast<std::size_t>(spec.m) * spec.n;
  CudaStream stream;
  CublasHandle handle(stream);
  DeviceBuffer<half> x(x_count), w(w_count), y(y_count);
  DeviceBuffer<std::int8_t> x8(mode == Mode::kInt8 ? x_count : 0);
  DeviceBuffer<std::int8_t> w8(mode == Mode::kInt8 ? w_count : 0);
  DeviceBuffer<std::int32_t> y32(mode == Mode::kInt8 ? y_count : 0);

  init_half<<<blocks_for(x_count), kThreads, 0, stream>>>(x.get(), x_count, false, seed);
  CUDA_CHECK(cudaGetLastError());
  init_half<<<blocks_for(w_count), kThreads, 0, stream>>>(w.get(), w_count, true, seed);
  CUDA_CHECK(cudaGetLastError());
  if (mode == Mode::kInt8) {
    quantize_int8<<<blocks_for(w_count), kThreads, 0, stream>>>(w.get(), w8.get(), w_count);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto launch = [&]() {
    if (mode == Mode::kInt8) launch_int8(handle, stream, spec, x.get(), x8.get(), w8.get(), y32.get(), y.get());
    else launch_fp16(handle, mode, spec, x.get(), w.get(), y.get());
  };
  for (int warmup = 0; warmup < warmups; ++warmup) {
    for (int iteration = 0; iteration < iterations; ++iteration) launch();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CudaEvent start, stop;
  result.times_ms.reserve(runs);
  for (int run = 0; run < runs; ++run) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iteration = 0; iteration < iterations; ++iteration) launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    result.times_ms.push_back(static_cast<double>(elapsed) / iterations);
  }
  result.median_ms = median(result.times_ms);
  check_numeric(result, mode, y.get(), seed);
  return result;
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
        else out << c;
    }
  }
  return out.str();
}

void print_results(Mode mode, const std::vector<Result>& results) {
  cudaDeviceProp properties{};
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::cout << std::setprecision(10);
  std::cout << "{\n  \"schema_version\": 1,\n"
            << "  \"benchmark\": \"native_gemm_replay\",\n"
            << "  \"mode\": \"" << mode_name(mode) << "\",\n"
            << "  \"measurement_kind\": \"isolated_gemm\",\n"
            << "  \"timing_scope\": \"one manifest-shaped GEMM per iteration using unoptimized real batch/token-shaped dimensions; repeats is metadata and is not executed inside an iteration\",\n"
            << "  \"workload_disclosure\": \"repeats models layer/task call counts but timing reuses one X/W working set; it does not measure distinct per-layer weight traffic, and decoder cases are not whole-model throughput estimates\",\n"
            << "  \"device\": {\"name\": \"" << json_escape(properties.name)
            << "\", \"compute_capability\": \"" << properties.major << "." << properties.minor
            << "\", \"total_memory_bytes\": " << properties.totalGlobalMem << "},\n"
            << "  \"cases\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const Result& result = results[i];
    std::cout << "    {\"id\": \"" << json_escape(result.spec.id) << "\", \"m\": " << result.spec.m
              << ", \"n\": " << result.spec.n << ", \"k\": " << result.spec.k
              << ", \"repeats\": " << result.spec.repeats << ", \"times_ms\": [";
    for (std::size_t j = 0; j < result.times_ms.size(); ++j) {
      if (j) std::cout << ", ";
      std::cout << result.times_ms[j];
    }
    std::cout << "], \"median_ms\": " << result.median_ms
              << ", \"max_abs_error\": " << result.max_abs_error
              << ", \"relative_l2_error\": " << result.relative_l2_error
              << ", \"finite\": " << (result.finite ? "true" : "false")
              << ", \"buffer_bytes\": " << result.buffer_bytes
              << ", \"error_check\": {\"kind\": \"" << (result.full_check ? "full" : "deterministic_sample")
              << "\", \"elements\": " << result.checked_elements << "}}";
    if (i + 1 != results.size()) std::cout << ',';
    std::cout << '\n';
  }
  std::cout << "  ]\n}\n";
}

void check_exact(const std::vector<half>& actual, const std::vector<float>& expected,
                 const std::string& label) {
  if (actual.size() != expected.size()) throw ExecutionError(label + " size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float value = __half2float(actual[i]);
    if (value != expected[i]) {
      std::ostringstream message;
      message << label << " mismatch at output " << i << ": got " << value
              << ", expected " << expected[i];
      throw ExecutionError(message.str());
    }
  }
}

void run_self_test() {
  // M=4, N=8, K=4 keeps the INT8 cuBLAS dimensions aligned while making W
  // and Y non-square. Expected arrays are independently hand-computed. The
  // first INT8 result deliberately covers symmetric saturation: 8 * 16 is
  // clamped from 128 to 127 and dequantizes to 7.9375.
  const Case spec{"self_test", 4, 8, 4, 1};
  const std::vector<float> x_first = {
      1, 2, 3, 4,  5, 6, 7, 8,  -1, 0, 1, 2,  2, -2, 4, -4};
  const std::vector<float> x_second = {
      0, 1, 0, 1,  1, 0, 1, 0,  -2, -1, 0, 1,  4, 3, 2, 1};
  const std::vector<float> weights = {
      1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1,
      1, 1, 1, 1,  1,-1, 0, 0,  0, 0, 1,-1,  1, 1,-1,-1};
  const std::vector<float> expected_first = {
      1,2,3,4,10,-1,-1,-4,  5,6,7,8,26,-1,-1,-4,
     -1,0,1,2, 2,-1,-1,-4,  2,-2,4,-4,0,4,8,0};
  const std::vector<float> expected_first_int8 = {
      1,2,3,4,10,-1,-1,-4,  5,6,7,7.9375f,25.9375f,-1,-0.9375f,-3.9375f,
     -1,0,1,2, 2,-1,-1,-4,  2,-2,4,-4,0,4,8,0};
  const std::vector<float> expected_second = {
      0,1,0,1,2,-1,-1,0,  1,0,1,0,2,1,1,0,
     -2,-1,0,1,-2,-1,-1,-4,  4,3,2,1,10,1,1,4};

  std::vector<half> h_x_first(x_first.size()), h_x_second(x_second.size()),
                    h_weights(weights.size());
  for (std::size_t i = 0; i < x_first.size(); ++i) {
    h_x_first[i] = __float2half(x_first[i]);
    h_x_second[i] = __float2half(x_second[i]);
  }
  for (std::size_t i = 0; i < weights.size(); ++i)
    h_weights[i] = __float2half(weights[i]);

  CudaStream stream;
  CublasHandle handle(stream);
  DeviceBuffer<half> x(h_x_first.size()), w(h_weights.size()), y(expected_first.size());
  DeviceBuffer<std::int8_t> x8(h_x_first.size()), w8(h_weights.size());
  DeviceBuffer<std::int32_t> y32(expected_first.size());
  CUDA_CHECK(cudaMemcpyAsync(w.get(), h_weights.data(), h_weights.size() * sizeof(half),
                             cudaMemcpyHostToDevice, stream));

  auto invoke = [&](Mode mode, const std::vector<half>& host_x,
                    const std::vector<float>& expected, const std::string& label) {
    CUDA_CHECK(cudaMemsetAsync(y.get(), 0x7f, y.size() * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(y32.get(), 0x7f, y32.size() * sizeof(std::int32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(x.get(), host_x.data(), host_x.size() * sizeof(half),
                               cudaMemcpyHostToDevice, stream));
    if (mode == Mode::kInt8) {
      quantize_int8<<<blocks_for(w.size()), kThreads, 0, stream>>>(w.get(), w8.get(), w.size());
      CUDA_CHECK(cudaGetLastError());
      launch_int8(handle, stream, spec, x.get(), x8.get(), w8.get(), y32.get(), y.get());
    } else {
      launch_fp16(handle, mode, spec, x.get(), w.get(), y.get());
    }
    std::vector<half> actual(expected.size());
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), y.get(), actual.size() * sizeof(half),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_exact(actual, expected, label);
  };

  for (Mode mode : {Mode::kFp16Acc32, Mode::kFp16Acc16, Mode::kInt8}) {
    const std::vector<float>& first_expected =
        mode == Mode::kInt8 ? expected_first_int8 : expected_first;
    invoke(mode, h_x_first, first_expected, std::string(mode_name(mode)) + " first invocation");
    invoke(mode, h_x_second, expected_second, std::string(mode_name(mode)) + " second invocation");
  }
  std::cout << "{\"schema_version\":1,\"benchmark\":\"native_gemm_replay_self_test\","
               "\"measurement_kind\":\"correctness_only\",\"passed\":true,"
               "\"modes\":[\"fp16_acc32\",\"fp16_acc16\",\"int8\"]}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.self_test) {
      run_self_test();
      return 0;
    }
    const std::vector<Case> cases = read_manifest(options.manifest);
    std::vector<Result> results;
    results.reserve(cases.size());
    for (const Case& spec : cases) {
      results.push_back(execute_case(spec, options.mode, options.runs, options.warmups,
                                     options.iterations, options.max_buffer_bytes));
    }
    print_results(options.mode, results);
    return 0;
  } catch (const InvalidInput& error) {
    std::cerr << "invalid input: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "execution failure: " << error.what() << '\n';
    return 1;
  }
}

