#include "model_loader.h"
#include "engine.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

using InvalidInput = ModelInputError;
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

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), self-contained so the binary keeps its CUDA + cuBLAS
// only dependency profile.
// ---------------------------------------------------------------------------

class Sha256 {
 public:
  Sha256() { reset(); }
  void reset() {
    state_[0] = 0x6a09e667u; state_[1] = 0xbb67ae85u; state_[2] = 0x3c6ef372u;
    state_[3] = 0xa54ff53au; state_[4] = 0x510e527fu; state_[5] = 0x9b05688cu;
    state_[6] = 0x1f83d9abu; state_[7] = 0x5be0cd19u;
    length_ = 0;
    buffer_used_ = 0;
  }
  void update(const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    length_ += size;
    while (size > 0) {
      const size_t take = std::min(size, sizeof(buffer_) - buffer_used_);
      std::memcpy(buffer_ + buffer_used_, bytes, take);
      buffer_used_ += take;
      bytes += take;
      size -= take;
      if (buffer_used_ == sizeof(buffer_)) {
        transform(buffer_);
        buffer_used_ = 0;
      }
    }
  }
  std::string hexdigest() {
    Sha256 copy = *this;  // keep the stream usable after finalization
    copy.finalize();
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t word : copy.digest_) {
      for (int shift = 28; shift >= 0; shift -= 4)
        out.push_back(kHex[(word >> shift) & 0xF]);
    }
    return out;
  }

 private:
  static uint32_t rotr(uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32 - bits));
  }
  void transform(const uint8_t* block) {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(block[4 * i]) << 24) | (uint32_t(block[4 * i + 1]) << 16) |
             (uint32_t(block[4 * i + 2]) << 8) | uint32_t(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = h + s1 + ch + k[i] + w[i];
      const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
  }
  void finalize() {
    const uint64_t bit_length = length_ * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t zero = 0;
    while (buffer_used_ != 56) update(&zero, 1);
    // The length is appended without going through update() so length_ stays
    // at its padded value; finalize() only runs on private copies.
    uint8_t length_bytes[8];
    for (int i = 0; i < 8; ++i) length_bytes[i] = uint8_t(bit_length >> (56 - 8 * i));
    std::memcpy(buffer_ + buffer_used_, length_bytes, 8);
    transform(buffer_);
    buffer_used_ = 0;
    for (int i = 0; i < 8; ++i) digest_[i] = state_[i];
  }

  uint32_t state_[8];
  uint32_t digest_[8];
  uint64_t length_;
  uint8_t buffer_[64];
  size_t buffer_used_;
};

// ---------------------------------------------------------------------------
// Minimal JSON parser for the safetensors header and config.json: objects,
// arrays, strings, numbers, booleans, null. Rejects trailing data.
// ---------------------------------------------------------------------------

struct JsonValue;
using JsonObject = std::vector<std::pair<std::string, JsonValue>>;

struct JsonValue {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
  Type type = Type::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string str;
  std::vector<JsonValue> array;
  std::shared_ptr<JsonObject> object;

  const JsonValue* find(const std::string& key) const {
    if (type != Type::kObject || !object) return nullptr;
    for (const auto& entry : *object)
      if (entry.first == key) return &entry.second;
    return nullptr;
  }
  int64_t as_int(const char* what) const {
    if (type != Type::kNumber)
      throw InvalidInput(std::string("expected number for ") + what);
    return static_cast<int64_t>(number);
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  JsonValue parse() {
    skip_ws();
    JsonValue value = parse_value(0);
    skip_ws();
    if (pos_ != text_.size()) throw InvalidInput("trailing JSON data");
    return value;
  }

 private:
  static constexpr int kMaxDepth = 32;

  void skip_ws() {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' ||
                                   text_[pos_] == '\n' || text_[pos_] == '\r'))
      ++pos_;
  }
  char peek() const {
    if (pos_ >= text_.size()) throw InvalidInput("unexpected end of JSON");
    return text_[pos_];
  }
  void expect(char c) {
    if (peek() != c)
      throw InvalidInput(std::string("expected '") + c + "' in JSON at offset " +
                         std::to_string(pos_));
    ++pos_;
  }

  JsonValue parse_value(int depth) {
    if (depth > kMaxDepth) throw InvalidInput("JSON nesting too deep");
    skip_ws();
    const char c = peek();
    if (c == '{') return parse_object(depth);
    if (c == '[') return parse_array(depth);
    if (c == '"') {
      JsonValue v;
      v.type = JsonValue::Type::kString;
      v.str = parse_string();
      return v;
    }
    if (c == 't' || c == 'f') return parse_bool();
    if (c == 'n') return parse_null();
    return parse_number();
  }

  JsonValue parse_object(int depth) {
    expect('{');
    JsonValue v;
    v.type = JsonValue::Type::kObject;
    v.object = std::make_shared<JsonObject>();
    skip_ws();
    if (peek() == '}') {
      ++pos_;
      return v;
    }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      v.object->emplace_back(std::move(key), parse_value(depth + 1));
      skip_ws();
      const char c = peek();
      if (c == ',') {
        ++pos_;
        continue;
      }
      if (c == '}') {
        ++pos_;
        return v;
      }
      throw InvalidInput("expected ',' or '}' in JSON object");
    }
  }

  JsonValue parse_array(int depth) {
    expect('[');
    JsonValue v;
    v.type = JsonValue::Type::kArray;
    skip_ws();
    if (peek() == ']') {
      ++pos_;
      return v;
    }
    while (true) {
      v.array.push_back(parse_value(depth + 1));
      skip_ws();
      const char c = peek();
      if (c == ',') {
        ++pos_;
        continue;
      }
      if (c == ']') {
        ++pos_;
        return v;
      }
      throw InvalidInput("expected ',' or ']' in JSON array");
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) throw InvalidInput("unterminated JSON string");
      const char c = text_[pos_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (pos_ >= text_.size()) throw InvalidInput("unterminated JSON escape");
        const char e = text_[pos_++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            if (pos_ + 4 > text_.size()) throw InvalidInput("short \\u escape");
            unsigned code = 0;
            for (int i = 0; i < 4; ++i) {
              const char h = text_[pos_++];
              code <<= 4;
              if (h >= '0' && h <= '9') code |= unsigned(h - '0');
              else if (h >= 'a' && h <= 'f') code |= unsigned(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') code |= unsigned(h - 'A' + 10);
              else throw InvalidInput("bad \\u escape");
            }
            if (code < 0x80) {
              out.push_back(char(code));
            } else if (code < 0x800) {
              out.push_back(char(0xC0 | (code >> 6)));
              out.push_back(char(0x80 | (code & 0x3F)));
            } else {
              out.push_back(char(0xE0 | (code >> 12)));
              out.push_back(char(0x80 | ((code >> 6) & 0x3F)));
              out.push_back(char(0x80 | (code & 0x3F)));
            }
            break;
          }
          default: throw InvalidInput("unknown JSON escape");
        }
      } else {
        out.push_back(c);
      }
    }
  }

  JsonValue parse_bool() {
    if (text_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      JsonValue v; v.type = JsonValue::Type::kBool; v.boolean = true;
      return v;
    }
    if (text_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      JsonValue v; v.type = JsonValue::Type::kBool; v.boolean = false;
      return v;
    }
    throw InvalidInput("invalid JSON literal");
  }

  JsonValue parse_null() {
    if (text_.compare(pos_, 4, "null") == 0) {
      pos_ += 4;
      return JsonValue();
    }
    throw InvalidInput("invalid JSON literal");
  }

  JsonValue parse_number() {
    const size_t start = pos_;
    if (peek() == '-') ++pos_;
    while (pos_ < text_.size() && ((text_[pos_] >= '0' && text_[pos_] <= '9') ||
                                   text_[pos_] == '.' || text_[pos_] == 'e' ||
                                   text_[pos_] == 'E' || text_[pos_] == '+' ||
                                   text_[pos_] == '-'))
      ++pos_;
    if (pos_ == start) throw InvalidInput("invalid JSON number");
    JsonValue v;
    v.type = JsonValue::Type::kNumber;
    v.number = std::stod(text_.substr(start, pos_ - start));
    return v;
  }

  const std::string& text_;
  size_t pos_ = 0;
};

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw InvalidInput("cannot open file: " + path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::string sha256_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw InvalidInput("cannot open file: " + path);
  Sha256 hash;
  std::vector<char> chunk(1 << 20);
  while (in) {
    in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    const std::streamsize got = in.gcount();
    if (got > 0) hash.update(chunk.data(), static_cast<size_t>(got));
    if (!in) break;
  }
  return hash.hexdigest();
}

// ---------------------------------------------------------------------------
// BF16 -> FP16 with round-to-nearest-even. NaN maps to the canonical quiet
// half NaN 0x7E00 so CPU and GPU conversions are bitwise identical.
// ---------------------------------------------------------------------------

uint16_t fp32_to_fp16_rne(uint32_t bits) {
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t exponent = (bits >> 23) & 0xFFu;
  const uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent == 0xFFu)
    return static_cast<uint16_t>(sign | (mantissa ? 0x7E00u : 0x7C00u));
  const int32_t e = static_cast<int32_t>(exponent) - 127 + 15;
  if (e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> +/-Inf
  if (e <= 0) {
    if (e < -10) return static_cast<uint16_t>(sign);  // rounds to zero
    const uint32_t m = mantissa | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - e);
    uint32_t half = m >> shift;
    const uint32_t remainder = m & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = (static_cast<uint32_t>(e) << 10) | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half;
  return static_cast<uint16_t>(sign | half);
}

uint16_t bf16_to_fp16_host(uint16_t bf16_bits) {
  if (((bf16_bits >> 7) & 0xFFu) == 0xFFu && (bf16_bits & 0x7Fu) != 0)
    return 0x7E00u;  // canonical quiet NaN
  return fp32_to_fp16_rne(static_cast<uint32_t>(bf16_bits) << 16);
}

float bf16_bits_to_float(uint16_t bits) {
  const uint32_t f = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

bool fp16_is_inf(uint16_t bits) {
  return (bits & 0x7FFFu) == 0x7C00u;
}

// Device-side BF16 -> FP16, bitwise-matching the host path: NaN inputs map to
// 0x7E00, everything else goes through __float2half_rn (round-to-nearest-even).
__global__ void convert_bf16_to_fp16_kernel(const uint16_t* __restrict__ in,
                                            uint16_t* __restrict__ out,
                                            uint64_t count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const uint16_t bf16_bits = in[index];
  if (((bf16_bits >> 7) & 0xFFu) == 0xFFu && (bf16_bits & 0x7Fu) != 0) {
    out[index] = 0x7E00u;
    return;
  }
  const uint32_t f = static_cast<uint32_t>(bf16_bits) << 16;
  const float value = __uint_as_float(f);
  const __half half = __float2half_rn(value);
  out[index] = __half_as_ushort(half);
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

const JsonValue& require(const JsonValue* parent, const char* key, const char* what) {
  const JsonValue* found = parent->find(key);
  if (!found)
    throw InvalidInput(std::string("missing key '") + key + "' in " + what);
  return *found;
}

int require_int(const JsonValue* parent, const char* key, const char* what) {
  const JsonValue& v = require(parent, key, what);
  const int64_t value = v.as_int((std::string(what) + "." + key).c_str());
  if (value <= 0 || value > 100000)
    throw InvalidInput(std::string("out-of-range integer for ") + what + "." + key);
  return static_cast<int>(value);
}

// Expected tensor inventory of the official GlmAsrForConditionalGeneration
// checkpoint: a Whisper-style audio tower (LayerNorm with bias; attention
// biases on q/v/o but not k), a Llama-style text decoder without biases, and
// a two-layer GELU projector with biases. The projector mid width is not part
// of config.json and is validated for internal consistency only.
struct ExpectedTensor {
  std::string name;
  std::vector<int64_t> shape;
};

std::vector<ExpectedTensor> expected_tensors(const AsrModelConfig& cfg) {
  std::vector<ExpectedTensor> out;
  const int ah = cfg.audio_hidden, ai = cfg.audio_intermediate;
  const int mel = cfg.audio_mel_bins;
  const int th = cfg.text_hidden, ti = cfg.text_intermediate;
  const int kv = cfg.text_kv_heads * cfg.text_head_dim;
  const int vs = cfg.vocab_size;

  auto add = [&out](std::string name, std::vector<int64_t> shape) {
    out.push_back(ExpectedTensor{std::move(name), std::move(shape)});
  };

  add("audio_tower.conv1.weight", {ah, mel, 3});
  add("audio_tower.conv1.bias", {ah});
  add("audio_tower.conv2.weight", {ah, ah, 3});
  add("audio_tower.conv2.bias", {ah});
  for (int layer = 0; layer < cfg.audio_layers; ++layer) {
    const std::string p = "audio_tower.layers." + std::to_string(layer) + ".";
    add(p + "input_layernorm.weight", {ah});
    add(p + "input_layernorm.bias", {ah});
    add(p + "self_attn.q_proj.weight", {ah, ah});
    add(p + "self_attn.q_proj.bias", {ah});
    add(p + "self_attn.k_proj.weight", {ah, ah});
    add(p + "self_attn.v_proj.weight", {ah, ah});
    add(p + "self_attn.v_proj.bias", {ah});
    add(p + "self_attn.o_proj.weight", {ah, ah});
    add(p + "self_attn.o_proj.bias", {ah});
    add(p + "post_attention_layernorm.weight", {ah});
    add(p + "post_attention_layernorm.bias", {ah});
    add(p + "mlp.fc1.weight", {ai, ah});
    add(p + "mlp.fc1.bias", {ai});
    add(p + "mlp.fc2.weight", {ah, ai});
    add(p + "mlp.fc2.bias", {ah});
  }
  add("audio_tower.norm.weight", {ah});
  add("audio_tower.norm.bias", {ah});

  // The projector consumes frame-merged encoder output: width ah * 4.
  add("multi_modal_projector.linear_1.weight", {cfg.projector_mid, ah * 4});
  add("multi_modal_projector.linear_1.bias", {cfg.projector_mid});
  add("multi_modal_projector.linear_2.weight", {th, cfg.projector_mid});
  add("multi_modal_projector.linear_2.bias", {th});

  add("language_model.model.embed_tokens.weight", {vs, th});
  for (int layer = 0; layer < cfg.text_layers; ++layer) {
    const std::string p = "language_model.model.layers." + std::to_string(layer) + ".";
    add(p + "input_layernorm.weight", {th});
    add(p + "self_attn.q_proj.weight", {th, th});
    add(p + "self_attn.k_proj.weight", {kv, th});
    add(p + "self_attn.v_proj.weight", {kv, th});
    add(p + "self_attn.o_proj.weight", {th, th});
    add(p + "post_attention_layernorm.weight", {th});
    add(p + "mlp.gate_proj.weight", {ti, th});
    add(p + "mlp.up_proj.weight", {ti, th});
    add(p + "mlp.down_proj.weight", {th, ti});
  }
  add("language_model.model.norm.weight", {th});
  add("language_model.lm_head.weight", {vs, th});
  return out;
}

class MappedCheckpoint {
 public:
  MappedCheckpoint(const std::string& path, uint64_t data_region_offset)
      : file_(std::fopen(path.c_str(), "rb")) {
    if (!file_) throw InvalidInput("cannot open safetensors file: " + path);
    std::fseek(file_, 0, SEEK_END);
    const long size = std::ftell(file_);
    if (size < 0) throw ExecutionError("ftell failed on safetensors file");
    data_region_offset_ = data_region_offset;
    data_size_ = static_cast<uint64_t>(size) - data_region_offset;
  }
  ~MappedCheckpoint() {
    if (file_) std::fclose(file_);
  }
  MappedCheckpoint(const MappedCheckpoint&) = delete;
  MappedCheckpoint& operator=(const MappedCheckpoint&) = delete;

  std::vector<uint8_t> read(uint64_t begin, uint64_t nbytes) const {
    if (begin > data_size_ || nbytes > data_size_ - begin)
      throw ExecutionError("tensor read out of bounds");
    std::vector<uint8_t> buffer(nbytes);
    if (nbytes == 0) return buffer;
    if (std::fseek(file_, static_cast<long>(data_region_offset_ + begin), SEEK_SET) != 0)
      throw ExecutionError("fseek failed in safetensors file");
    if (std::fread(buffer.data(), 1, nbytes, file_) != nbytes)
      throw ExecutionError("short read from safetensors file");
    return buffer;
  }

 private:
  std::FILE* file_ = nullptr;
  uint64_t data_region_offset_ = 0;
  uint64_t data_size_ = 0;
};

void accumulate_stats(const std::vector<uint8_t>& bf16_bytes,
                      const std::vector<uint8_t>& fp16_bytes, ConvertedTensor* out) {
  const size_t count = bf16_bytes.size() / 2;
  const uint16_t* in = reinterpret_cast<const uint16_t*>(bf16_bytes.data());
  const uint16_t* conv = reinterpret_cast<const uint16_t*>(fp16_bytes.data());
  for (size_t i = 0; i < count; ++i) {
    const float value = bf16_bits_to_float(in[i]);
    if (std::isnan(value)) {
      ++out->nan_count;
    } else {
      if (fp16_is_inf(conv[i])) ++out->overflow_count;
      const double magnitude = std::fabs(static_cast<double>(value));
      if (magnitude > out->max_abs_value) out->max_abs_value = magnitude;
    }
  }
}

void merge_into_report(const ConvertedTensor& converted, LoadReport* report) {
  report->tensors.push_back(converted);
  report->overflow_count += converted.overflow_count;
  report->nan_count += converted.nan_count;
  report->max_abs_value = std::max(report->max_abs_value, converted.max_abs_value);
}

void collect_inventory(const std::string& model_dir, LoadReport* report,
                       std::vector<TensorSpec>* tensors, uint64_t* data_offset) {
  const AsrModelConfig cfg = parse_model_config(model_dir);
  *tensors = read_safetensors_header(model_dir + "/model.safetensors", data_offset);
  validate_inventory(*tensors, cfg, report);
  if (!report->missing_tensors.empty() || !report->unexpected_tensors.empty() ||
      !report->shape_mismatches.empty()) {
    throw InvalidInput(
        "checkpoint inventory does not match the official GLM-ASR layout "
        "(missing/unexpected/shape mismatches would be reported in JSON)");
  }
  report->tensor_count = tensors->size();
  for (const TensorSpec& t : *tensors) {
    report->total_params += t.element_count();
    report->fp16_bytes += t.nbytes();
  }
}

}  // namespace

uint64_t TensorSpec::element_count() const {
  uint64_t count = 1;
  for (int64_t dim : shape) count *= static_cast<uint64_t>(dim);
  return count;
}

AsrModelConfig parse_model_config(const std::string& model_dir) {
  const std::string text = read_file(model_dir + "/config.json");
  const JsonValue root = JsonParser(text).parse();
  if (root.type != JsonValue::Type::kObject) throw InvalidInput("config.json: expected object");

  AsrModelConfig cfg;
  const JsonValue& audio = require(&root, "audio_config", "config.json");
  cfg.audio_hidden = require_int(&audio, "hidden_size", "audio_config");
  cfg.audio_intermediate = require_int(&audio, "intermediate_size", "audio_config");
  cfg.audio_layers = require_int(&audio, "num_hidden_layers", "audio_config");
  cfg.audio_heads = require_int(&audio, "num_attention_heads", "audio_config");
  cfg.audio_head_dim = require_int(&audio, "head_dim", "audio_config");
  cfg.audio_mel_bins = require_int(&audio, "num_mel_bins", "audio_config");

  const JsonValue& text_cfg = require(&root, "text_config", "config.json");
  cfg.text_hidden = require_int(&text_cfg, "hidden_size", "text_config");
  cfg.text_intermediate = require_int(&text_cfg, "intermediate_size", "text_config");
  cfg.text_layers = require_int(&text_cfg, "num_hidden_layers", "text_config");
  cfg.text_heads = require_int(&text_cfg, "num_attention_heads", "text_config");
  cfg.text_kv_heads = require_int(&text_cfg, "num_key_value_heads", "text_config");
  cfg.text_head_dim = require_int(&text_cfg, "head_dim", "text_config");
  cfg.vocab_size = require_int(&text_cfg, "vocab_size", "text_config");

  if (cfg.audio_heads * cfg.audio_head_dim != cfg.audio_hidden)
    throw InvalidInput("audio_config: heads * head_dim != hidden_size");
  if (cfg.text_heads * cfg.text_head_dim != cfg.text_hidden)
    throw InvalidInput("text_config: heads * head_dim != hidden_size");
  return cfg;
}

std::vector<TensorSpec> read_safetensors_header(const std::string& path,
                                                uint64_t* data_region_offset) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw InvalidInput("cannot open safetensors file: " + path);
  uint8_t size_bytes[8];
  if (!in.read(reinterpret_cast<char*>(size_bytes), 8))
    throw InvalidInput("safetensors file too short: " + path);
  uint64_t header_size = 0;
  for (int i = 0; i < 8; ++i) header_size |= uint64_t(size_bytes[i]) << (8 * i);
  constexpr uint64_t kMaxHeader = 64u << 20;
  if (header_size == 0 || header_size > kMaxHeader)
    throw InvalidInput("implausible safetensors header size: " + std::to_string(header_size));
  std::string header(header_size, '\0');
  if (!in.read(&header[0], static_cast<std::streamsize>(header_size)))
    throw InvalidInput("failed to read safetensors header");
  *data_region_offset = 8 + header_size;

  const JsonValue root = JsonParser(header).parse();
  if (root.type != JsonValue::Type::kObject)
    throw InvalidInput("safetensors header: expected object");

  in.seekg(0, std::ios::end);
  const uint64_t file_size = static_cast<uint64_t>(in.tellg());
  const uint64_t data_size = file_size - *data_region_offset;

  std::vector<TensorSpec> tensors;
  for (const auto& entry : *root.object) {
    if (entry.first == "__metadata__") continue;
    const JsonValue& spec = entry.second;
    if (spec.type != JsonValue::Type::kObject)
      throw InvalidInput("safetensors header: tensor entry is not an object: " + entry.first);
    TensorSpec tensor;
    tensor.name = entry.first;
    const JsonValue& dtype = require(&spec, "dtype", tensor.name.c_str());
    if (dtype.type != JsonValue::Type::kString) throw InvalidInput("bad dtype for " + tensor.name);
    tensor.dtype = dtype.str;
    const JsonValue& shape = require(&spec, "shape", tensor.name.c_str());
    if (shape.type != JsonValue::Type::kArray) throw InvalidInput("bad shape for " + tensor.name);
    for (const JsonValue& dim : shape.array) {
      const int64_t value = dim.as_int("shape");
      if (value <= 0 || value > (1 << 30)) throw InvalidInput("bad shape dim for " + tensor.name);
      tensor.shape.push_back(value);
    }
    const JsonValue& offsets = require(&spec, "data_offsets", tensor.name.c_str());
    if (offsets.type != JsonValue::Type::kArray || offsets.array.size() != 2)
      throw InvalidInput("bad data_offsets for " + tensor.name);
    tensor.begin = static_cast<uint64_t>(offsets.array[0].as_int("data_offsets"));
    tensor.end = static_cast<uint64_t>(offsets.array[1].as_int("data_offsets"));
    if (tensor.begin > tensor.end || tensor.end > data_size)
      throw InvalidInput("data_offsets out of range for " + tensor.name);
    if (tensor.dtype != "BF16")
      throw InvalidInput("unsupported dtype " + tensor.dtype + " for " + tensor.name);
    if (tensor.nbytes() != tensor.element_count() * 2)
      throw InvalidInput("byte length mismatch for " + tensor.name);
    tensors.push_back(std::move(tensor));
  }
  std::sort(tensors.begin(), tensors.end(),
            [](const TensorSpec& a, const TensorSpec& b) { return a.name < b.name; });
  if (tensors.empty()) throw InvalidInput("safetensors file contains no tensors");
  return tensors;
}

void validate_inventory(const std::vector<TensorSpec>& tensors, const AsrModelConfig& cfg,
                        LoadReport* report) {
  const std::vector<ExpectedTensor> expected = expected_tensors(cfg);
  std::map<std::string, const TensorSpec*> actual;
  for (const TensorSpec& t : tensors) actual[t.name] = &t;

  for (const ExpectedTensor& want : expected) {
    const auto it = actual.find(want.name);
    if (it == actual.end()) {
      report->missing_tensors.push_back(want.name);
      continue;
    }
    if (it->second->shape != want.shape) {
      std::ostringstream detail;
      detail << want.name << ": expected [";
      for (size_t i = 0; i < want.shape.size(); ++i)
        detail << (i ? ", " : "") << want.shape[i];
      detail << "] got [";
      for (size_t i = 0; i < it->second->shape.size(); ++i)
        detail << (i ? ", " : "") << it->second->shape[i];
      detail << "]";
      report->shape_mismatches.push_back(detail.str());
    }
  }
  std::map<std::string, bool> expected_names;
  for (const ExpectedTensor& want : expected) expected_names[want.name] = true;
  for (const TensorSpec& t : tensors)
    if (!expected_names.count(t.name)) report->unexpected_tensors.push_back(t.name);
  std::sort(report->missing_tensors.begin(), report->missing_tensors.end());
  std::sort(report->unexpected_tensors.begin(), report->unexpected_tensors.end());
  std::sort(report->shape_mismatches.begin(), report->shape_mismatches.end());
}

LoadReport dump_plan_pass(const std::string& model_dir, bool hash_tensors) {
  LoadReport report;
  report.model_dir = model_dir;
  report.config_sha256 = sha256_file(model_dir + "/config.json");
  std::vector<TensorSpec> tensors;
  uint64_t data_offset = 0;
  collect_inventory(model_dir, &report, &tensors, &data_offset);
  if (!hash_tensors) return report;

  MappedCheckpoint checkpoint(model_dir + "/model.safetensors", data_offset);
  Sha256 hash;
  std::vector<uint8_t> fp16;
  for (const TensorSpec& t : tensors) {
    const std::vector<uint8_t> bf16 = checkpoint.read(t.begin, t.nbytes());
    fp16.resize(bf16.size());
    const size_t count = bf16.size() / 2;
    const uint16_t* in = reinterpret_cast<const uint16_t*>(bf16.data());
    uint16_t* out16 = reinterpret_cast<uint16_t*>(fp16.data());
    for (size_t i = 0; i < count; ++i) out16[i] = bf16_to_fp16_host(in[i]);
    hash.reset();
    hash.update(fp16.data(), fp16.size());

    ConvertedTensor converted;
    converted.name = t.name;
    converted.shape = t.shape;
    converted.sha256_fp16 = hash.hexdigest();
    converted.element_count = count;
    accumulate_stats(bf16, fp16, &converted);
    merge_into_report(converted, &report);
  }
  return report;
}

LoadReport device_load_pass(const std::string& model_dir, uint64_t max_vram_bytes) {
  LoadReport report;
  report.model_dir = model_dir;
  report.vram_budget_bytes = max_vram_bytes;
  report.config_sha256 = sha256_file(model_dir + "/config.json");
  std::vector<TensorSpec> tensors;
  uint64_t data_offset = 0;
  collect_inventory(model_dir, &report, &tensors, &data_offset);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  report.device_name = properties.name;
  std::ostringstream cc;
  cc << properties.major << "." << properties.minor;
  report.compute_capability = cc.str();
  report.device_total_memory = static_cast<uint64_t>(properties.totalGlobalMem);

  DeviceWeights weights = load_weights_resident(model_dir, max_vram_bytes);
  MappedCheckpoint checkpoint(model_dir + "/model.safetensors", data_offset);
  Sha256 hash;
  for (const TensorSpec& t : tensors) {
    const DeviceWeights::Entry* entry = weights.find(t.name);
    if (!entry) throw ExecutionError("resident weights missing tensor: " + t.name);
    std::vector<uint8_t> fp16_host(t.nbytes());
    CUDA_CHECK(cudaMemcpy(fp16_host.data(), entry->ptr, t.nbytes(), cudaMemcpyDeviceToHost));
    hash.reset();
    hash.update(fp16_host.data(), fp16_host.size());

    const std::vector<uint8_t> bf16 = checkpoint.read(t.begin, t.nbytes());
    ConvertedTensor converted;
    converted.name = t.name;
    converted.shape = t.shape;
    converted.sha256_fp16 = hash.hexdigest();
    converted.element_count = t.element_count();
    accumulate_stats(bf16, fp16_host, &converted);
    merge_into_report(converted, &report);
  }
  report.weights_device_bytes = weights.device_bytes();
  uint64_t largest_bf16 = 0;
  for (const TensorSpec& t : tensors) largest_bf16 = std::max(largest_bf16, t.nbytes());
  report.peak_device_bytes = report.fp16_bytes + largest_bf16;
  return report;
}

DeviceWeights load_weights_resident(const std::string& model_dir, uint64_t max_vram_bytes) {
  LoadReport inventory;
  inventory.model_dir = model_dir;
  inventory.config_sha256 = sha256_file(model_dir + "/config.json");
  std::vector<TensorSpec> tensors;
  uint64_t data_offset = 0;
  collect_inventory(model_dir, &inventory, &tensors, &data_offset);

  uint64_t largest_bf16 = 0;
  for (const TensorSpec& t : tensors) largest_bf16 = std::max(largest_bf16, t.nbytes());
  const uint64_t peak_estimate = inventory.fp16_bytes + largest_bf16;
  if (peak_estimate > max_vram_bytes) {
    std::ostringstream detail;
    detail << "estimated device peak " << peak_estimate << " bytes exceeds VRAM budget "
           << max_vram_bytes << " bytes";
    throw ExecutionError(detail.str());
  }

  MappedCheckpoint checkpoint(model_dir + "/model.safetensors", data_offset);
  DeviceWeights weights;
  uint16_t* arena = nullptr;
  uint16_t* staging = nullptr;
  try {
    CUDA_CHECK(cudaMalloc(&arena, inventory.fp16_bytes));
    CUDA_CHECK(cudaMalloc(&staging, largest_bf16));
    uint64_t cursor = 0;
    for (const TensorSpec& t : tensors) {
      const std::vector<uint8_t> bf16 = checkpoint.read(t.begin, t.nbytes());
      const uint64_t count = t.element_count();
      CUDA_CHECK(cudaMemcpy(staging, bf16.data(), bf16.size(), cudaMemcpyHostToDevice));
      const unsigned threads = 256;
      const uint64_t blocks = (count + threads - 1) / threads;
      convert_bf16_to_fp16_kernel<<<static_cast<unsigned>(blocks), threads>>>(
          staging, arena + cursor, count);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      DeviceWeights::Entry entry;
      entry.ptr = arena + cursor;
      entry.shape = t.shape;
      entry.elements = count;
      weights.entries_.emplace_back(t.name, std::move(entry));
      cursor += count;
    }
    weights.arena_ = arena;
    weights.bytes_ = inventory.fp16_bytes;
    cudaFree(staging);
  } catch (...) {
    if (staging) cudaFree(staging);
    if (arena) cudaFree(arena);
    throw;
  }
  return weights;
}
