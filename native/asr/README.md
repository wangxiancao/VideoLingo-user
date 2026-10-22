# Native isolated GEMM replay

`glm_gemm_replay` measures selected GLM matrix shapes with CUDA and cuBLAS. It
is an isolated GEMM tool. Its output is not ASR RTFx and cannot establish the
540 RTFx acceptance target.

The manifest `repeats` field records how often a shape occurs in the planned
model workload. A timing run repeatedly uses the same X and W allocation. It
therefore does not measure the memory traffic of distinct weights for every
layer or task call. Cache reuse can especially overstate small-M decoder GEMM
performance. Decoder results must not be converted into a whole-model
throughput claim. Report the unoptimized real batch and token assumptions next
to any replay result, and continue to label every performance number as an
isolated GEMM measurement.

Each JSON case preserves the manifest's real batch/token-shaped `m`, `n`, `k`
and `repeats` values. Its `timing_scope` states that one GEMM is timed per
iteration and that `repeats` remains metadata. Scenario metadata from the shape
planner must accompany a result so the unoptimized batch and token assumptions
remain explicit.

## Build

CUDA, cuBLAS and a C++17 compiler are required. Build products are kept under
the ignored `native/asr/build/` directory:

```sh
cmake -S native/asr -B native/asr/build
cmake --build native/asr/build -j 4
```

The checked-in build targets the RTX 3090 (`sm_86`). On this development host,
CMake prefers `/usr/local/cuda-12.8/bin/nvcc` because `/usr/bin/nvcc` is an
obsolete CUDA 10.1 installation. It also selects `/usr/bin/g++` as the host
compiler so the executable uses the system-compatible C++ runtime.

The executable links CUDA runtime and cuBLAS only; it does not use Python,
PyTorch, libtorch, vLLM, SGLang or TensorRT.

## Manifest and command line

Input is UTF-8 TSV with an exact header:

```text
id\tm\tn\tk\trepeats
encoder_qkv_b8\t12000\t3840\t1280\t480
```

X is row-major `[m,k]`, W is row-major `[n,k]`, and output is row-major
`[m,n]`, computing `X * W^T`. IDs may contain ASCII letters, digits, dot,
underscore and hyphen. Dimensions and repeats must be positive integers and IDs
must be unique.

```sh
native/asr/build/glm_gemm_replay --self-test
native/asr/build/glm_gemm_replay \
  --manifest shapes.tsv --mode fp16_acc32 \
  --runs 3 --warmups 2 --iterations 5 --max-buffer-mib 4096
```

Modes are:

- `fp16_acc32`: FP16 operands and output with `CUBLAS_COMPUTE_32F` and float
  alpha/beta.
- `fp16_acc16`: FP16 operands and output with `CUBLAS_COMPUTE_16F` and half
  alpha/beta.
- `int8`: real INT8 operands and INT32 accumulation. Static W quantization is
  outside timing; activation quantization and output dequantization are inside
  every timed iteration. The deterministic replay uses a fixed scale of 16 and
  symmetric `[-127, 127]` clipping, then dequantizes the INT32 output by 256.

When supplied, `--max-buffer-mib` caps explicitly allocated X, W, output and
quantized working buffers before allocation. With no cap, allocation is limited
by checked host size arithmetic and available device memory. JSON is written to
stdout and diagnostics to stderr. Exit status 0 means success, 1 execution or
numeric failure, and 2 invalid CLI or manifest input.

`buffer_bytes` is the simultaneous explicit device allocation for a case. The
FP16 modes count FP16 X, W, and output. INT8 additionally counts INT8 X and W
plus the INT32 accumulator; the FP16 X/W buffers are the deterministic source
values used for quantization and the FP16 output holds the dequantized result.
CUDA context memory and any internal cuBLAS workspace are not included. Cases
execute sequentially, so their buffer counts are not added together.

## Timing and numeric checks

CUDA events surround GPU work. Each reported run is divided by `--iterations`;
the median is the ordinary middle value for odd counts and the mean of the two
middle values for even counts. Warmups and numeric checks are outside timing.

Inputs are generated deterministically on the device. Outputs with at most
4096 elements are checked in full against a deterministic CPU reference.
Larger outputs check 64 evenly spaced deterministic elements. FP16 modes use
FP32 accumulation as the reference; INT8 uses the exact quantized INT32 dot
product followed by the documented scale. Each JSON case
records the check kind and element count alongside finite, maximum absolute
error and relative L2 error fields. The self-test uses explicit, hand-computed
non-square matrices in all three modes. It poisons and reuses the same output
buffers for a second changed input, catching stale-output reuse.

The versioned JSON also includes `measurement_kind: "isolated_gemm"` and a
`workload_disclosure` explaining the repeated working-set limitation. Norms,
attention, positional operations, KV traffic, tokenization, frontend,
scheduling, model loading and full ASR execution remain outside this tool.
