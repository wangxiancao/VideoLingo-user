# Native ASR primitive replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a native CUDA executable and a deterministic GLM shape planner
that compare real-shape FP16 accumulation and W8A8 primitives with reproducible,
explicitly non-ASR performance evidence.

**Architecture:** The Python planner reads only the official model config and
writes a TSV matrix manifest. A C++/CUDA executable consumes that manifest,
executes FP16/FP32-accumulation, FP16/FP16-accumulation or W8A8 GEMM, and emits
versioned JSON. A Python summarizer validates every result and estimates only
the measured linear workload, with omitted operations explicitly listed.

**Tech Stack:** Python 3.10 from /root/VideoLingo/.venv/bin/python, C++17 or
newer, CUDA 12.8, cuBLAS, existing pytest; no downloaded code or new packages.

## Global Constraints

- Spec: docs/superpowers/specs/2026-10-22-native-asr-540-design.md.
- One RTX 3090 with 24 GB VRAM.
- ASR-only hot RTFx >= 540: one hour of audio in at most 6.666667 seconds.
- No vLLM, SGLang, TensorRT, PyTorch or libtorch in native execution.
- No new dependencies; no pipeline/config/output/history changes.
- Replay is not a complete ASR implementation and cannot pass the 540 gate.
- Do not stop unrelated GPU services. Shared-GPU measurements are diagnostic.
- The implementers do not run GPU experiments concurrently. The leader owns
  benchmark scheduling; native self-test may run when explicitly scheduled.

---

## Task 1: Native GEMM replay executable

**Files:** Create native/asr/CMakeLists.txt, native/asr/src/gemm_replay.cu,
native/asr/README.md. Build products stay in native/asr/build/.

**Consumes:** UTF-8 TSV with header `id\tm\tn\tk\trepeats`; X is row-major
[m,k], W is row-major [n,k], output is row-major [m,n]. Positive integer
dimensions, repetitions and unambiguous identifiers only.

**Produces:** `native/asr/build/glm_gemm_replay` with:

```sh
glm_gemm_replay --self-test
glm_gemm_replay --manifest shapes.tsv --mode fp16_acc32 --runs 3 --warmups 2 --iterations 5
```

Modes: fp16_acc32, fp16_acc16, int8. Optional `--max-buffer-mib` caps the
explicitly allocated working buffers. JSON goes to stdout; errors to stderr;
exit 0 success, 1 execution/numerical failure, 2 invalid arguments/manifest.

Result contract:

```json
{
  "schema_version": 1,
  "benchmark": "native_gemm_replay",
  "mode": "fp16_acc32",
  "device": {"name": "RTX 3090", "compute_capability": "8.6", "total_memory_bytes": 25769803776},
  "cases": [{"id": "encoder_qkv", "m": 12000, "n": 3840, "k": 1280,
    "repeats": 480, "times_ms": [1.0, 1.1, 1.2], "median_ms": 1.1,
    "max_abs_error": 0.0, "relative_l2_error": 0.0,
    "finite": true, "buffer_bytes": 1234}]
}
```

Numeric values above illustrate types, not performance expectations. Quantize
activations and dequantize INT32 outputs inside each timed W8A8 iteration;
static weight quantization can precede timing. FP16-acc16 must select
CUBLAS_COMPUTE_16F, with half alpha/beta; FP16-acc32 uses
CUBLAS_COMPUTE_32F, with float alpha/beta; integer GEMM uses actual INT8 operands
and INT32 accumulation. Norms/attention are outside this replay's scope.

- [x] Write a `--self-test` using non-square matrices and a hand-computed
  expected output; include a second invocation to catch stale output buffers.
- [x] Add negative CLI/manifest checks for zero dimensions, duplicate IDs,
  unsupported mode, malformed columns, allocation-size overflow and memory cap.
- [x] Implement deterministic device-side inputs, RAII allocations/events and
  stream-aware cuBLAS calls; check all CUDA and cuBLAS statuses.
- [x] Use CUDA events around all timed GPU work and standard median for odd
  and even run counts. Warmups and output/reference checks are outside timing.
- [x] Verify all output elements on a small deterministic correctness case.
  For large matrix numerical summaries use a documented deterministic sample
  against FP32-accumulation reference; label sampling in JSON/README.
- [x] Build with `cmake -S native/asr -B native/asr/build` and
  `cmake --build native/asr/build -j 4`; report unavailable CUDA separately.
- [x] On the leader's schedule run `glm_gemm_replay --self-test`, then inspect
  `ldd` to confirm no forbidden inference runtime dependency.

## Task 2: Shape planner, result contract and focused tests

**Files:** Create bench/native_asr_replay.py and tests/test_native_asr_replay.py.

**Interfaces:**

```python
def build_cases(config: dict, chunks: int = 120, encoder_batch: int = 8,
                prefill_batch: int = 32, decode_batch: int = 120,
                output_tokens: int = 120, prompt_tokens: int = 20) -> list[dict]: ...
def write_manifest(cases: list[dict], path) -> None: ...
def summarize(cases: list[dict], result: dict) -> dict: ...
```

The 120/20 token values are explicit scenario assumptions, never observations.
Support partial final batches and require decode_batch to cover all chunks or
account for each decode group. Use model-config layer counts and GQA K/V
dimensions; do not count dense K/V projections for a grouped-query decoder.

For each encoder batch: M=batch*1500, K=1280; fused QKV N=3840, O N=1280,
MLP up N=5120 and down K=5120/N=1280. Repeat each across 32 layers and every
batch. Projector after frame merge uses tokens=375, 5120->4096->2048.
Decoder prefill uses 375+prompt_tokens, 28 layers, hidden=2048,
QKV N=(16+2*4)*128=3072, O=2048, fused gate/up N=12288 and down K=6144.
The first generated token comes from a prefill lm_head on the last prompt row
of each sequence. Decode uses one token per active sequence, 28 layers and
output_tokens-1 repeats; include the 2048->59264 lm_head at prefill and each
decode step, and clearly label constant-active-length
assumption. Shapes are validated against official config values, not silently
hardcoded for an unrelated checkpoint.

CLI:

```sh
/root/VideoLingo/.venv/bin/python -m bench.native_asr_replay plan --model-config /root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512/config.json --manifest /tmp/glm-shapes.tsv --metadata /tmp/glm-shapes.json
/root/VideoLingo/.venv/bin/python -m bench.native_asr_replay summarize --metadata /tmp/glm-shapes.json --result /tmp/glm-result.json --output /tmp/glm-summary.json
```

Summaries require exact case IDs/dimensions/repeats, supported schema and
benchmark name, matching finite run measurements, true finite flags and
nonnegative error norms. Reject missing, duplicate or additional cases and
nonfinite/negative timings. Include `metric_kind=isolated_gemm_estimate`,
`asr_acceptance=false`, weighted estimated linear milliseconds and omissions:
mel, convolutions, attention, norms, activations, residuals, KV, routing,
tokenization, scheduling and model loading. Do not print ASR RTFx.

- [x] Write contract tests before implementation. Required cases:

```python
def glm_config():
    return {
        'model_type': 'glmasr',
        'audio_config': {'hidden_size': 1280, 'intermediate_size': 5120,
                         'num_hidden_layers': 32, 'num_attention_heads': 20,
                         'num_key_value_heads': 20, 'max_position_embeddings': 1500},
        'text_config': {'hidden_size': 2048, 'intermediate_size': 6144,
                        'num_hidden_layers': 28, 'num_attention_heads': 16,
                        'num_key_value_heads': 4, 'head_dim': 128,
                        'vocab_size': 59264},
    }

def valid_native_result(cases):
    return {'schema_version': 1, 'benchmark': 'native_gemm_replay',
            'mode': 'fp16_acc32',
            'device': {'name': 'test', 'compute_capability': '8.6',
                       'total_memory_bytes': 24 * 1024**3},
            'cases': [dict(case, times_ms=[1.0, 2.0, 3.0], median_ms=2.0,
                           finite=True, max_abs_error=0.0,
                           relative_l2_error=0.0, buffer_bytes=1024)
                      for case in cases]}

def test_gqa_projection_uses_four_kv_heads():
    cases = build_cases(glm_config())
    assert next(c for c in cases if c['id'] == 'prefill_qkv_b32')['n'] == 3072

def test_summary_cannot_claim_asr_acceptance():
    cases = build_cases(glm_config(), chunks=1, encoder_batch=1,
                        prefill_batch=1, decode_batch=1)
    summary = summarize(cases, valid_native_result(cases))
    assert summary['asr_acceptance'] is False
    assert summary['metric_kind'] == 'isolated_gemm_estimate'
    assert 'attention' in summary['omitted_operations']
```

- [x] Cover non-divisible batches, zero/boolean/oversized parameters, unsupported
  model dimensions, stable TSV formatting, missing/extra/duplicate results,
  nonfinite and negative results, even-count medians and CLI missing paths.
- [x] Implement the functions and CLI with standard-library imports only.
- [x] Run targeted pytest and compileall; ensure importing this module does not
  load torch, transformers or any inference framework.

## Task 3: Integrate, independently verify and record the gate

**Files:** native/asr/README.md, docs/agents/native-asr-replay.md,
docs/superpowers/plans/2026-10-22-native-asr-replay.md.

- [x] Generate manifests from the local official config, execute small safe
  diagnostic cases in all three modes sequentially and validate JSON results.
- [x] Run an available full shape scenario without interfering with other
  services; label shared-GPU results diagnostic and record compute process IDs,
  model-config SHA256, command, binary SHA256 and commit.
- [x] Have an independent verifier run executable self-tests, failure cases,
  the result summarizer and dependency inspection. It must report evidence,
  not change implementation to make checks pass.
- [x] Record precision errors, missing operations and limits. Store generated
  raw experiment output under ignored bench/results/native-asr/; commit a
  small reviewed evidence summary, never large generated tensors.
- [x] Commit reviewed changes on feat/native-asr-540 and create a draft PR.
- [x] Select the next primitive candidate using measured evidence. Write the
  next independently executable minimal-complete-engine plan before touching
  that subsystem. A successful replay is permission to evaluate that candidate,
  not a claim of 540 or a reason to run the product pipeline.

## Self-review and execution rulings

- This plan implements only the first independent subsystem of the approved
  design. Complete GLM inference, cascade routing and ForceAligner are explicit
  downstream work, not hidden stub functions in this deliverable.
- Maintainer approved execution; continue natively without asking which
  implementation mode to use or requesting design approval again.
- Use installed native executor/verifier roles and bounded, disjoint ownership.
  The project AGENTS.md governs delegation over external workflow defaults.
- Missing local superpowers execution skills were found at their upstream
  source. No skill or project dependency is installed by this plan.
