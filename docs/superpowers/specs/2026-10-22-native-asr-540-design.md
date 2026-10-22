# Native ASR on RTX 3090: approved design

Accepted by the maintainer on 2026-10-22: “同意执行”. This records the design
approved in the conversation; it is not a performance result.

## Fixed requirements

- One RTX 3090 with 24 GB VRAM.
- ASR-only hot RTFx >= 540: one hour of audio in at most 6.666667 seconds.
- A self-developed C++/CUDA inference runtime. No vLLM, SGLang, TensorRT,
  PyTorch or libtorch in the production inference path.
- CUDA, cuBLAS/cuBLASLt, cuFFT, CUTLASS/CUB, FA2/FlashInfer and Hugging Face
  tokenizers may supply mathematical or tokenization primitives.
- Official model weights only; no VideoMiner source, executable, bridge or
  generated assets.
- E2E is measured separately and may be slower than 540 RTFx.
- No new project dependency, changes to local config.yaml, replacement of the
  existing entry points, or full ingest-to-burn execution during engine work.

## Two strategies

Route 2 runs GLM-ASR-Nano-2512 over all audio. Compare FP16 weights with explicit
FP16 GEMM accumulation against an encoder using INT8 weights and activations
(W8A8), with the projector and decoder retaining FP16. Numerically sensitive
operations may retain FP32. Quantization and scale application are timed.

Route 3 runs a lightweight draft ASR over all audio, then GLM over selected
regions. Both models must execute on our runtime. Shared infrastructure means
loaders, primitive implementations, allocators and graph management code; it
does not mean shared weights, vocabulary, frontend parameters, encoder results,
execution graphs or KV buffers. Qwen3-ASR-0.6B is a draft candidate, not an
approved replacement or a proven fast implementation.

Route 2 is the primary strategy. Route 3 requires a positive runtime budget and
a quality upper-bound experiment before implementing a real router. All draft,
routing, expanded-region GLM execution, merging and within-run model swapping
contribute to ASR-only wall time. Selected regions are batched before GLM
execution, and their actual batch sizes determine performance estimates.

## Execution architecture

Own the model graph, scheduler, allocations, KV management and lifecycle. For
the initial 30-second chunk contract, a one-hour input has 120 chunks. Encoder,
prefill and decode choose their batch sizes independently. Preallocate
contiguous KV and capture graph buckets for active sequence counts. Add paged
KV only if measurements establish a benefit.

The minimal engine is PCM -> mel -> complete encoder -> projector -> greedy
decoder -> text. It uses fixed official weights, prompt and chunking before
adding service management or broader model support. Initial model loading may
be outside a hot measurement; input-dependent work may not.

## Evidence gates

1. Replay real model matrix shapes to compare primitive implementations and
   working-set requirements. Label results as isolated primitive measurements,
   never ASR RTFx. Synthetic data, repeated buffers, omitted operations and
   extrapolation assumptions must be disclosed. A replay cannot prove 540.
2. Execute the minimal complete GLM engine on fresh PCM. Recompute mel and
   encoder outputs each run. Keep normal within-request decoder KV.
3. Measure independent, non-looped one-hour Chinese and English audio, three
   hot runs and their median. Include frontend, encoder, prefill, decode and
   final text output. Record actual generated tokens and truncation.
4. Compare quality against current GLM on the same human-transcribed data,
   reporting Chinese CER, English WER, omissions, repetition, silence
   hallucinations, names and numbers. Quality non-regression is the conservative
   design assumption; BF16 agreement is not ground-truth accuracy.
5. For route 3, first use labels to choose ideal review regions and obtain an
   optimistic quality/time curve. If it cannot meet the budget, reject that
   draft candidate without building a router. Actual routing is evaluated on
   separate videos and cannot inherit the ideal-router result.
6. Have an agent independent of implementation verify each executable gate.
   Benchmark runs sharing a GPU with unrelated compute processes are diagnostic
   only. No existing service is stopped without applicable authorization.

## Progression and stopping

Primitive replay and complete inference are separate sub-projects and plans.
After the replay gate, select a measured primitive candidate and implement the
complete vertical slice. Do not infer end-to-end throughput by multiplying
independent optimization estimates. If a candidate fails, record the measured
gap and evaluate the remaining approved candidate; the GPU and 540 target stay
fixed. Do not claim a particular implementation failure proves all engines
impossible.

Only after a strategy passes ASR-only quality and throughput gates, implement
the self-developed ForceAligner path and connect the stage-level backend.
Maintain the current pipeline artifact contract and report ASR+ForceAligner
and ingest-to-burn E2E separately.

## Relationship to existing decisions

This design supersedes the full-self-developed-FP16-only implementation detail
in ADR-0001 and issues #33/#37: self-development remains mandatory, while W8A8
and conditional cascade execution are permitted research candidates. The
swap-GPU/lower-target choices in #38 are excluded by the latest maintainer
instructions. Existing 300-323 estimates and the cached vLLM result are not
acceptance evidence.
