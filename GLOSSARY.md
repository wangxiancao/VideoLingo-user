# VideoLingo

> Reconstruction note (2026-10-23): the original glossary was verified never committed (2026-10-22); this file was rebuilt from the wayfinder map (#33) and its decision tickets and landed in a70ede7. Provenance recorded in the v0.6 closeout (#43).

VideoLingo turns a video into translated subtitles and a burned-in video through a six-stage pipeline. This glossary fixes the vocabulary for the ASR-engine decision.

## Language

**ASR-only**:
The ASR engine metric covering the audio frontend (mel), encoder, and decoder text output; it excludes ForceAligner, ingest, language detection, and translation.
_Avoid_: ASR pipeline, transcription-only

**End-to-end (E2E)**:
A metric covering a complete product path (at minimum ASR + ForceAligner; separately ingest → burn), distinct from ASR-only.
_Avoid_: full pipeline (ambiguous)

**RTFx**:
Audio duration divided by wall-clock time for the measured scope. Always qualify with ASR-only or E2E, and hot or cold.
_Avoid_: speedup, real-time factor (unqualified)

**Hot / cold**:
Hot means the model or engine is resident after a warmup and repeated runs; cold includes model load and service startup.
_Avoid_: warm (ambiguous)

**FP16 accumulation**:
Weights and GEMM accumulation are FP16. It is distinct from BF16 weights with FP32 accumulation, and from FP16 weights with FP32 accumulation.
_Avoid_: FP16 (unqualified)

**Self-developed engine**:
Our own inference runtime that owns the scheduler, paged KV, kernels, audio frontend, timestamp logic, and daemon. It uses official model weights and allowed third-party math libraries, but not vLLM.
_Avoid_: custom model, wrapper (ambiguous)

**ASR model / ASR engine**:
The ASR model is the official GLM-ASR-Nano-2512 weights and architecture; the ASR engine is our runtime that executes it.
_Avoid_: engine as model

**ForceAligner**:
The inference engine for the official Qwen3-ForcedAligner-0.6B weights that produces word-level timestamps; it is separate from the ASR engine.
_Avoid_: aligner (unqualified)