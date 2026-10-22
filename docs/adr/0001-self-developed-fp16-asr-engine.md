# Self-developed FP16 ASR/ForceAligner engine; vLLM rejected

Status: accepted (2026-10-22)

> Reconstruction note (2026-10-23): the original ADR was verified never committed (2026-10-22); this durable summary was rebuilt from the wayfinder map (#33) and its decision tickets and landed in a70ede7 (#43).

VideoMiner's 540× ASR-only reference comes from a self-written C++/CUDA engine, not vLLM. On an RTX 3090 24 GB, BF16 with FP32 accumulation cannot reach 540 (about 440 TFLOP/h against 71.2 TFLOPS dense; 540 needs 92.8% peak), so we choose FP16 weights and FP16 accumulation (142.3 TFLOPS; 540 needs 46.4%) and a narrow self-developed engine for the official GLM-ASR-Nano-2512 + Qwen3-ForcedAligner-0.6B weights. Production runtime may use CUDA math libraries, CUTLASS, FA2/FlashInfer, FFmpeg, and tokenizers; vLLM, TensorRT, and PyTorch/libtorch are excluded from production, with PyTorch allowed only for reference tests. No VideoMiner source, binary, bridge, or mel asset is reused. This decision is subject to the FP16 quality gate and the ASR-only 540 contract; if FP16 fails either, the target GPU or target RTFx must be revisited rather than silently relaxing the gate.