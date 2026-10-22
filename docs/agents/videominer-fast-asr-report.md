# VideoMiner「快速转录引擎」容器取证报告

> 重建说明（2026-10-23）：原件经 2026-10-22 核实从未提交；本文从 wayfinder 地图 #33 与取证票评论重建，随 a70ede7 落盘，非原件。取证素材 `/tmp/videominer-inspect` 与脚本归档 `bench/results/fp16-probe/scripts/` 仍在，可交叉核验（#43）。

- 取证对象：`videominer-inspect-20261022`（镜像 `jaceju68/video-miner:cuda-latest`，retag 自 `docker.1ms.run/jaceju68/video-miner:cuda-latest`）
- 取证方式：只读 `docker inspect/exec/cp` + 宿主机 `file/ldd/readelf/nm/objdump/strings`；未启动应用、未下载权重、未跑任何推理、未改动容器与 `/root/VideoLingo`。
- 标注约定：**【观察】** 为命令/文件直接可见；**【推断】** 为由符号、反汇编、配置和文档推导，未做运行时验证。

---

## 1. 结论摘要

1. **它不是 vLLM、TensorRT 或 TVM 图执行。** 主 ASR 与 ForceAligner 都是自研的 C++/CUDA 可执行文件（PIE、未 strip、内嵌 `sm_86` cubin），只依赖 cuBLAS/cuBLASLt/cuFFT/cuDART；没有 PyTorch、ONNX Runtime、TensorRT、TVM runtime。TVM 仅以 `libtvm_ffi.so` 的形式服务于 CuteDSL 生成的 FlashAttention2 bridge。
2. **吞吐来自「30 秒定长分块 + 大批量提交 + 常驻 daemon + Paged KV/连续批处理 + FA2 编码器注意力 + 贪心解码」**，而不是某个单一魔法 kernel。CLI 默认一次 `BATCH 128`（约 64 分钟音频），Web/WSR 路径固定 32（约 16 分钟音频）；模型只加载一次。
3. **显存策略是「BF16 权重 + 设备级 90% 上限 + 800 MiB 余量 + 按剩余显存分配 Paged KV」**。GLM-ASR 权重约 4.52 GB，ForceAligner 约 1.84 GB；两者在 Web 路径顺序执行且 ASR 进程先退出。按预算推算，8 GB 可跑 ASR-only，但这不等于 8 GB 能跑出 540×，更不等于整机 8 GB 能同时容纳 VLM/OCR。
4. **官方镜像里的 GLM-ASR 路径实际不可用（重要异常）**：`bin/` 下 6 个脚本（`glm_asr_infer`、`force_aligner`、`glm-asr`、`glm-asr-doctor`、`detect_arch`、`select_fa2_bridge`）权限都是 `664`（无执行位），而默认 symlink 又指向 `sm120`；即使补上模型权重，默认调用也会 `Permission denied`。默认引擎是 FunASR-GGUF，GLM-ASR 是可选引擎。
5. **我们不需要复制它的二进制。** 本地 venv 已是 `torch 2.11.0+cu129`、`transformers 5.14.1`、`vLLM 0.24.0`；transformers 原生有 `GlmAsrForConditionalGeneration`，vLLM 0.24 同时注册了 `GlmAsrForConditionalGeneration` 和 `Qwen3ASRForcedAlignerForTokenClassification`。Paged KV、连续批处理、FA2 这些机制可以在 Python/vLLM 里复现大部分；不可复现的是它的 C++ 调度器、逐架构 cubin 和定制 NVFP4 路径。

---

## 2. 容器与运行时事实

### 2.1 镜像与容器

| 项 | 值 | 来源 |
|---|---|---|
| 镜像 ID | `sha256:b9c909a2dc1e411809a1d54a689d216f5c18ba3cdda2e720ce115eaf06019356` | `docker image inspect` |
| 镜像 digest | `sha256:c88d260ca9922abdd78cd49071c38854fc30d31eff32fd4828d1f16a9c7e8cfe`（`docker.1ms.run/...` tag） | `docker image ls --digests` |
| 大小 | `5,404,036,244` bytes（5.4 GB），**单层** | image inspect |
| 创建时间 | `2026-10-15T14:35:32.106917868Z` | image inspect |
| 架构 | `linux/amd64` | image inspect |
| Labels | `null` | image inspect |
| 容器 ID | `7654594daf5c...`，创建于 `2026-10-22T04:08:30Z` | container inspect |
| 启动命令 | `["/bin/bash","-lc","sleep infinity"]`，覆盖了原 entrypoint | container inspect |
| 用户 | 容器 `root`；镜像默认 `vidgo` | inspect |
| 网络 | `none` | HostConfig |
| GPU | `--gpus all`，`NVIDIA_VISIBLE_DEVICES=all` | HostConfig |
| 挂载 | `/app/database`、`/app/media`、`/app/models`、`/app/config` 四个命名 volume | container inspect |
| 应用状态 | 只有 PID 1 `sleep infinity`；无 gunicorn/uvicorn 监听 | `ps -eo pid,comm,args`、`ss -ltnp` |

镜像原 entrypoint 是 `/app/entrypoint.sh`，默认 Cmd `gunicorn`；本次被 `sleep infinity` 覆盖，因此没有启动 Django。

### 2.2 版本

| 项 | 值 | 命令 |
|---|---|---|
| OS | Ubuntu 24.04.1 LTS (Noble) | `/etc/os-release` |
| 内核（容器内可见） | `5.15.0-139-generic` | `uname -a` |
| glibc | 2.39-0ubuntu8.3 | `ldd --version` |
| Python | 3.12.3（`/usr/bin/python3`） | `python3 -V` |
| pip | **不存在**（`No module named pip`） | `python3 -m pip list` |
| CUDA toolkit | 12.8.1；`libcublas 12.8.4.1`、`libcufft 11.3.3.83`、`libcudart 12.8.90`、cuDNN 9.8.0.87 | env + `dpkg -l` |
| NVIDIA driver | 570.144（内核模块 2025-04-10） | `nvidia-smi`、`/proc/driver/nvidia/version` |
| GPU | **RTX 3090 24,576 MiB，compute cap 8.6** | `nvidia-smi --query-gpu` |
| 取证时显存 | 约 9.6 GiB used / 14.1 GiB free（宿主机共享，非容器占用；数值随宿主进程波动） | 同上 |
| ffmpeg | 6.1.1-3ubuntu5 | `ffmpeg -version` |
| 引擎构建编译器 | GCC 11.4.0（Ubuntu 22.04） | `readelf -p .comment` |
| `file/readelf` | 容器内**没有**；已 `docker cp` 到宿主机分析 | `file: command not found` |

### 2.3 Python 包（镜像内 80 个）

`pip` 缺失，用 `importlib.metadata` 枚举。关键结论：**没有 torch / transformers / vllm / tvm_ffi**；唯一存在的推理运行时是 `onnxruntime-gpu`。

- `onnxruntime-gpu 1.30.0`（仅用于 VLM/OCR/VAD，不用于 GLM-ASR）
- `tokenizers 0.23.3`（仅 orchestrator 的可选 prompt 用）
- `numpy 2.5.4`、`scipy 1.18.1`、`soundfile 0.14.0`、`pydub 0.25.1`
- `huggingface_hub 2.2.0`、`gguf 0.19.0`、`av 19.0.1`
- **缺失**：`torch`、`transformers`、`vllm`、`tvm_ffi`、`flash-attn`。`python3 -c "import vllm"` → `ModuleNotFoundError`，全盘 `find -iname "*vllm*"` 为空。

### 2.4 权重是否在镜像里

**不在。** `/app/models` 是空 volume（只有目录本身）；`glm-asr-doctor` 明确报告：

```text
[models]
/app/models/glm-asr/glm-asr-nano-2512
  missing config.json
  missing model.safetensors
/app/models/glm-asr/qwen3-forcealigner-0.6b
  missing config.json
  missing model.safetensors
```

镜像内也没有 `/app/logs` 以外的 GLM 运行日志或 benchmark 产物；唯一日志是 `fun-asr-stack/.../logs/latest.log`，来自开发机 RTX 5070 Ti（见 §8 证据索引）。

---

## 3. ASR 引擎组成与架构

### 3.1 目录与文件（`/app/asr_utils/glm-asr-stack`）

| 路径 | 大小 | 说明 |
|---|---:|---|
| `Infer-engine/glm_asr_infer_sm86` | 685,968 | **sm86 主 ASR 引擎**（本机 3090 用这个） |
| `Infer-engine/glm_asr_infer_sm89` | 681,872 | sm89 重编译版本（与 sm86 不同 hash） |
| `Infer-engine/glm_asr_infer_sm120` | 862,112 | sm120 版本 |
| `Infer-engine/glm_asr_infer` | symlink → `glm_asr_infer_sm120` | 发行包默认链接，**不是**按本机架构 |
| `forcealigner-engine/force_aligner_sm86` | 651,560 | sm86 ForceAligner |
| `forcealigner-engine/force_aligner_sm89` | 651,560 | sm89，hash 不同 |
| `forcealigner-engine/force_aligner_sm120` | 745,792 | sm120 |
| `forcealigner-engine/force_aligner` | symlink → `force_aligner_sm120` | 默认链接 |
| `forcealigner-engine/mel_filterbank.bin` | 102,912 | 128×201 float32（aligner 从 cwd 读） |
| `resources/mel_filters.bin` | 102,912 | 128×201 float32（ASR 用；与本地 repo 同 hash） |
| `cutedsl_fa/fa2_bridge_sm86.so` | 163,520 | CuteDSL/TVM-FFI FA2 bridge |
| `cutedsl_fa/fa2_bridge_sm89.so` | 163,520 | 同上，sm89 |
| `cutedsl_fa/fa2_bridge_sm120.so` | 312,392 | 同上，sm120 |
| `cutedsl_fa/fa2_bridge.so` | symlink → `fa2_bridge_sm120.so` | 默认链接 |
| `lib/libtvm_ffi.so` | 1,548,608 | TVM FFI ABI（不是 TVM 编译器/运行时） |
| `bin/orchestrator.py` | 64,686 | 长音频编排 |
| `bin/glm_asr_infer` / `bin/force_aligner` | 1,108 / 628 | 架构选择 wrapper（**664，无执行位**） |
| `bin/glm-asr` / `bin/glm-asr-doctor` / `bin/detect_arch` / `bin/select_fa2_bridge` | 2,409 / 1,693 / 441 / 360 | 全部 **664** |

`resources/mel_filters.bin` 的 sha256 与本地 checkout 完全一致：`0121c2da4b578323fd4704023378c5277619565ee398fdee05a29ce5fbba6eec`。

### 3.2 主 ASR 引擎：自研 C++/CUDA，不是框架

`file` 结果：

```text
ELF 64-bit LSB shared object, x86-64, dynamically linked,
interpreter /lib64/ld-linux-x86-64.so.2,
BuildID[sha1]=2b2427d1e59776cef3c9ae64e2db73c7001454b5,
for GNU/Linux 3.2.0, not stripped
```

它是 PIE 可执行文件（有 `.interp`），只是 ELF type 显示为 `DYN`。

- **依赖**：`libcublas.so.12`、`libcublasLt.so.12`、`libcufft.so.11`、`libcudart.so.12`、`libstdc++`、`libm`、`libc`。没有 `libtorch`、`libnvinfer`、`libonnxruntime`、`libtvm`。
- **RPATH**：`/data/fwsr/glm-asr/glm-asr-stack/lib:/usr/local/cuda-12.8/lib64`（开发机绝对路径泄漏）。
- **导入的 CUDA API**：`cublasGemmEx/StridedBatchedEx`、`cublasLtMatmul`、`cufftPlanMany/ExecR2C`、`cudaMalloc/Memcpy/LaunchKernel`、`cudaMemGetInfo`、`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 等；**没有** `cudaGraph*`、`cudaStreamCreate`（单流）。
- **导出符号**：`nm -D --defined-only` 只有 `stdin/stdout/stderr` 三个，说明主入口是静态链接的内部代码，没有稳定的 C ABI。
- **本地符号 296 个文本符号（未 strip）**，足以还原实现轮廓：
  - 模型：`model_load`、`safetensors_load`、`WeightStore`、`ModelConfig`、`audio_tower`/`language_model` 权重名。
  - 编码器：`encoder_forward`、`encoder_attention_cublas`、`encoder_attention_flash_fa2_launch`、`encoder_flash_attention_tile_kernel_fa2`、`encoder_flash_attention_tile_kernel_pv_mma_q32_k32`、`concat_4frames_kernel`、`conv1d_forward`、`mel_kernel`。
  - 解码器：`decoder_prefill`、`decoder_prefill_paged[_batched]`、`decoder_decode_step[_batched/_paged]`、`paged_decode_attention*`、`reshape_and_cache_kernel`、`build_slot_mapping`、`block_table_*`、`kv_pool_*`。
  - 调度：`scheduler_init/schedule/postprocess/preempt_sequence`、`seq_*`、`mixed_batch`、`engine_submit_with_prompt_ids`、`engine_transcribe`。
  - 量化：`fp4_dequant_kernel`、`quantize_bf16_to_fp4_kernel`、`apply_prequant_kernel`、`swizzle_rowwise_scales_kernel`、`nvfp4_linear`、`nvfp4_cutlass_available`。
  - 音频 I/O：`drwav_*`、内嵌 ffmpeg 命令行（`ffmpeg -nostdin ... -f f32le ...`）。
- **内嵌 cubin**：`.nv_fatbin` 大小 `0x65e98`，`strings` 只看到 `.target sm_86`；未用 `cuobjdump` 验证是否额外内嵌 PTX。sm89/sm120 是分别重编译的二进制（hash 不同），不是一份 fatbin 多架构。
- **CUDA 版本**：`.comment` 只给 GCC 11.4.0；RPATH 指向 CUDA 12.8；cubin 由 CUDA 12.8 nvcc 生成（`tmpxft_*` 中间名）。
- **源文件名**：`cuda/encoder.cu`、`cuda/decoder.cu`、`cuda/kernels.cu`、`cuda/mel.cu`、`paged_kv.cu`、`nvfp4_cutlass.cu`、`kv_pool.c`、`sequence.c`、`safetensors.c`、`audio_io.c`、`tokenizer.cpp`、`src/safetensors.cpp`。这些源文件**不在镜像里**。

**结论【观察+推断】**：这是一个手写的、面向 GLM-ASR 架构的 C++17/CUDA 推理引擎，结构上类似「小型 vLLM」：连续批处理 + Paged KV + 自定义 GQA decode kernel + 可选 FA2 encoder。它不是 vLLM 的封装，也不是 TensorRT/TVM 产物。

### 3.3 ForceAligner：另一个自研 C++/CUDA 引擎

`force_aligner_sm86`（651,560 B）同样是 PIE、未 strip、只依赖 cuBLAS/cuFFT/cuDART，`.nv_fatbin` 内 `sm_86`。

- 符号：`aligner_model_load`、`aligner_align[_with_features]`、`tokenizer_load`、`build_alignment_prompt`、`extract_timestamps`、`timestamps_to_words`、`fix_timestamps_lis`、`write_srt`、`cuda_audio_encoder_forward`、`cuda_text_decoder_forward`、`flash_attention_q32_k32_kernel`、`attention_prefill_cublas`、`mel_project_kernel`。
- **【观察+推断】** 它不是自回归生成：一次 forward 得到每个 `<timestamp>` 位置的分类 logits（`classify_num=5000`、`timestamp_segment_time=80` ms、`timestamp_token_id=151705`），再抽时间戳、做 LIS 单调修正、输出词级 `text\tstart\tend`。
- 配置读取：`config.json` 的 `thinker_config.audio_config/text_config`、`n_window`、`n_window_infer`、`max_source_positions`、`classify_num`、`audio_token_id` 等；tokenizer 读 `vocab.json` + `merges.txt` + `tokenizer_config.json`（不是 `tokenizer.json`）。
- 音频：WAV mono 16 kHz；`mel_filterbank.bin` 从 cwd 读，`n_fft=400`、`hop_length=160`、`n_mels=128`。
- 运行形态：`--daemon` 从 stdin 读 `AUDIO<TAB>TEXT<TAB>LANGUAGE`，每次一个 chunk，输出若干 `word\tstart\tend` 行 + 空行；**单请求串行**，没有 batch/paged KV。

### 3.4 FlashAttention bridge：CuteDSL 生成 + TVM FFI

`fa2_bridge_sm86.so` 的依赖：

```text
libtvm_ffi.so => /app/asr_utils/glm-asr-stack/cutedsl_fa/../lib/libtvm_ffi.so
libcudart.so.12
```

RUNPATH：`$ORIGIN/../lib:/data/fwsr/glm-asr/GLM-ASR/venv/lib/python3.12/site-packages/tvm_ffi/lib`。

`strings`/`nm` 证据：

- `p_tvm_ffi_extension.c.o`、`__tvm_ffi_fa2_enc_varlen_fwd`、`__tvm_ffi_fa2_dec_varlen_fwd`
- `__cute_internal_fa2_enc_varlen_fwd_cuda_init/load/load_to_device/num_binaries`
- `kernel_cutlass_kernel___main__FlashAttention2VarlenSm80_object_at__tensorptrbf16gmemalign16...`
- `cuda_dialect_init_library_once`、`cuda_dialect_*`、`cuLibraryLoadData`、`cuTensorMapEncodeTiled`
- 错误签名：`fa2_enc_varlen_fwd(mQ: Tensor([n0,n1,n2], bfloat16), mK, mV, mO, softmax_scale: float32, max_seqlen: int32, num_seqs: int32, mCuSeqlens: Tensor([n12], int32), stream: Stream)`

**判断【观察】**：这是 NVIDIA CuteDSL（CuTe DSL）生成的 CUTLASS FlashAttention2 varlen kernel，通过 TVM FFI 扩展 ABI 暴露成 `.so`，每个架构单独编译/打包。`libtvm_ffi.so` 只有 FFI/对象/张量/DLPack 符号（93 个导出），没有 TVM 编译器或图执行器；它只是 ABI 层。

引擎还定义了 fallback：
- `FA2 CuteDSL: loaded direct-link kernel`
- `FA2 CuteDSL: not available, trying vLLM...`
- `FA2: dlopen vllm.so: %s`
- `FA2 thin: loaded (gqa=%d)`
- `FA2 launch failed, falling back`
- `encoder_attn_paths fa_calls=... pipeline_calls=... cublas_calls=...`

**重要【观察+推断】**：镜像里**没有 vLLM**，也没有 `libthin_fa2_wrapper.so`；字符串里的 vLLM 路径是构建机 venv 的绝对路径。因此 `GLMASR_ENCODER_FA_VLLM=1` 在这个镜像里大概率 dlopen 失败，实际走 CuteDSL bridge；若 bridge 架构不匹配（默认 symlink 是 sm120，而本机是 sm86），再退到自定义 `pv_mma_q32_k32` 或 cuBLAS。我们没有跑推理，无法确认最终命中哪条路径。

### 3.5 它「不是什么」

| 候选 | 是否使用 | 证据 |
|---|---|---|
| vLLM runtime | **否** | 无 `vllm` 包、无 `_vllm_fa2_C` 文件；只有可选 FA2 `.so` 的 build path 字符串 |
| TensorRT | 否 | 无 `libnvinfer`；`ldd`/`nm` 无 TensorRT 符号 |
| ONNX Runtime | 否（ASR） | 主引擎不链接 onnxruntime；ORT 只用于 VLM/OCR/VAD |
| TVM 编译器/图执行 | 否 | 仅 `libtvm_ffi.so` ABI，无 `libtvm.so`、无 graph executor |
| PyTorch | 否 | 无 `libtorch`；镜像无 torch |
| FlashAttention 官方包 | 否 | 无 `flash-attn`；FA2 是 CuteDSL 自生成 kernel |
| 自定义 CUDA kernel | **是** | 296 个本地符号，含 paged decode、GQA、RoPE、RMSNorm、SiLU、FP4 dequant 等 |
| 自定义 Paged KV / 调度器 | **是** | `kv_pool.c`、`sequence.c`、`scheduler_*`、`block_table_*`、`preempt_sequence` |

---

## 4. 编排与批处理

### 4.1 四条编排路径

| 路径 | 入口 | ASR batch | 说明 |
|---|---|---:|---|
| Web/WSR（选择 GLM 引擎时） | `asr_utils/glm_asr_wsr.py` → `video.asr_engine.ASREngineDaemon` | **32** | 进程内复用常驻 daemon；转完调用 `stop()` 释放 ASR，再起 ForceAligner |
| CLI 单视频 | `bin/orchestrator.py` `run_pipeline` | **128**（`--batch-size` 默认） | 先 ASR 全部 chunk，再对齐 |
| CLI 队列 | `bin/orchestrator.py --run-queue` | 128 | `InferDaemonPool` 跨任务常驻；**对齐阶段 ASR 进程仍在** |
| 流式 | `video/stream_transcriber.py` | 由 live VAD 决定 | 同一 `asr_engine_daemon` |

### 4.2 分块

`bin/orchestrator.py` 顶部常量：

```python
SEGMENT_SEC = 30.0
OVERLAP_SEC = 0.0
SAMPLE_RATE = 16000
ASR_WINDOW_CHUNKS = 256
```

- 输入先 `ffmpeg -ac 1 -ar 16000 -acodec pcm_s16le` 解成单声道 16 kHz int16 WAV。
- `split_wav()` 按 30 s 切块，默认 **0 overlap**；`--segment-sec`/`--overlap-sec` 可改，但 `merge_to_srt()` 只做时间轴 clamp，不做重叠文本去重，实际应保持 0。
- 1 小时音频 = 120 个 chunk；每个 chunk 是独立 WAV 文件（`chunk_000000.wav` …）。
- 没有对整段音频做 VAD 再切分；VAD 只用于 live session 的追加块（Silero/FireRed）。

### 4.3 批处理与调度

- CLI：`for start in range(0, len(segments), batch_size)`，默认 `batch_size=128`；一次 `BATCH 128` 提交约 64 分钟音频。
- WSR：硬编码 `batch_size = 32`；一次约 16 分钟音频。
- 队列模式先把多个任务拼成一个 window，默认 `--asr-window-chunks 256`，再按 128 切 batch。
- 引擎内部还有第二层限制：
  - `GLMASR_MAX_SEQS=256`：最多 256 个并发序列；
  - `GLMASR_MAX_BATCHED_TOKENS=10000`：每个 prefill 调度步的 token 预算；
  - Paged KV 的 block 数由剩余显存决定。
  因此 `BATCH 128` 是**提交队列深度**，不等于 128 个序列同时驻留 KV。**【推断】** 实际并发由 token 预算和 KV block 共同限制。

### 4.4 Infer-daemon 协议

`bin/glm_asr_infer` 是架构 wrapper；真正的二进制支持两种模式：

```text
Daemon mode (stdin/stdout):
  glm_asr_infer_<arch> --daemon

One-shot mode:
  glm_asr_infer_<arch> --model DIR --audio FILE [--audio FILE ...] --mel-filters FILE
```

daemon 协议（`orchestrator.py` 的 `InferDaemon` 与 `video/asr_engine.py` 一致）：

```text
LOAD <model_dir>          -> READY
MEL <mel_filters_path>    -> READY
PROMPT_IDS <n> <id...>    -> READY   # n=0 表示清空
BATCH <n>
<audio_path_1>
...
<audio_path_n>
RUN
  <path>: <text>          # 每个输入一行；空文本也输出
  INFER_TIME: <ms> (<n> samples, <ms>/sample)
  OUTPUT_TOKENS: <n> (<tok/s>, <tok/sample>)
  READY
QUIT
```

- `LOAD` 必须在 `MEL` 前；`RUN` 前必须先 `LOAD` + `BATCH`。
- 错误走 stderr，daemon 可能直接退出；`ASREngineDaemon` 会捕获 stderr 尾部 30 行作为错误信息。
- `PROMPT_IDS` 把用户 prompt/hotwords 的 token id 加到每个序列前；orchestrator 用 `transformers/tokenizers` 编码，限制 512 token。WSR 路径**没有**传 prompt，所以 Web UI 的热词不会进入 GLM 引擎。

### 4.5 ASR → ForceAligner 的顺序与显存释放

| 路径 | 顺序 | ASR 显存是否在对齐前释放 |
|---|---|---|
| WSR/Web | `asr_engine_daemon.transcribe()` 全部 batch → `asr_engine_daemon.stop()` → `FADaemon` | **是**（QUIT + terminate，进程退出） |
| CLI 单视频 | `InferDaemon` 全部 batch → `infer.close()` → `FADaemon` | **是**（QUIT，进程退出） |
| CLI 队列 | `InferDaemonPool` 先跑完 ASR，再跑 align；pool 在 `run_queue` 结束才 `close_all()` | **否**（ASR 进程与模型常驻，VRAM 叠加） |
| 流式 | 同一个 daemon 持续跑；对齐由上层另行触发 | 取决于调用方 |

WSR 代码原文（`glm_asr_wsr.py`）：

```python
# 4. Release ASR daemon GPU memory
asr_engine_daemon.stop()
progress_cb("ASR: Done, released GPU")

# 5. ForceAligner
fa = FADaemon(fa_model_dir)
```

对齐是**逐 chunk 串行**的：`for seg in segments: fa.align_one(...)`，没有 batch。每个 chunk 的 `words` 时间戳是 chunk 内相对时间，`merge_to_srt()` 再加 `seg["start_sec"]` 偏移。

### 4.6 对齐与 SRT

- ForceAligner 输出词级 `(text, start, end)`；`fix_timestamps_lis` 在引擎内做 LIS 单调修正。
- `subtitle_mode="word"`：每个词一条 cue；`"sentence"`：用标点分句，再用词序列做模糊匹配，把句子边界映射到词索引。
- `merge_to_srt()` 有全局 `seen_end`：后一条 cue 若完全落在前一条结束之前会被丢弃，若部分重叠会被 clamp 到 `seen_end`。这能掩盖少量 chunk 边界重叠，但不是可靠的去重；默认 0 overlap 是正确选择。
- 空文本或 `[ERROR]` 的 chunk 会被跳过，不产生 cue。

### 4.7 与本地 checkout 的差异

本地 `/root/VideoLingo/external/video-miner/backend/asr_utils/glm-asr-stack/`：

- `bin/orchestrator.py`、`bin/*` 脚本与容器**逐字节一致**（sha256 相同，`diff` 为空）。
- `asr_utils/glm_asr_wsr.py` 与容器一致。
- **没有** `Infer-engine/glm_asr_infer_sm*`、`forcealigner-engine/force_aligner_sm*`、`cutedsl_fa/fa2_bridge_sm*.so`、`lib/libtvm_ffi.so`、`forcealigner-engine/mel_filterbank.bin`；这些都在 `.gitignore` 里。
- 本地有 `resources/mel_filters.bin`，hash 与容器一致。
- 公开 repo 也没有 `scripts/build_infer_arch.sh` / `build_forcealigner_arch.sh`（README 提到但未跟踪）。

---

## 5. 显存参数与 8 GB 的真实含义

### 5.1 环境变量逐项

镜像 env（`docker inspect`）与 wrapper/orchestrator 的默认值一致：

| 变量 | 镜像值 | 作用与证据 |
|---|---:|---|
| `GLMASR_VRAM_UTIL` | `0.9` | `kv_pool_warmup` 里 `getenv`；默认 double `0.9`，有效范围 `(0.01, 1.0]`；`cap = total_VRAM * util`。用于算 KV pool 预算，不是硬性 per-process 上限。 |
| `GLMASR_MAX_SEQS` | `256` | `engine_init` 里 `engine_int_from_env_bounded("GLMASR_MAX_SEQS", 256, 256)`；默认 256、上限 256。分配 per-seq 状态、block table、decode metadata，并作为 scheduler 最大并发序列数。 |
| `GLMASR_MAX_BATCHED_TOKENS` | `10000` | `engine_init` 里默认 10000、上限 65536；prefill 调度步的 token 预算；≤0 时打印 `max_batched_tokens=disabled`（不限制）。 |
| `GLMASR_MAX_NEW_TOKENS` | `256` | `model_config_from_json` 里 `getenv`；编译默认 **500**，clamp 到 `[1, 8192]`；每序列最多生成 token 数。镜像显式设 256。 |
| `GLMASR_ENCODER_FA_PIPELINED` | `1` | `encoder_fa_pipelined_enabled()`，默认 0、范围 0/1；开启单独的 pipeline 路径（profile 里有 `pipeline_calls`）。引擎不导入 `cudaStreamCreate`，所以**不是**多流重叠；更像行/层流水。 |
| `GLMASR_ENCODER_FA_VLLM` | `1` | `encoder_fa_vllm_enabled()`，默认 0；优先尝试 vLLM FA2（`dlopen` 构建机 venv 的 `_vllm_fa2_C.abi3.so`）。镜像无 vLLM，预计回退到 CuteDSL bridge。 |
| `GLMASR_ENCODER_FA_LONG_MIN_SEQ` | `1` | `encoder_fa_long_min_seq_len()`；编译默认 **1500**（=30 s 编码器最大帧数）。设 1 表示所有长度都走 long/FA 路径。 |
| `GLMASR_ENCODER_FA_MIN_SEQ` | `1` | `encoder_fa_legacy_min_seq_len()`；编译默认 **1500**（legacy 阈值，上限 3000）。设 1 表示 1 帧以上都用 FA，几乎禁用 cuBLAS fallback。 |
| `GLMASR_GPU_COMPUTE_CAP` | 未设 | **只**被 `bin/detect_arch` 读取；覆盖 `nvidia-smi compute_cap`，映射 `8.6/86→sm86`、`8.9/89→sm89`、`12.0/120→sm120`，其他值报错。引擎/aligner 二进制不读它。 |

其他可见但未在 env 中设置的旋钮：`GLMASR_ENCODER_FA_ENABLE/FA2/BATCHED/MEDIUM_MIN_SEQ/PV_MMA_Q32_K32/FORCE_FALLBACK/DENSE_TMA`、`GLMASR_ENCODER_QKV_FUSED`、`GLMASR_DECODER_FA_DISABLE`、`GLMASR_BF16_LINEAR_TENSOR_OP/LT_BIAS`、`GLMASR_PROFILE`、`GLMASR_USE_CUTLASS_NVFP4`、`GLMASR_NVFP4_COMPARE_ONCE`。aligner 侧还有 `AFA_FORCEALIGNER_USE_CUBLAS_DECODER`、`FA_WINDOW_INFER_OVERRIDE`、`FA_FORCEALIGNER_USE_CUBLAS`、`FA_DUMP_*`。

### 5.2 KV pool 预算公式（反汇编）

`kv_pool_warmup`（`glm_asr_infer_sm86` @ `0x11e00`）的实际逻辑：

```c
cudaMemGetInfo(&free, &total);
used = total - free;                       // 设备级，包含其他进程
util = getenv("GLMASR_VRAM_UTIL", 0.9);    // clamp 到 (0.01, 1.0]
cap  = total * util;
if (used + 800MiB >= cap) {
    warn("VRAM cap ... already exceeded ... KV pool will be minimal");
    blocks = min_blocks;                   // 4
} else {
    budget = min(free, cap - used - 800MiB);
    blocks = max(4, budget / block_bytes);
}
block_bytes = 4 * block_size(128) * layers * kv_heads * head_dim;
```

- `headroom` 硬编码 **800 MiB**（`0x32000000`，printf 里 push 的也是 `0x320`）。
- `block_size` 硬编码 **128 token**（`edi=0x80`）。
- GLM-ASR 的 `layers=28`、`kv_heads=4`、`head_dim=128` → `block_bytes = 4*128*28*4*128 = 7,340,032 B`（7.0 MiB/block）。
- 每 token KV = `2(K/V) * 28 * 4 * 128 * 2 B = 57,344 B`（56 KiB/token）。
- 最小 pool 4 blocks（28 MiB），用于 cap 已超限时仍能启动。
- `cudaMemGetInfo` 的 `used` 是**整卡**（含宿主机其他进程），所以 `GLMASR_VRAM_UTIL` 不是「本进程最多用 90%」，而是「整卡已用 + 本进程预留不超过 90%」。在共享卡上这会让 KV pool 变小。
- 引擎启动时会把这个预算打印到 stderr：`VRAM: total=... MB, used=... MB, util_cap=...% (... MB), headroom=... MB, kv_budget=... MB`；cap 超限时打印 `WARNING: VRAM cap ... KV pool will be minimal.`

### 5.3 权重与 KV 的账

| 模型 | 参数量（HF safetensors） | BF16 权重 | 备注 |
|---|---:|---:|---|
| `zai-org/GLM-ASR-Nano-2512` | 2,257,843,200 | **4.52 GB / 4,306 MiB** | 单文件 `model.safetensors`；audio encoder 32 层/1280d/20 heads，text decoder 28 层/2048d/16 heads/4 KV heads/head_dim 128/vocab 59264 |
| `Qwen/Qwen3-ForcedAligner-0.6B` | 917,728,896 | **1.84 GB / 1,750 MiB** | 单文件；audio encoder 24 层/1024d，text decoder 28 层/1024d/8 KV heads；`max_source_positions=1500`、`n_window=50`、`n_window_infer=800` |

8 GB 卡的粗略账（**【推断】**，未跑推理）：

- `cap = 8192 MiB * 0.9 = 7,373 MiB`；`headroom = 800 MiB`。
- ASR 权重 4,306 MiB 后，留给 workspace + KV 约 `7,373 - 4,306 - 800 = 2,267 MiB`。
- 若 encoder/decoder workspace 占 1–2 GB，KV 只剩 0.3–1.3 GB，即 40–180 个 block（5k–23k token）。按每 30 s chunk 约几百 token 估，同时只有个位数到二十几个序列，scheduler 会靠 preemption 降并发。
- 因此 **按此预算，8 GB 理论上能跑 ASR-only，但不可能维持 128 并发，也不可能稳定复现 540×**。24 GB 卡上 KV 预算约 15–16 GB，才能容纳 256 序列级别。

### 5.4 「8 GB」到底覆盖什么

- 公开 repo `README.md` 第 179 行、`docs/en/deployment.md` 第 9 行：`Recommended GPU VRAM ≥ 8GB`，上下文是 **Docker 部署前置条件**，不是 ASR 基准。
- `docs/en/transcription/index.md` 只给了 FunASR-GGUF「约 1.5GB VRAM」，**没有**给 GLM-ASR 的显存数字。
- 镜像内没有 8 GB 相关说明（只有 `glm-asr-stack/README.md` 这类栈内文档），也没有 GLM-ASR 的显存数字；`/app/models` 为空。
- 默认转录引擎是 `funasr_gguf`（`set_setting.py:143`），GLM-ASR 是可选；默认 LLM 是云端 DeepSeek（`selected_model_provider=deepseek`、`translate_*_provider=deepseek`）。
- 整机还有 MiniCPM-V 4.5（约 6.88 GB）、GLM-OCR（代码注释称 6–8 GB）、WeMM（约 4.8 GB）；这些模型按 GPU slot 串行，但单个 6.88 GB 模型 + 运行时开销已经接近/超过 8 GB。因此 8 GB 是「默认 FunASR + 小模型/低 GPU 层数」的整机下限，不是「GLM-ASR 540× + VLM + OCR 同时跑」的承诺。

### 5.5 540× 的算术

- `540× RTFx` = 1 小时音频约 `3600/540 = 6.67 s`。
- README 同时说「1 小时视频转录 + 强制对齐约 15–20 s」；若 ASR-only 是 6.67 s，则对齐约 8–13 s，正好符合「对齐逐 chunk 串行、0.9B 模型、120 次 forward」的量级。
- **【推断】** 540× 指 ASR-only（且大概率是 5070 Ti/5090 级别、batch 128），15–20 s 是 ASR+对齐的端到端数字；两者不矛盾，但 540× 不是端到端。
- 本机是 3090 24 GB，且没有权重，无法验证；镜像内也没有 GLM 的 benchmark 日志。

---

## 6. 可复现项 vs 不可复现项

### 6.1 我们 venv 的现状（`/root/VideoLingo/.venv`）

| 包 | 版本 | 与复现的关系 |
|---|---|---|
| torch | `2.11.0+cu129` | CUDA 12.9 build，3090 sm86 可用；SDPA flash/mem-efficient/math 均 enabled |
| transformers | `5.14.1` | 原生 `transformers.models.glmasr.GlmAsrForConditionalGeneration`、`GlmAsrProcessor`；`AutoModelForMultimodalLM` 存在 |
| vLLM | `0.24.0+cu129` | 注册 `GlmAsrForConditionalGeneration`（STT transcribe/translate）与 `Qwen3ASRForcedAlignerForTokenClassification`（pooling token_classify） |
| tokenizers | `0.22.2` | prompt 编码 |
| flash-attn | 缺失 | vLLM 自带 attention backend；transformers 可用 SDPA |
| triton | `3.6.0` | torch.compile / vLLM kernel |
| tvm-ffi | 缺失 | 不需要；CuteDSL bridge 不可复现也不必要 |

### 6.2 可以在 Python/transformers/vLLM 里复现的机制

| 机制 | 复现方式 | 备注 |
|---|---|---|
| 30 s 定长分块、0 overlap、16 kHz mono | 直接照搬 orchestrator 的 `split_wav` 语义 | 最简单、收益最大 |
| 批量提交 chunk | HF `generate` 静态 batch，或 vLLM 连续批处理 | vLLM 的 Paged KV/prefix caching 更接近原实现 |
| `max_new_tokens=256`、贪心 | `generate(max_new_tokens=256, do_sample=False)` | 与原引擎一致 |
| BF16 权重 | `torch_dtype=torch.bfloat16` | 3090 支持 BF16 |
| Encoder FA2 / SDPA | `scaled_dot_product_attention` 或 vLLM 的 FA2/FlashInfer backend | 不需要自写 kernel |
| Paged KV + 连续批处理 | vLLM `LLM(...)` / `AsyncLLMEngine` | 这是最值得复用的部分 |
| ASR→aligner 顺序、释放 ASR | 两个进程/两个 `LLM` 实例，先停 ASR 再起 aligner | 8 GB 必须顺序；24 GB 可重叠 |
| 词级时间戳 + chunk offset + SRT | 自己实现 `merge_to_srt`，注意 cue 单调性 | 对齐模型输出 80 ms bin，需要平滑/LIS |
| 多 chunk 对齐 batch | vLLM pooling 模式可一次 encode 多个 prompt | 原实现是串行，我们可以做得更好 |
| CUDA graphs | `torch.compile(mode="reduce-overhead")` 或 vLLM 默认 | 原引擎没有 cudaGraph；这是我们可以额外拿到的收益 |

### 6.3 不可复现 / 不建议复现的部分

| 项 | 原因 |
|---|---|
| 自研 C++/CUDA 引擎 | 无源码；只有 sm86/sm89/sm120 三个 cubin；无 PTX fallback；依赖 CUDA 12.8 soname |
| 自定义 paged decode / GQA / RoPE kernel | 源码不在镜像；vLLM 已有等价实现 |
| CuteDSL FA2 bridge | 需要 CuteDSL/TVM-FFI 构建链；vLLM/FA2 可替代 |
| 定制 NVFP4 dequant / CUTLASS path | `GLMASR_USE_CUTLASS_NVFP4` 明确打印 "native path is not ready, falling back to bf16 dequant path"；官方权重也是 BF16 |
| 精确的 540× | 需要同型号 GPU、同权重、同 batch；我们无法在 3090 上声明 |
| daemon 协议 | 可以重写，但没有必要；vLLM 的 Python API 更可维护 |
| 二进制复制 | 无 license、无源码、架构绑定、可能含 proprietary 代码；见 §10 |

### 6.4 量化支持结论

| 精度 | 支持情况 | 证据 |
|---|---|---|
| BF16 | **默认、完整** | 权重名、`bf16_linear`、`cublasGemmEx`、官方 config `dtype: bfloat16` |
| FP16 | aligner loader 接受 `F16_BF16`；主引擎重点 BF16 | aligner `dtype` 字符串；主引擎未见 FP16 linear 符号 |
| NVFP4 | **有条件**：能读 `weight_scale`/`weight_scale_2`/`input_scale`/`pre_quant_scale`，但 native CUTLASS 路径未启用，回退到 BF16 dequant；不省显存 | `GLMASR_USE_CUTLASS_NVFP4` 警告、`fp4_dequant_kernel`、`nvfp4_linear`、`nvfp4_cutlass_available` |
| FP8 E4M3 | dtype 字符串存在，但无 FP8 GEMM/dequant kernel；**不支持** | `F8_E4M3` 仅出现在 safetensors dtype 表 |
| INT8/INT4/AWQ/GPTQ/GGUF | **不支持** | 无对应 kernel/字符串；GGUF 属于 FunASR 栈 |
| KV cache 量化 | **不支持** | paged KV 以 BF16 存储；无 KV quant 字符串。`VIDUNDER_KV_CACHE_TYPE=q4_0` 是 VLM llama.cpp 的，不是 ASR |

---

## 7. 对我们 pipeline 的具体建议（按优先级）

### P0：先用 vLLM 0.24 验证「GLM-ASR + ForcedAligner」可行性

- 我们 venv 已装 `vllm 0.24.0`，注册表里两者都有：
  - `GlmAsrForConditionalGeneration`（STT，支持 `transcribe`/`translate`，prompt 由 vLLM 生成："can you transcribe the speech into a written format?"）
  - `Qwen3ASRForcedAlignerForTokenClassification`（pooling，`pooling_task="token_classify"`，docstring 给了 `llm.encode([{"prompt":..., "multi_modal_data":{"audio":...}}], pooling_task="token_classify")` 的用法）
- 先做一个 30 s chunk 的 smoke test：ASR 输出文本，aligner 输出每 token 的 5000 类时间 bin，映射到 80 ms，转词级时间戳。
- 这一步不需要任何 VideoMiner 二进制，也不需要 CuteDSL/TVM-FFI。

### P0：把「30 s 分块 + 0 overlap + 256 max_new_tokens」作为基线

- 30 s 是 GLM-ASR/aligner 的原生窗口（`max_source_positions=1500`、`chunk_length=30`），不要先改 60 s。
- 不要开 overlap；如果以后要开，必须在 merge 阶段做文本级去重，而不是只 clamp 时间。
- `max_new_tokens=256` 与 VideoMiner 一致；先不要放大到 512，避免 KV/时延上升。

### P1：用 vLLM 的连续批处理/Paged KV 替代固定 batch=128

- 固定 128 在 24 GB 上可能可行，在 8 GB 上会被 KV 预算压到很低；让 vLLM scheduler 按 token/block 自动限流。
- 如果必须固定 batch，按「总音频 token 预算」而不是「chunk 数」来定：先测每个 30 s chunk 的 audio token 数，再定 8/16/32。
- 对齐器可以用 vLLM pooling 一次 batch 多个 chunk；这比 VideoMiner 的串行 aligner 更有吞吐优势。

### P1：8 GB 机器上顺序执行，24 GB 上再考虑重叠

- 8 GB：ASR 完成 → 完全卸载/停进程 → aligner；不要同时驻留 4.5 GB + 1.8 GB + 两个 CUDA context。
- 24 GB：可以尝试「ASR chunk N+1」与「aligner chunk N」重叠，但要先测峰值；VideoMiner 的 Web 路径本身是顺序的，它的 540× 也不是靠重叠。
- 如果使用 vLLM，两个 `LLM` 实例会各自占 KV cache；在 8 GB 上建议用两个独立进程并显式销毁。

### P1：复刻它的「显存护栏」而不是它的二进制

- 在 vLLM 里用 `gpu_memory_utilization`（0.85–0.9）+ `max_model_len` + `max_num_seqs` 控制；不要假设 90% 是 per-process。
- 给 aligner 单独设更小的 `gpu_memory_utilization` 或独立进程，避免和 ASR 的 KV cache 抢。
- 记录 `torch.cuda.mem_get_info()` 的 used/free，像它一样在启动时打印预算。

### P2：性能优化顺序

1. 先测 vLLM BF16 的 RTFx（ASR-only 与 ASR+aligner 分开）。
2. 再开 `torch.compile`/CUDA graphs（vLLM 默认已有；HF 路径用 `mode="reduce-overhead"`）。
3. 再考虑 chunk 内 batch、prefix caching（多个 chunk 的 prompt 相同，可复用）。
4. 最后才考虑自定义 kernel；VideoMiner 的 540× 主要来自架构选择（30 s + 大 batch + paged KV + FA2），不是单个 kernel。

### P2：不要复制二进制

- 无 LICENSE、无源码、架构绑定、wrapper 还不可执行；复制到我们 repo 既不可维护也有法律风险。
- 如果只想复用 mel filters：`resources/mel_filters.bin` 是 128×201 float32，我们可以用 `WhisperFeatureExtractor(feature_size=128, n_fft=400, hop_length=160)` 生成等价表，或直接引用 repo 里的这份（它已在公开 repo 跟踪，hash 一致）。

---

## 8. 证据索引

所有原始输出在 `/tmp/videominer-inspect/`（二进制仅复制到 `/tmp`，未进任何 repo）：

| 文件 | 内容 |
|---|---|
| `01-docker-inspect.json` / `01-docker-inspect-summary.txt` | 容器 config、env、mounts、GPU |
| `02-container-basics.txt` | OS、kernel、Python、nvidia-smi、id |
| `03-app-layout.txt` | `/app` 目录与体积 |
| `04-glm-stack-tree.txt` | GLM 栈完整文件/大小 |
| `05-glm-stack-scripts.txt` | README、models/README、`bin/*` 脚本全文 |
| `06-file-ldd.txt` | `file` 缺失时的 `ldd` 输出（引擎/aligner/bridge/tvm） |
| `07-file-readelf.txt` | ELF 头、BuildID、`.comment`、NEEDED/RPATH/RUNPATH |
| `08-nm-engines.txt` | 引擎/aligner 动态符号与导入 |
| `09-asr-engine-py.txt` | `video/asr_engine.py`（daemon 生命周期、DEFAULT_ENGINE_ENV） |
| `11-cli-help.txt` | `detect_arch`、`glm-asr help`、wrapper `--help`（含 Permission denied） |
| `12-doctor.txt` | `glm-asr-doctor` 完整输出 |
| `14-pip.txt`、`23-glibc-packages.txt` | 80 个包、缺 torch/transformers/vllm |
| `19-kvpool.asm`、`20-engine-full.asm` | `kv_pool_warmup` / 全引擎反汇编 |
| `24-cuda-libs.txt` | cuBLAS/cuFFT/cuDART 版本 |
| `31-funasr-log-head.txt` | FunASR 开发机日志（5070 Ti、llama.cpp、Qwen3 0.6B Q5_K） |
| `34-sha256.txt` | 所有 ASR 产物 hash |
| `40-git-modes.txt`、`41-modes.txt` | git mode 100644 与容器 664/777 对照 |
| `refs/glmasr_config.json`、`refs/aligner_config.json` | HF config（模型结构、30 s、80 ms、5000 bins） |
| `refs/glmasr_api.json`、`refs/aligner_api.json` | HF safetensors 参数量 |
| `refs/README.md`、`refs/deployment.md`、`refs/gitignore`、`refs/Dockerfile` | 公开 repo 的 8 GB 声明、ASR 文件 gitignore、构建方式 |
| `scripts/orchestrator.py`、`scripts/glm_asr_wsr.py`、`scripts/transcription_engine.py` | 容器内编排代码（与本地 checkout 同 hash） |

关键命令（可复现）：

```bash
docker inspect videominer-inspect-20261022
docker exec videominer-inspect-20261022 bash -lc 'cat /etc/os-release; python3 -V; nvidia-smi'
docker exec videominer-inspect-20261022 bash -lc 'cd /app/asr_utils/glm-asr-stack; bash bin/glm-asr-doctor'
docker exec videominer-inspect-20261022 bash -lc 'cd /app/asr_utils/glm-asr-stack; stat -c "%A %a %n" bin/*'
docker cp videominer-inspect-20261022:/app/asr_utils/glm-asr-stack/Infer-engine/glm_asr_infer_sm86 /tmp/videominer-inspect/bin/
readelf -d /tmp/videominer-inspect/bin/glm_asr_infer_sm86
nm -C --defined-only /tmp/videominer-inspect/bin/glm_asr_infer_sm86
objdump -d -M intel --demangle --start-address=0xf7c0 --stop-address=0x10110 /tmp/videominer-inspect/bin/glm_asr_infer_sm86
strings -a /tmp/videominer-inspect/bin/fa2_bridge_sm86.so | grep -E '__cute_internal|tvm_ffi|FlashAttention2'
```

未执行的命令及原因：

- `bin/select_fa2_bridge`：会重写 `cutedsl_fa/fa2_bridge.so` 符号链接，违反只读；已用 `readlink -f` 记录当前指向 `fa2_bridge_sm120.so`，`detect_arch` 输出 `sm86`，两者不一致。
- `glm-asr transcribe` / `asr-only` / `BATCH` / `RUN`：会加载模型或推理；且模型不存在。
- 容器内 `file/readelf`：镜像未安装；已改用宿主机工具分析 `docker cp` 出来的文件。

---

## 9. 未解问题

1. **没有权重，无法测峰值显存与 RTFx。** 8 GB 的实际 batch/KV 行为、3090 上的 540× 是否成立，都只能推断。
2. **540× 的测试条件不明。** repo 只说 "consumer GPUs"、docs 提到 5070 Ti；没有 batch、GPU、音频时长、是否含对齐的 benchmark 产物。
3. **镜像里的 GLM 栈是否真的被人跑过。** wrapper 664 + 默认 sm120 symlink + 无权重 + 默认 FunASR，四个事实叠加，使「开箱即用」不成立；不排除生产环境有额外 chmod/包装脚本。
4. **`GLMASR_ENCODER_FA_VLLM=1` 的实际命中路径。** 镜像无 vLLM，预期回退 CuteDSL；但 CuteDSL bridge 默认是 sm120，3090 上可能再回退到自定义 kernel/cuBLAS；没有运行日志。
5. **`GLMASR_MAX_BATCHED_TOKENS=10000` 对应的 audio token 数。** 需要知道 GLM-ASR encoder 的时间下采样率（`concat_4frames` 与 conv stride）才能精确算一次 prefill 能容纳多少 30 s chunk；未加载模型/processor，未验证。
6. **FP8 支持程度。** `F8_E4M3` 只是 dtype 字符串；没有 FP8 GEMM/dequant 符号，实际可能直接拒绝。
7. **引擎许可证。** 镜像与公开 repo 都没有 LICENSE；GitHub API `license: null`。无法确认是否允许再分发二进制。
8. **`bin/glm_asr_infer` 的 11 个硬编码 prompt token 具体内容。** 需要 GLM tokenizer 才能解码；不影响架构结论，但影响与 vLLM prompt 的对齐。
9. **ForceAligner 的 batch 能力。** 当前 daemon 是单请求串行；vLLM pooling 可以 batch，但时间 bin 到词级的映射/平滑还需要我们实现并与原实现对比精度。

---

## 10. 许可证与来源

- **公开 repo**：`https://github.com/JazerJu/video-miner.git`，本地 checkout commit `d8b9bb46bc06bc574eab407895417fef0f21494f`（2026-09-27）。仓库根**没有 LICENSE/COPYING**；GitHub API 返回 `"license": null`，即默认保留所有权利。
- **引擎二进制**：不在公开 repo（`.gitignore` 第 92–96 行显式忽略 `Infer-engine/glm_asr_infer_sm*`、`forcealigner-engine/force_aligner_sm*`、`cutedsl_fa/fa2_bridge_sm*.so`、`lib/`）。镜像根目录只有基础 CUDA 镜像的 `/NGC-DL-CONTAINER-LICENSE`，没有应用或引擎自身的 LICENSE/NOTICE。
- **构建痕迹**：
  - RPATH `/data/fwsr/glm-asr/glm-asr-stack/lib:/usr/local/cuda-12.8/lib64`
  - bridge RUNPATH `/data/fwsr/glm-asr/GLM-ASR/venv/lib/python3.12/site-packages/tvm_ffi/lib`
  - FunASR 开发日志路径 `/media/jju/ExtraDisk/VidGo/backend/...`
  - 二进制 mtime `2026-06-14 11:20`；镜像创建 `2026-10-15`
  - 源文件名 `cuda/encoder.cu`、`cuda/decoder.cu`、`cuda/kernels.cu`、`cuda/mel.cu`、`paged_kv.cu`、`nvfp4_cutlass.cu`、`kv_pool.c`、`sequence.c`、`safetensors.c`、`audio_io.c`、`tokenizer.cpp`、`aligner.cpp`、`timestamp.cpp`
- **模型许可证**：GLM-ASR-Nano-2512 为 MIT；Qwen3-ForcedAligner-0.6B 为 Apache-2.0（HF model card tags）。
- **TVM FFI**：`libtvm_ffi.so` 是 Apache TVM 项目的 FFI ABI（上游 Apache-2.0），但镜像内没有附带其 LICENSE；CuteDSL/CUTLASS 生成代码的再分发条款也未在镜像内说明。
- **结论**：**技术上**二进制只能在 CUDA 12.8、glibc 2.39、sm86/89/120 的特定组合下工作，wrapper 还缺执行位；**法律上**没有明确授权，repo 也没有 license。**不建议把 `glm_asr_infer_sm86`、`force_aligner_sm86`、`fa2_bridge_sm86.so`、`libtvm_ffi.so` 复制进 VideoLingo 仓库或再分发。** 需要复现时，用 vLLM/transformers + 公开模型权重重写同等机制。