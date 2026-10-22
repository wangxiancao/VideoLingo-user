# ASR 引擎可行性研究：540 RTFx、RTX 3090 与自研边界（2026-10-22）

> 重建说明（2026-10-23）：原件经 2026-10-22 核实从未提交；本文从 wayfinder 地图 #33 与决策票评论重建，随 a70ede7 落盘，非原件（#43）。

> 状态：只读研究。未运行 VideoMiner 二进制、未跑推理、未下载权重、未改 `config.yaml` 或产品代码。本文只新增本文件。
>
> 标注约定：**【实测】** 本仓库已有基准且命令可复跑；**【官方】** NVIDIA/模型卡/框架文档；**【取证】** VideoMiner 容器报告的直接观察；**【源码】** 本机安装的框架源码；**【估算】** 按公式或经验推导，未跑推理；**【推断】** 由证据推导但未验证。

---

## 0. 结论摘要

1. **VideoMiner 的自研程度很高，但不是“从零训练模型”。** 模型权重与训练来自智谱 GLM-ASR 与 Qwen3-ForcedAligner（MIT / Apache-2.0）；推理引擎、CUDA kernel、连续批处理调度器、Paged KV、音频前端、BPE tokenizer、ForceAligner 引擎与 Python 编排均为自研 C++/CUDA 或自写 Python。唯一明显的第三方运行时组件是 cuBLAS/cuBLASLt/cuFFT/cuDART，以及 CuteDSL 生成、经 TVM FFI 暴露的 FlashAttention2 bridge。[S5]
2. **“540 RTFx ASR-only on RTX 3090”在 BF16/FP32 累加、30 s 分块、batch ≤128 的假设下不可达。** 按模型结构估算，1 小时音频约需 **440 TFLOP**；3090 的 BF16 稠密张量峰值是 **71.2 TFLOPS**（官方表；142.4 是 2:4 稀疏值，142.3 是 FP16/FP16 累加值），理论下限 **6.18 s/h = 582 RTFx**。540 需要 **92.8%** 的峰值利用率，还没有算 mel、tokenizer、调度和显存搬运；换精度或换 GPU 才可能。540 更可能是 5090（需 31.5% 峰值）或 5070 Ti（需 75.1% 峰值）上的数字。[S1][S2][S3][S4]
3. **建议走“vLLM 优先 → 混合自研”的两阶段路径，而不是直接写全套 C++/CUDA。** vLLM 0.24 已原生注册两个模型，能提供 Paged KV、连续批处理、chunked prefill、解码器 CUDA graphs、FlashInfer/FA2；但音频编码器是 eager、没有 CUDA graph，mel 在 CPU，aligner 的 `token_classify` 会对所有 token 做 5000 类分类。预计 vLLM-first 的 3090 ASR-only 上限约 **250–350 RTFx**，混合自研约 **350–450**；540 仍不在 3090 的 BF16 预算内。[S6][S7]
4. **最大未知不是工程，而是 540 的出处。** 取证报告里没有 GLM-ASR 的权重、benchmark 日志或可执行 GLM 路径；默认引擎是 FunASR-GGUF。540 可能来自 FunASR、5090/5070 Ti，或排除了编码器的“ASR-only”。在拿到同口径复现前，不应把 540 当作 3090 的验收线。[S5]

---

## 1. 证据边界与现有基线

### 1.1 已有实测

本仓库的基线是 transformers + SDPA，不是 VideoMiner 引擎，也不是 vLLM。[S12]

| 口径 | 结果 | 说明 |
|---|---:|---|
| V600 真实视频，600 s，热，batch 20 | **106.66 RTFx** | ASR + aligner；ASR 3.92 s，aligner 1.14 s |
| V600 冷链路 ingest→transcribe | **35.58 RTFx** | 含进程启动、模型加载 4–10 s |
| V600 batch 20 显存 | 10,294 / 13,284 MB | allocated / reserved；GPU 另有约 9.5 GB 占用 |
| ASR-only（由实测拆分） | **约 153 RTFx**（600/3.92） | 派生值，不是文档原表 |
| 合成 S600，batch 20 | 147.95 RTFx | 与真实视频同量级 |

这些数字说明：当前 HF 路径的有效算力约 **18.7 TFLOPS**（600 s 音频约 73.4 TFLOP / 3.92 s），只有 3090 BF16 峰值的 **26%**。从 153 到 540 是 3.5 倍，不是靠“把 batch 调大”能补的。**本报告没有在 3090 上测出新的 RTFx；下文所有 3090/vLLM 数字均为估算，必须用 §7 的测量替换。**

### 1.2 模型结构事实

| 项 | GLM-ASR-Nano-2512 | Qwen3-ForcedAligner-0.6B |
|---|---|---|
| 参数 | 2.26B（编码器 0.63B，解码器 1.35B，embed/lm_head 0.24B） | 0.92B（编码器 0.30B，解码器 0.44B，其余为 embedding/分类头/卷积） |
| 音频编码器 | 32 层 / 1280d / 20 heads，conv1 stride 1、conv2 stride 2 | 24 层 / 1024d / 16 heads，3×Conv2d stride 2 |
| 30 s token 数 | 3000 mel 帧 → 1500 编码 token → 4 帧合并 → **375 解码 token** | 3000 帧 → 3 次下采样 → **375 音频 token** |
| 文本解码器 | 28 层 / 2048d / 16 Q / 4 KV / head_dim 128 / vocab 59264 | 28 层 / 1024d / 16 Q / 8 KV / vocab 152064 |
| 输出 | 自回归贪心，`max_new_tokens` 256（VideoMiner）/512（本仓库） | 一次 forward，5000 类时间 bin，80 ms/bin |
| 权重 | 4.30 GiB BF16，MIT | 1.75 GiB BF16，Apache-2.0 |

来源：本机缓存 `config.json` / `processor_config.json`、模型卡、取证报告。[S3][S4][S5][S14]

关键点：**30 s 音频进入解码器只有 375 个 token**（4 帧合并），不是 1500。这一点决定了 prefill 不算大，但编码器的 1500 token × 32 层是全量计算，是主要算力。

---

## 2. Q1：VideoMiner 的自研程度

### 2.1 组件矩阵

| 层 | 组成 | 归属 | 证据 | 可复用性 |
|---|---|---|---|---|
| 模型与训练 | GLM-ASR、Qwen3-ForcedAligner 权重 | **第三方官方**（智谱/Qwen），非 VideoMiner 训练 | HF 模型卡、权重 hash、MIT/Apache-2.0 | 可直接用权重 |
| 推理引擎 | C++17/CUDA 12.8 PIE，296 个本地符号，只依赖 cuBLAS/cuBLASLt/cuFFT/cuDART | **自研** | 无 torch/TRT/ORT/TVM runtime；RPATH 泄漏构建机 | 无源码、无 license，不可复用 |
| GEMM | cuBLAS / cuBLASLt | 第三方（NVIDIA） | `cublasGemmEx`、`cublasLtMatmul` | 通过 CUDA 依赖使用 |
| 编码器注意力 | 自定义 tile kernel + 可选 CuteDSL FA2 bridge | **自研 kernel + 第三方生成 kernel** | `encoder_attention_flash_fa2_launch`、`fa2_bridge_sm*.so` | bridge 无 license、架构绑定，不可复制；可自己用 FA2/CUTLASS |
| 解码器 | 自定义 paged decode、GQA、RoPE、RMSNorm、SiLU、KV pool | **自研** | `paged_decode_attention*`、`reshape_and_cache_kernel`、`kv_pool.c` | 不可复用；vLLM 有等价实现 |
| 调度器 | 连续批处理、preempt、token 预算 | **自研** | `scheduler_*`、`GLMASR_MAX_SEQS=256`、`GLMASR_MAX_BATCHED_TOKENS=10000` | 不可复用；vLLM 有等价实现 |
| Paged KV | block 128、按剩余显存分配、800 MiB headroom | **自研** | `kv_pool_warmup` 反汇编；56 KiB/token | 不可复用；vLLM 有等价实现 |
| FA2 bridge | CuteDSL 生成 FlashAttention2，TVM FFI ABI | **第三方生成 + 自研 bridge/选择** | `__cute_internal_fa2_enc_varlen_fwd`、`libtvm_ffi.so` | 生成物无明确再分发条款；建议自己构建 |
| ForceAligner | 独立 C++/CUDA 引擎；非自回归，一次 forward 出 5000 类 logits，LIS 单调修正 | **自研引擎 + 官方权重** | `aligner_align`、`extract_timestamps`、`fix_timestamps_lis` | 不可复用；算法可独立实现 |
| 音频前端 | ffmpeg 子进程 + drwav + 自定义 mel + cuFFT | **自研 + 第三方 cuFFT** | `mel_kernel`、`cufftPlanMany`、`ffmpeg -f f32le` | 不可复用；mel 表可重建 |
| Tokenizer | 自写 C++ BPE（vocab.json + merges.txt） | **自研** | `tokenizer.cpp`；orchestrator 用 Python `tokenizers` | 不可复用；HF tokenizer 可替代 |
| 编排 | `orchestrator.py`、`glm_asr_wsr.py`、`asr_engine.py` | **自研 Python** | daemon 协议、30 s 分块、batch 128/32 | 无 license，不可复制；可重写 |

### 2.2 缺失的源码/二进制与许可

**缺失：** `cuda/encoder.cu`、`cuda/decoder.cu`、`cuda/kernels.cu`、`cuda/mel.cu`、`paged_kv.cu`、`nvfp4_cutlass.cu`、`kv_pool.c`、`sequence.c`、`safetensors.c`、`audio_io.c`、`tokenizer.cpp`、`aligner.cpp`、`timestamp.cpp` 全部不在镜像；`scripts/build_infer_arch.sh` / `build_forcealigner_arch.sh` 也未跟踪；权重和 benchmark 日志都不在镜像。[S5]

**存在但不可用：** sm86/sm89/sm120 三份二进制、`fa2_bridge_sm*.so`、`libtvm_ffi.so`、`mel_filters.bin`；`bin/` 下 6 个脚本权限 664 无执行位，默认 symlink 指向 sm120，而本机是 sm86。[S5]

**许可：** VideoMiner 公开 repo 根目录没有 LICENSE/COPYING，GitHub API `license: null`，即默认保留所有权利；镜像内也没有应用或引擎的 LICENSE/NOTICE。**不能复制其源码、二进制、bridge、mel 文件或编排代码。** 模型权重是 MIT/Apache-2.0，可用；CUDA/cuBLAS/cuFFT 按 NVIDIA EULA 作为依赖使用；CUTLASS 是 BSD-3，FlashAttention 是 BSD-3，FlashInfer 是 Apache-2.0，TVM 是 Apache-2.0，vLLM 是 Apache-2.0，这些可以合法依赖或改写。[S8][S9][S10][S11]

**边界声明：** 取证只能证明二进制形态、符号和配置，不能证明源码来源。本文**不声称** VideoMiner 抄自 vLLM，也**不声称**我们可以复用它的任何代码；只把它当作“同架构独立实现的性能存在性证据”。

---

## 3. Q2：要构建什么才能冲 540（以及为什么 3090 上冲不到）

### 3.1 先看屋顶线，再看组件

在 BF16/FP32 累加、30 s 分块、0 overlap、batch 128、每块约 120 个输出 token 的假设下，1 小时音频的算力账如下（【估算】，公式见 §4）：

| 阶段 | 每 30 s 块（TFLOP） | 每小时（TFLOP） | 备注 |
|---|---:|---:|---|
| 编码器线性层 | 1.887 | 226.5 | 32 层 × 1500 token |
| 编码器注意力 | 0.369 | 44.2 | 1500² 全注意力 |
| 4 帧合并 + projector | 0.022 | 2.6 | 375 token |
| 解码器 prefill | 1.067 | 128.0 | 375 音频 + ~20 prompt |
| 解码器 decode | 0.324 | 38.9 | ~120 步，实际受显存带宽约束 |
| **合计** | **3.669** | **440.3** | 不含 mel/tokenizer/调度 |

3090 的 BF16 稠密峰值是 **71.2 TFLOPS**（FP32 累加；官方表 71/142 中 142 是 2:4 稀疏）。因此纯计算下限是 `440.3 / 71.2 = 6.18 s/h`，即 **582 RTFx 是 100% 利用率的理论上限**。540 对应 6.667 s，需要 **92.8%** 的峰值利用率；任何真实 kernel 都达不到。

| 峰值利用率 | 1 h 用时（s） | ASR-only RTFx | 说明 |
|---:|---:|---:|---|
| 30% | 20.6 | 175 | 当前 HF 路径约 26% |
| 40% | 15.5 | 233 | vLLM 保守 |
| 50% | 12.4 | 291 | vLLM 调优后 |
| 60% | 10.3 | 350 | 混合自研 |
| 70% | 8.8 | 408 | 很好的自定义引擎 |
| 80% | 7.7 | 466 | 接近极限 |
| 90% | 6.9 | 524 | 仍低于 540 |
| 100% | 6.2 | 582 | 绝对上限 |

这不是“再优化一点”的差距，而是**目标本身贴在屋顶线上**。在 3090 上，540 不是工程问题，是物理问题；除非改变精度或 GPU。

### 3.2 组件清单：哪些能委托，哪些必须自研

| 组件 | 建议实现 | 归属 | 说明 |
|---|---|---|---|
| 30 s 分块、0 overlap、尾块合并 | 自写编排 | 必须自研（简单） | 直接复用本仓库 `_chunks` 语义 [S13] |
| ffmpeg 解码/重采样 16 kHz mono | 委托 ffmpeg | 第三方 | 不是瓶颈 |
| log-mel 128×3000 | GPU `torch.stft`/cuFFT，或 HF CPU | 可委托，但 GPU 版需自研/封装 | vLLM 现在在 CPU 做 mel，120 块串行是可见成本 [S6] |
| 编码器 GEMM（QKV/O/MLP） | cuBLAS/CUTLASS | 委托 | 大 batch 下效率高 |
| 编码器注意力 | FlashAttention2 / FlashInfer | 委托 | 1500 token 全注意力 |
| 编码器融合与 CUDA graph | 自研 `torch.compile`/CUDA graph/融合 kernel | **必须自研** | vLLM 的音频编码器是 eager，默认不捕获 graph [S6] |
| 4 帧合并 + projector | vLLM 已有 reshape+MLP；峰值需融合 | 可委托/需自研融合 | 375 token，占比小 |
| 解码器 Paged KV + 连续批处理 | vLLM | 委托 | 最值得复用 |
| 解码器 CUDA graphs | vLLM | 委托 | 解码器默认支持 |
| 贪心采样、EOS、max_new_tokens | vLLM | 委托 | 与 VideoMiner 一致 |
| Tokenizer / detokenizer | HF `tokenizers` | 委托 | Rust 实现，够快 |
| ForceAligner 编码器/解码器 | vLLM pooling，或自研 | 可委托 | 一次 forward；VideoMiner 是单请求串行，vLLM 可 batch 多块 |
| 5000 类时间 bin 分类 | 只对 `<timestamp>` 位置做 | **必须自研** | vLLM 现在对全部 token 做，浪费 4–5 倍 |
| 时间戳解码、LIS、词级映射 | 自写 | **必须自研** | 契约与质量的核心 |
| 常驻 daemon、VRAM 预算、OOM 回退 | 自写 | **必须自研** | 冷启动与稳定性 |
| 评测/质量 harness | 自写 | **必须自研** | CER、时间戳、可复现 |

### 3.3 如果一定要在 3090 上冲 540，只有三条路

1. **FP16 权重 + FP16 累加**：3090 的 FP16/FP16 累加稠密峰值是 142.3 TFLOPS，540 只需 46.4%。但官方权重是 BF16，转 FP16 有溢出/精度风险，必须用 CER 验证；这不是“免费加速”。
2. **INT8 量化**：3090 INT8 峰值 284.7 TOPS，540 只需 23.2%。但 GLM-ASR 没有官方 INT8 checkpoint，vLLM 也没有该架构的 INT8 路径；需要自研量化 kernel 与校准，质量风险最高。
3. **换 GPU**：5090 的 BF16 稠密峰值 209.5 TFLOPS，540 只需 31.5%；5070 Ti 是 87.9 TFLOPS，需 75.1%。如果 540 是硬指标，最便宜的做法是换 5090，而不是在 3090 上写引擎。

FP8 不在选项内：3090 是 sm86，没有 FP8 tensor core。VideoMiner 的 NVFP4 路径也不能当作捷径：取证显示 `GLMASR_USE_CUTLASS_NVFP4` 的 native 路径未启用、回退到 BF16 dequant；3090 本身也没有原生 FP4 tensor core。[S5]

---

## 4. Q3：RTX 3090 屋顶线

### 4.1 官方规格

| GPU | 显存 | 带宽 | FP32 | BF16 稠密（FP32 累加） | FP16 稠密（FP16 累加） | 来源 |
|---|---:|---:|---:|---:|---:|---|
| RTX 3090 | 24 GB GDDR6X | 936 GB/s | 35.6 TFLOPS | **71.2 TFLOPS**（稀疏 142.4） | 142.3 TFLOPS | [S1] |
| RTX 5070 Ti | 16 GB GDDR7 | 896 GB/s | 43.9 TFLOPS | **87.9 TFLOPS**（稀疏 175.8） | 175.8 TFLOPS | [S2] |
| RTX 5090 | 32 GB GDDR7 | 1792 GB/s | 104.8 TFLOPS | **209.5 TFLOPS**（稀疏 419） | 419 TFLOPS | [S2] |

注意：很多二手资料把 3090 的 142 当作 BF16 稠密值；官方表写得很清楚，**BF16/FP32 累加是 71.2，142.4 是 2:4 稀疏（脚注 2 = with Sparsity）**。官方权重不是 2:4 稀疏的，不能直接吃这 2 倍。取证只能确认 VideoMiner 使用 BF16 权重和 `cublasGemmEx`；若按 cuBLAS 常见的 FP32 累加语义，应按 71.2 算。若它实际走了 FP16 累加，540 的算术才可能成立，但这需要运行时证据，不能从符号推出。

### 4.2 分阶段瓶颈

- **编码器是主瓶颈，不是解码器。** 编码器线性层 + 注意力约 270.7 TFLOP/h，占总量的 61.5%；解码器 prefill 128 TFLOP/h（29%），decode 38.9 TFLOP/h（8.8%，实际是带宽约束）。即使解码器完美，编码器在 60% 效率下也要 6.3 s/h，已经接近 540 的 6.67 s 总预算。
- **解码器 decode 是带宽约束，不是算力约束。** 每步要读约 2.7 GB BF16 权重；batch 128 时这一步被 128 个序列分摊，1 小时 120 步约 `120 × 2.7 GB / 936 GB/s ≈ 0.35 s`（忽略 KV）。所以 decode 不是 540 的主要障碍；batch 越大越 amortize。
- **ForceAligner 是“第二引擎”，不是零成本。** 按 24 层/1024d 编码器 + 28 层/1024d 解码器 + 5000 类头估算，约 0.84 TFLOP/块（含全 token 分类约 6.9 GFLOP），约 101 TFLOP/h。当前实测 aligner 1.14 s/600 s = **6.84 s/h**，对应约 **21%** 的峰值效率。VideoMiner 的 aligner daemon 是单请求串行；vLLM pooling 可以一次 batch 多个 chunk，这是我们相对它的潜在优势。即使优化到 1–2 s/h，ASR+aligner 的 540 也比 ASR-only 更难（约 415–470 RTFx）；540 只能是 ASR-only 口径。
- **调度不是算力瓶颈，但会毁掉 batch。** vLLM 默认 `max_num_batched_tokens=2048`、`max_num_seqs=128`；每块 375 个解码 token，默认一步只排 ~5 块，编码器 batch 只有 5，GEMM 效率骤降。要跑到 batch 128，必须把 token 预算提到 ~48k，或自研按块调度。[S6]

### 4.3 显存账

| 项 | 估算 | 说明 |
|---|---:|---|
| ASR 权重 | 4.30 GiB | BF16 |
| Aligner 权重 | 1.75 GiB | BF16 |
| ASR KV / token | 56 KiB | 28 层 × 4 KV × 128 × 2 × 2 B |
| 每块 631 token 的 KV | 34.5 MiB | 375 音频 + 256 生成 |
| batch 128 ASR KV | 4.31 GiB | 128 × 34.5 MiB |
| batch 128 编码器激活 | 5–8 GiB | 1500×1280×2 B×128，含 QKV/MLP 中间量；取决于融合 |
| ASR b128 合计 | 13.6–16.6 GiB | 24 GB 可容纳；8 GB 不行 |
| Aligner b128 KV | 9.2 GiB | 112 KiB/token × 675 × 128 |
| ASR + aligner 同时 b128 | 24.5–27.5 GiB | **放不下 24 GB，必须顺序执行或降 aligner batch** |

结论：24 GB 不是 540 的阻碍；算力才是。8 GB 只能跑很小的 batch，报告里“8 GB 可跑 ASR-only”不等于“8 GB 能跑 540”。[S5]

### 4.4 540 是否物理可行

**在 3090 + BF16/FP32 累加下：在本文假设下不可行。** 540 需要 92.8% 的稠密峰值；理论绝对上限是 582 RTFx，且这个上限还没扣 mel、tokenizer、调度、显存搬运和 kernel 启动。即使把 decode 完全去掉，编码器 + prefill 也要 5.64 s/h，540 仍需 84.5% 峰值。真实引擎能到 70–80% 已经是极好的手写实现（408–466 RTFx）。这与取证报告“540 大概率是 5070 Ti/5090 条件”一致：5070 Ti 需 75%，5090 只需 31.5%。[S5] 取证还显示该引擎没有 `cudaGraph*`、单流；这使 3090/5070 Ti 达到 75–93% 峰值更不现实，进一步指向 5090。5070 Ti 只有 16 GB，batch 128 的 13.6–16.6 GiB 加 aligner 也放不下，实际 batch 会更低；5090 的 32 GB 才是同时满足 batch 与 540 算力的配置。

**在 3090 + FP16/FP16 累加下：算术上可行（46.4%），但质量与工程风险高。** 需要把 BF16 权重转 FP16、验证不溢出、重跑 CER；这是一条独立的研究线，不应默认成立。

**在 3090 + INT8 下：算术上最宽裕（23.2%），但离现有代码最远。** 需要量化校准、自定义 GEMM、质量验证；不建议作为第一阶段目标。

---

## 5. Q4：vLLM 0.24 的天花板

### 5.1 已经提供什么

本机 vLLM `0.24.0+cu129` 已注册 `GlmAsrForConditionalGeneration`（transcription）和 `Qwen3ASRForcedAlignerForTokenClassification`（pooling，`token_classify`）。[S6]

| 能力 | 状态 | 对本任务的意义 |
|---|---|---|
| PagedAttention / Paged KV | 有（v1） | 复现 VideoMiner 的核心显存机制 |
| 连续批处理 | 有 | 自动按 token/block 限流，替代固定 batch 128 |
| Chunked prefill | 有 | 长 prompt/多块 prefill 可切片 |
| Prefix caching | 有 | 对 30 s 块 + 固定 prompt 有少量收益；真实音频块内容不同，编码器缓存按 `mm_hash`，跨请求复用有限 |
| 解码器 CUDA graphs | 有 | 解码器的主要加速 |
| FlashAttention / FlashInfer / Triton | 有 | 本机装了 FlashInfer 0.6.12；sm86 可用 |
| 音频编码器 | 有，但 **eager** | `embed_multimodal` 在 `gpu_model_runner` 里单独调用，不在编译后的语言模型 forward 内 |
| 音频编码器 CUDA graph | **不支持 GLM-ASR** | `cudagraph_mm_encoder` 默认 False；GLM-ASR 没有实现 `SupportsEncoderCudaGraph` [S6] |
| mel 前端 | CPU `WhisperFeatureExtractor` + `torch.stft` | 没有 GPU mel；120 块串行是可见成本 |
| Aligner `token_classify` | 有，但对 **所有 token** 做 5000 类分类 | 输出巨大的 logits；没有时间戳解码/LIS |
| 转写 prompt | 固定 `"can you transcribe the speech into a written format?"` | GLM-ASR 的 `get_generation_prompt` 忽略 hotwords/request_prompt；与模型卡 prompt 不同，质量需对比 |
| 量化 | BF16/FP16；3090 无 FP8 | 没有官方 INT8/AWQ 权重，不能靠量化白拿 2× |

### 5.2 可能慢/缺失的地方

1. **编码器 eager + 未融合。** 32 层 × 每层 QKV、RoPE、Attention、O、LayerNorm、GELU、MLP，约 300 个 kernel/批；没有 CUDA graph，Python/launch 开销和中间张量读写都在。编码器又是 61.5% 的算力，这是 vLLM-first 与手写引擎最大的差距。
2. **mel 在 CPU。** HF 的 `_torch_extract_fbank_features` 用 CPU `torch.stft`；30 s 块约几十毫秒，120 块串行可能 1–3 s/h。需要多进程前端或 GPU mel 才能摊掉。
3. **Aligner 的分类头浪费。** `TokenClassifierPoolerHead` 对 `pooling=ALL` 的每个 token 做 `1024×5000` 分类；675 token/块 × 5000 × 2 B ≈ 6.75 MB logits/块，1 小时约 810 MB 传输；只对 135 个 `<timestamp>` 位置分类可省约 5 倍算力和约 5 倍输出。
4. **调度默认值不适合 30 s 块。** 2048 token 预算 / 128 序列，每块 375 token，默认只能排 ~5 块；必须显式把 `max_num_batched_tokens` 提到 ~48k、`max_num_seqs` 提到 128，否则编码器 batch 很小、GEMM 效率低。
5. **没有 30 s / 0 overlap 分块契约。** `SpeechToTextConfig` 的通用默认是 `max_audio_clip_s=30`、`overlap_chunk_second=1`；GLM-ASR 的 `get_speech_to_text_config` 又把 `max_audio_clip_s` 覆写为 processor 的 `max_audio_len`（655 s），长音频会走能量切分和 1 s 重叠 [S6]。我们必须自己切 30 s、0 overlap 的块再喂，否则时间戳和边界语义会变。
6. **两个模型两个实例。** ASR 与 aligner 各自占 KV/权重；24 GB 上同时跑 batch 128 放不下（§4.3），需要顺序执行或共享进程/降 batch。

### 5.3 估算上限

【估算】按编码器 40–55%、解码器 45–60% 的混合效率，vLLM-first 的 3090 ASR-only 热 RTFx 约 **250–350**；加上 aligner 约 **200–300**；冷启动仍受 4–10 s 加载限制。它能把当前 153 提升约 1.6–2.3 倍，但离 540 仍有 1.5–2 倍缺口。这个数字必须用 §7 的测量替换，不能当承诺。

---

## 6. Q5：架构选项

### 6.1 A/B/C 对比

| 维度 | A：vLLM-first 服务 | B：vLLM + 自研 kernel（推荐） | C：全自研 C++/CUDA |
|---|---|---|---|
| 组成 | vLLM 离线 `LLM`/`AsyncLLM` 进程 + 自写编排/对齐后处理 | vLLM 解码器/Paged KV/调度 + 自研 GPU mel、编码器融合/graph、projector、aligner 头 | 自写引擎、kernel、调度、Paged KV、mel、tokenizer、aligner |
| 工程量（1 名资深工程师） | MVP 2–3 周；生产 6–10 周 | 3–6 个月；生产再加 2–3 个月 | 12–24 个月（2–3 人），生产级更久 |
| 主要风险 | 性能不达标；vLLM 版本/接口变动 | CUDA/Triton 人才；vLLM 内部耦合；回归 | 正确性、tokenizer/音频/量化、架构移植、长期维护 |
| 3090 ASR-only 上限（估算） | 250–350 | 350–450（FP16 累加可到 ~500） | 400–500；540 仍需 93% BF16 或换精度 |
| 维护成本 | 低 | 中 | 高 |
| 许可 | Apache-2.0 | Apache-2.0 + BSD-3 | 自研代码可闭源；依赖 CUTLASS/FA2/FlashInfer 的 BSD/Apache |
| 与“自研”的关系 | 产品运行时/契约自研，推理内核依赖 vLLM | 运行时与热点 kernel 自研，vLLM 作为内部库 | 完全自研 |

### 6.2 推荐路径与决策门

**推荐 B，但从 A 开始。** 理由：

- 540 在 3090 的 BF16 预算内不可达；C 的 12–24 个月投入最多只能把 350–450 抬到 400–500，边际收益远小于成本。
- A 能在 2–3 周内给出“vLLM 到底能到多少”的硬数据；如果 A 已经 300+，B 的优化目标明确（编码器、mel、aligner 头）。
- B 保留“自研”的实质：分块契约、编码器融合、aligner 时间戳、daemon、显存策略都由我们拥有；vLLM 只是解码器/Paged KV 的内部实现。将来若必须去掉 vLLM，B 的接口已经把替换面收敛到解码器。
- C 只在“B 实测 < 350 且 540 被确认为 3090 硬指标”时启动；即使启动，也要先接受“540 可能仍达不到”。

这里的“自研”指我们拥有运行时、阶段契约、调度策略、显存策略和热点 kernel，而不是禁用所有第三方库。若管理层把“自研”定义为连 vLLM 都不能依赖，则只剩 C；此时应同时接受 3090 上 400–500 的上限、12–24 个月工期，或者把 540 的 GPU 条件改为 5090。

**阶段与门：**

| 阶段 | 内容 | 退出标准 | 门 |
|---|---|---|---|
| S0（1 周） | 定义 540：GPU、ASR-only、热/冷、batch、质量线；确认出处 | 一页 target contract | **G0 人类签字** |
| S1（2–3 周） | vLLM smoke：两模型加载、30 s 块、时间戳合理 | 输出契约与现有 `Transcription` 一致 | G1 |
| S2（2 周） | vLLM 上限：600 s/1 h，batch 16/32/64/128，拆 encoder/decoder/mel/aligner | 热 RTFx、显存、profile | G2：≥300 进 B；<250 先查瓶颈 |
| S3（6–10 周） | 混合优化：GPU mel、编码器 `torch.compile`/CUDA graph/融合、projector、aligner 只分类时间戳 | 编码器效率 ≥55%；ASR-only ≥400 | G3：<350 则重定目标或评估 C |
| S4（4–6 周） | 质量与生产化：CER、时间戳、daemon、OOM、并发 | 质量不降、冷启动可接受 | G4 |
| S5 | Spec v0.5 | 决策记录 + 测量证据 | G5 |

---

## 7. Q6：失效条件与先验测量

### 7.1 什么结果会让 540 不可能/不值得

**不可能：**

- BF16/FP32 累加的屋顶线要求 >90% 峰值（本文算得 92.8%）。
- 540 的出处是 5090/5070 Ti、或 FunASR 引擎、或排除了编码器；不是 3090 + GLM-ASR 的同口径。
- 编码器在最好的融合/graph 下仍 <55% 峰值；编码器占 61.5%，它上不去，总量就上不去。
- 质量要求锁死 BF16，不允许 FP16/INT8；那 3090 没有第二条路。
- 24 GB 被其他进程占用（当前机器就有 9.5 GB 被 infinity_emb/mineru 占用），batch 128 的 14–17 GB 需求很可能无法满足；必须先清卡。

**不值得：**

- 从 450 到 540 可能需要数倍工程投入，而产品端到端瓶颈在翻译/LLM/烧录，不在 ASR。
- 540 是热启动 ASR-only；冷启动仍要 4–10 s 加载，短视频体验由加载主导。
- ASR+aligner 即使 ASR 540，aligner 1–2 s/h 也会把端到端拉到约 415–470；若 aligner 仍是 6.8 s/h，端到端只有 ~270。
- 自研引擎的长期维护（CUDA 版本、新架构、模型更新）会持续吞噬收益。

### 7.2 先做什么原型

1. **vLLM smoke（1 周）**：30 s 块，两个模型，确认文本、词级时间戳、`token_classify` 切片方式；不追性能。
2. **vLLM 上限（2 周）**：600 s + 1 h，batch 16/32/64/128，记录热/冷、显存、encoder/decoder/mel/aligner 分解；这是所有后续决策的分母。
3. **编码器微基准（1 周）**：batch 128、1500 token、32 层，单独测 TFLOPS；对比 71.2 峰值，判断“编码器到底能到 55% 还是 75%”。
4. **mel 前端（3 天）**：CPU vs GPU `torch.stft`，120 块串行/并行耗时；确认是否要自研 GPU mel。
5. **aligner 头（1 周）**：vLLM 全 token 分类 vs 只取 `<timestamp>` 位置；测时间和显存。
6. **混合编码器（2–4 周）**：`torch.compile`/CUDA graph/融合 QKV+RoPE+LayerNorm+GELU；只有在第 3 步显示编码器是瓶颈时才做。

### 7.3 必须测的矩阵

| 维度 | 取值 | 目的 |
|---|---|---|
| 口径 | ASR-only / ASR+aligner | 540 到底指哪个 |
| 温度 | 冷（含加载）/ 热（warmup 后） | 产品体验 vs 吞吐 |
| 音频 | 600 s / 3600 s | 摊销 decode 与加载 |
| batch | 16 / 32 / 64 / 128 | 找 GEMM 与 KV 的拐点 |
| 分块 | 30 s（固定） | 与模型窗口一致 |
| 显存 | alloc / reserved / 整卡 used | 24 GB 是否真够 |
| 分解 | encoder / decoder prefill / decode / mel / aligner | 找瓶颈 |
| 质量 | CER（zh/en 固定集）、时间戳单调/覆盖 | 速度不能换质量 |
| 环境 | GPU 时钟、功耗、其他进程 | 当前机器有 9.5 GB 占用，必须记录 |

所有数字取 3 次中位数，报告方差；不要把单次最好值当结论。

---

## 8. Q7：决策票结构（wayfinder map，仅提案，不建 issue）

在 Spec v0.5 之前，用一张 map + 子票把“目标是否成立”与“引擎怎么做”分开。建议结构：

| ID | 类型 | 问题 | 产出/证据 | 退出标准 | 依赖 |
|---|---|---|---|---|---|
| D0 | decision（人类） | 540 的精确定义：GPU、ASR-only、热/冷、batch、质量线、1 h 还是 600 s | 一页 target contract | 签字 | — |
| D1 | research | VideoMiner 自研/许可边界 | 本报告 §2 + 取证报告 | 确认不可复用 | — |
| D2 | research | 3090 屋顶线与 540 可达性 | 本报告 §3–4 | 接受或带数字反驳 | D0 |
| D3 | prototype | vLLM 两模型 smoke | 30 s 文本/时间戳 | 契约一致 | D0 |
| D4 | measurement | vLLM 3090 上限 | 热/冷 RTFx、显存、batch 曲线 | 可复跑证据 | D3 |
| D5 | measurement | 瓶颈分解 | encoder/decoder/mel/aligner 占比 | 编码器效率数字 | D4 |
| D6 | prototype | aligner 只分类时间戳 vs vLLM 全 token | 时间/显存/质量 | 选定实现 | D3 |
| D7 | decision（人类） | A/B/C 架构选择 | D4–D6 数据 | 签字 | D5, D6 |
| D8 | decision（人类） | 目标重定：3090 350–450，或 5090 540，或 FP16/INT8 | 成本/风险对比 | 签字 | D2, D4 |
| D9 | task | 质量 parity harness | CER/时间戳报告 | 质量不降 | D7 |
| D10 | task | 常驻 daemon/VRAM/OOM/并发设计 | 运行契约 | 24 GB 稳定 | D7 |
| D11 | decision | Spec v0.5 go/no-go | 全部证据 | 签字 | D8–D10 |

Map 的顺序应是 **D0 → D2 → D3 → D4 → D5 → D7 → D8 → D9/D10 → D11**；D1 已由取证报告完成，D6 可与 D5 并行。D0/D2/D7/D8/D11 是 human gate，不能由 agent 自行关闭。不要在 D4 之前开实现票；也不要在 D8 之前写 Spec v0.5。

---

## 9. 来源表

| ID | 来源 | 类型 | 用途 |
|---|---|---|---|
| S1 | NVIDIA, *NVIDIA Ampere GA102 GPU Architecture* whitepaper v2.1, <https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf> | 官方规格 | 3090：24 GB、936 GB/s、BF16 71.2/142.4、FP16 142.3 |
| S2 | NVIDIA, *NVIDIA RTX Blackwell GPU Architecture* whitepaper, <https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf> | 官方规格 | 5090/5070 Ti：带宽、显存、BF16 209.5/87.9 |
| S3 | `zai-org/GLM-ASR-Nano-2512` 模型卡 + 本机 `config.json`/`processor_config.json`，<https://huggingface.co/zai-org/GLM-ASR-Nano-2512> | 官方模型卡/配置 | 2.26B、32/28 层、30 s、1500→375 token、MIT |
| S4 | `Qwen/Qwen3-ForcedAligner-0.6B` 模型卡 + 本机 config，<https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B> | 官方模型卡/配置 | 0.92B、24/28 层、5000 bin/80 ms、Apache-2.0 |
| S5 | `/root/VideoLingo-refactor/docs/agents/videominer-fast-asr-report.md` | 容器取证 | 引擎组成、batch、Paged KV、540 算术、许可 |
| S6 | 本机 vLLM 0.24.0 源码：`model_executor/models/{glmasr,qwen3_asr_forced_aligner}.py`、`model_executor/layers/attention/mm_encoder_attention.py`、`v1/worker/{gpu_model_runner,encoder_cudagraph}.py`、`config/scheduler.py`、`model_executor/layers/pooler/tokwise/*`、`config/speech_to_text.py` | 框架源码 | 注册、encoder eager、pooling、默认调度、prompt |
| S7 | vLLM 文档：PagedAttention、CUDA Graphs、Pooling Models、Supported Models，<https://docs.vllm.ai/en/latest/> | 官方文档 | Paged KV、CUDA graph、token_classify |
| S8 | NVIDIA CUTLASS，BSD-3-Clause，<https://github.com/NVIDIA/cutlass> | 第三方代码/许可 | 选项 B/C 的 GEMM/kernel 基础 |
| S9 | Dao-AILab FlashAttention，BSD-3-Clause，<https://github.com/Dao-AILab/flash-attention> | 第三方代码/许可 | 编码器/解码器注意力 |
| S10 | FlashInfer，Apache-2.0，<https://github.com/flashinfer-ai/flashinfer> | 第三方代码/许可 | vLLM 的 attention backend |
| S11 | Apache TVM，Apache-2.0，<https://github.com/apache/tvm> | 第三方代码/许可 | VideoMiner 的 TVM FFI ABI；不建议依赖 |
| S12 | `/root/VideoLingo-refactor/docs/agents/rtfx-baseline.md`、`rtfx-optimization.md` | 本仓库实测 | 106.66/35.58 RTFx、10.3/13.2 GB、ASR/aligner 拆分 |
| S13 | `/root/VideoLingo-refactor/pipeline/asr/glm.py`、`pipeline/asr/contracts.py` | 本仓库源码 | 现有 30 s 分块、HF generate、`Transcription` 契约 |
| S14 | `/root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512`、`/root/.cache/modelscope/Qwen/Qwen3-ForcedAligner-0___6B-hf` | 本机权重 | 4.3 GB / 1.8 GB safetensors |

---

## 10. 最终建议（一句话版）

**把“540 on 3090”改写成“3090 上 350–450 ASR-only，或 5090 上 540”，然后用 2–3 周 vLLM 实测决定是否进入混合自研；不要直接开全套 C++/CUDA 引擎，也不要把 VideoMiner 的二进制当作可复用资产。**