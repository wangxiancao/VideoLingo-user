# FP16 精度与 ASR-only 540 RTFx 探针（Decision #34）

> 状态：只读研究探针。未使用 vLLM，未下载模型，未改 `config.yaml` 或任何跟踪代码；临时脚本在 `/tmp/fp16-probe/`，原始证据在 `bench/results/fp16-probe/`，未提交、未推送。
>
> 日期：2026-10-22。GPU：RTX 3090 24 GB（驱动 570.144，共享环境，约 9.6–9.9 GB 被外部进程占用）。venv：`/root/VideoLingo/.venv`（torch 2.11.0+cu129，transformers 5.14.1，CUDA 12.9）。

## 0. 结论摘要

1. **FP16 权重本身的质量可行。** 30 s、60 s、600 s（中文 R600）、600 s（英文 V600）、1200 s（中文，batch 40）上，FP16 与 BF16 的文本 CER 为 0%–0.19%，词级时间戳平均偏差 < 0.7 ms、p95 为 0、最大 0.16–0.24 s；无 NaN/Inf、无空文本、无 512-token 截断、无重复尾部。BF16 是对照，不是 ground truth。
2. **但 FP16 权重在 PyTorch/cuBLAS 下不加速。** `torch.matmul` 的 FP16 GEMM 实际使用 `CUBLAS_COMPUTE_32F`（cuBLAS 日志确认），与 BF16 一样受 71.2 TFLOPS 的 FP32-累加峰值约束。600 s 中文 b20：FP16 **149.27** RTFx vs BF16 **150.57**（FP16 慢 0.9%）；1200 s b40：FP16 **210.15** vs BF16 **212.84**（慢 1.3%）；英文 600 s b20：FP16 **123.34** vs BF16 **123.68**。差异在噪声内，**没有 2×**。
3. **真正的 FP16/FP16 累加在 3090 上确实有 2× GEMM 峰值**（cuBLAS 微基准：57.1 → 114.3 TFLOPS，2.00×，达 142.3 TFLOPS 的 80.3%），但 HF 路径没有暴露这个模式。用 ctypes/cuBLAS 强制 `CUBLAS_COMPUTE_16F` 替换所有 `nn.Linear` 后，600 s 中文 b20 反而降到 **141.22** RTFx（Python/ctypes 逐层调用开销），质量仍可接受（CER 0.229%，无 NaN）。
4. **540 没有达到，差距是结构性的。** 最好实测 ASR-only 是 1200 s 中文 b40 的 **210.15 RTFx**，距 540 差 **2.57×**；600 s b20 差 **3.62×**；英文 600 s b20 差 **4.38×**。按 b20/b40 的 kernel 分解，GEMM 约占 GPU 时间 60%–63%；即使 FP16 累加把 GEMM 完美加速 2×，同一实现的乐观上限也只有 **300–323 RTFx**（b40），仍差 **1.67–1.80×**。这 1.7× 必须来自 batch 128、CUDA graphs、融合非 GEMM 算子、Paged KV 等引擎工程，不是 dtype 能补的。
5. **显存没有收益。** FP16 和 BF16 都是 16-bit，权重同为约 5.83 GB。600 s b20：10,300 / 13,224 MB（allocated / reserved，与 BF16 基线完全一致）；1200 s b40：12,999.9 / 13,078 MB；1440 s b48 在 aligner 的 `conv2d1` 上 OOM（尝试分配 4.12 GiB）。在当前共享 GPU 上 **b40 是实测上限，b48 已 OOM，b128 不可测**；即使干净 24 GB，ASR+aligner 同时 b128 也放不下（可行性报告估算 24.5–27.5 GB）。
6. **判定：FP16 作为精度路线可行；FP16 作为 540 的吞吐路线不成立（至少在本探针的 HF 实现上）。** 若门控要求“先证明 540 再写引擎”，本探针的答案是 **NO-GO**；若允许“FP16 精度 + 自研引擎原型带硬性能门控”，则是 **conditional GO**，但原型必须补上约 1.7–1.8× 的非 GEMM/引擎效率，并且要在干净 24 GB 或 5090 上验证。540 目标本身未被修改。

---

## 1. 方法

### 1.1 模型与路径

按 `pipeline/asr/glm.py` 的方式解析，不下载：

```python
from modelscope import snapshot_download
snapshot_download("ZhipuAI/GLM-ASR-Nano-2512", cache_dir="/root/.cache/modelscope", local_files_only=True)
snapshot_download("Qwen/Qwen3-ForcedAligner-0.6B-hf", cache_dir="/root/.cache/modelscope", local_files_only=True)
```

实际路径：

- GLM-ASR：`/root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512`（2,136,470,528 参数，原生 BF16）
- Aligner：`/root/.cache/modelscope/Qwen/Qwen3-ForcedAligner-0___6B-hf`（917,728,896 参数，原生 BF16）

加载方式与 `glm.py` 一致：

```python
AutoProcessor.from_pretrained(path)
GlmAsrForConditionalGeneration.from_pretrained(path, dtype=torch.float16, device_map="cuda")
AutoModelForTokenClassification.from_pretrained(path, dtype=torch.float16, device_map="cuda")
```

BF16 对照只把 `dtype` 换成 `torch.bfloat16`。30 s 分块、`max_new_tokens=512`、`apply_transcription_request`、`prepare_forced_aligner_inputs`、`decode_forced_alignment` 均直接使用 `pipeline/asr/glm.py` 的函数，未改跟踪代码。

### 1.2 音频

| ID | 路径 | 时长 | 语言 | 来源/生成 |
|---|---|---:|---|---|
| ZH30 | `/tmp/fp16-probe/zh_30s.wav` | 30.0 s | zh | 新闻源 `-ss 15 -t 30` |
| ZH60 | `/tmp/fp16-probe/zh_60s.wav` | 60.0 s | zh | 新闻源 `-ss 15 -t 60` |
| R600 | `/tmp/fp16-probe/pi_600s.wav` | 600.0 s | zh | Pi 中文配音源 `-ss 15 -t 600` |
| R720/R960/R1200 | `/tmp/fp16-probe/pi_{720,960,1200}s.wav` | 720/960/1200 s | zh | 同一 Pi 源，等长裁剪 |
| V600EN | `/tmp/fp16-probe/en_v600.wav` | 600.0036 s | en | `/tmp/t2b-video/pi_600s.mp4` 抽音轨，16 kHz mono |

源文件与 SHA-256（与 `docs/agents/rtfx-optimization.md` 一致）：

- 新闻源：`/root/VideoLingo/history/每日必看吳乃仁僅管收12天比教召少陳昭姿誰給台糖壓力請辭了嗎姜至剛耍嘴皮委員心裡答案跟我一樣20260810/normalized_dub.wav`，`7a2f84c216ca3a06f941a933f9bf9a590aa252104cdde98050c7bf95bb4e9e11`
- Pi 源：`/root/VideoLingo/history/Pi-to-Pi_-Two-Way-Agent-Orchestration-with-the-Pi-Coding-Agent_1080p/normalized_dub.wav`，`e2dff1fe9b33970b6223d19c61644452801d4ec30aa9152beb5ea3bd5e2788ad`

生成命令（示例）：

```bash
ffmpeg -y -ss 15 -i <source.wav> -t 600 -ar 16000 -ac 1 -c:a pcm_s16le /tmp/fp16-probe/pi_600s.wav
ffmpeg -y -i /tmp/t2b-video/pi_600s.mp4 -vn -ar 16000 -ac 1 -c:a pcm_s16le /tmp/fp16-probe/en_v600.wav
```

### 1.3 计时口径

- **热 RTFx**：模型加载完成并做 1 次 warmup 后，`音频时长 / 推理耗时`。短音频 3 次、长音频 2–3 次取中位数，同时记录范围。
- **ASR-only**：`asr_s` = `apply_transcription_request`（CPU mel）+ H2D + `generate` + `batch_decode`；不含 aligner。另记录 `asr_generate_s`（仅 `generate`）供对照。
- **ASR+aligner**：`asr_s + align_s`；`align_s` 含 aligner processor、forward、`decode_forced_alignment`。
- **冷加载**：`load_s` 从 `AutoProcessor` 开始到两个模型 `.eval()` + `torch.cuda.synchronize()`；不含 ModelScope 路径解析。每个 JSON 都记录该值。
- **显存**：`torch.cuda.max_memory_allocated()` / `max_memory_reserved()`；另起线程每 0.2 s 采样 `nvidia-smi memory.used` 的最大值。外部进程占用不计入本进程 alloc/reserved。
- **时间戳**：`decode_forced_alignment` 的原始输出；`_clamp_words` 只用于验证契约，失败会计入 `clamp_errors`。零长度词在两种 dtype 下都存在（标点/特殊 token），由 `_clamp_words` 夹到 1 ms，不是 FP16 回归。

### 1.4 质量口径

- **CER**：FP16 对 BF16 的 Levenshtein 编辑距离 / BF16 字符数；另算去空白 CER。
- **exact chunks**：逐块文本完全一致的块数。
- **时间戳**：只对文本完全一致的块按词序配对，统计 start/end 绝对差（mean、p95、max、>80/160/320 ms 计数）；另记录词数不匹配、单调性、范围、零长度、clamp 错误。
- **NaN/Inf**：诊断模式检查 prefill logits、每个 decode step 的 scores、aligner logits 是否 finite；检查生成 token 数是否达到 `max_new_tokens`、空文本、替换字符、重复尾部。
- **BF16 不是 ground truth**：CER 衡量的是 FP16 与 BF16 的分歧，不是绝对 WER；本探针没有人工转写参考。

### 1.5 边界

- 未跑 vLLM；未下载模型；未改 `config.yaml` 或跟踪代码；未杀外部 GPU 进程。
- 未跑 1 h。b40（1200 s）是当前共享 GPU 上能稳定跑的最大 batch；1 h 只会增加 batch 数，不会增加 batch size。b48 已 OOM。
- 未在干净 24 GB 上测 b128；当前 GPU 约 9.6 GB 被外部进程占用，free 约 14.1–14.5 GB。
- `/tmp/fp16-probe/probe.py`、`compare.py`、`profile_asr.py`、`gemm_fp16acc.cu`、`fp16acc.cu` 是临时脚本，不随仓库提交。原始 JSON 含完整 text/timestamps、配置、环境与 summary。
- 参考：`docs/agents/asr-engine-feasibility-540.md`（屋顶线与 24.5–27.5 GB 估算）、`docs/agents/videominer-fast-asr-report.md`（自研引擎取证）、`docs/agents/rtfx-optimization.md`（BF16 基线口径与真实音频）。

---

## 2. 结果

### 2.1 热吞吐（中位数）

| 音频 | 语言 | dtype | batch | ASR-only RTFx | ASR+aligner RTFx | ASR s | aligner s | 峰值 alloc/reserved MB | load s | 词数 | 生成 token |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ZH30 | zh | FP16 | 1 | 20.29 | 19.42 | 1.478 | 0.067 | 6,062 / 6,228 | 4.05 | 83 | 62 |
| ZH30 | zh | BF16 | 1 | 20.40 | 19.56 | 1.471 | 0.064 | 6,062 / 6,228 | 4.11 | 83 | 62 |
| ZH60 | zh | FP16 | 2 | 37.72 | 34.33 | 1.591 | 0.161 | 6,286 / 6,646 | 4.07 | 150 | 112 |
| ZH60 | zh | BF16 | 2 | 36.93 | 31.61 | 1.625 | 0.257 | 6,286 / 6,646 | 4.12 | 150 | 112 |
| R600 | zh | FP16 | 20 | **149.27** | **114.85** | 4.020 | 1.211 | 10,300 / 13,224 | 4.10 | 2,696 | 1,719 |
| R600 | zh | BF16 | 20 | **150.57** | **115.46** | 3.985 | 1.212 | 10,300 / 13,224 | 4.10 | 2,695 | 1,718 |
| V600EN | en | FP16 | 20 | **123.34** | **100.37** | 4.865 | 1.029 | 10,294 / 13,284 | 4.12 | 1,841 | 2,323 |
| V600EN | en | BF16 | 20 | **123.68** | **103.39** | 4.851 | 0.946 | 10,294 / 13,284 | 4.14 | 1,839 | 2,325 |
| R1200 | zh | FP16 | 40 | **210.15** | **143.42** | 5.711 | 2.656 | 13,000 / 13,078 | 4.11 | 5,550 | 3,525 |
| R1200 | zh | BF16 | 40 | **212.84** | **143.88** | 5.638 | 2.702 | 13,000 / 13,078 | 4.12 | 5,549 | 3,524 |

要点：

- **FP16 对 BF16 没有加速**：600 s 中文慢 0.9%，1200 s b40 慢 1.3%，英文 600 s 慢 0.3%。全部在运行间噪声内。
- 30 s/60 s 是单块/双块延迟主导，RTFx 不代表长音频吞吐；540 只看 600 s/1200 s。
- 英文 V600 的生成 token 数（2,323）明显高于中文 R600（1,719），所以同样 600 s，英文 ASR-only 只有 123 RTFx，中文 149 RTFx。540 对英文更难。
- 与已有基线的一致性：BF16 R600 总 RTFx 115.46 vs 报告 120.10（-3.9%）；BF16 V600 总 103.39 vs 106.66（-3.1%）；BF16 R600 ASR-only 150.57 vs 派生 153（-1.6%）。差异约 1.6%–4%，与共享 GPU 状态/热状态一致。

### 2.2 质量（FP16 对 BF16）

| 音频 | dtype | CER | 去空白 CER | exact chunks | 配对词数 | start 偏差 mean / p95 / max (s) | end 偏差 mean / p95 / max (s) | clamp 错误 | NaN/Inf | 截断/重复 |
|---|---|---:|---:|---:|---:|---|---|---:|---|---|
| ZH30 | FP16 | 0.000% | 0.000% | 1/1 | 83 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 无 | 无 |
| ZH60 | FP16 | 0.000% | 0.000% | 2/2 | 150 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 无 | 无 |
| R600 | FP16 | 0.164% | 0.131% | 16/20 | 2,181 | 0.00026 / 0 / 0.160 | 0.00070 / 0 / 0.160 | 0 | 无 | 无 |
| R1200 b40 | FP16 | 0.191% | 0.177% | 32/40 | 4,339 | 0.00024 / 0 / 0.160 | 0.00044 / 0 / 0.160 | 0 | 无 | 无 |
| V600EN | FP16 | 0.153% | 0.162% | 18/20 | 1,669 | 0.00062 / 0 / 0.240 | 0.00019 / 0 / 0.080 | 0 | 无 | 无 |
| R600 | FP16+FP16acc | 0.229% | 0.197% | 16/20 | 2,181 | 0.00029 / 0 / 0.160 | 0.00055 / 0 / 0.160 | 0 | 无 | 无 |
| V600EN | FP16+FP16acc | 0.153% | 0.162% | 18/20 | 1,669 | 0.00062 / 0 / 0.240 | 0.00062 / 0 / 0.080 | 0 | 无 | 无 |

文本差异都是 1–2 个字符：标点（逗号有无）、`Production JComs` vs `ProductionJComs`、`代理门` vs `代理们`（同音）。没有整句丢失、重复或截断。R600/R1200 各有 1 个块词数不同（146 vs 145，源于英文缩写的分词），其余块词数一致。所有块的时间戳单调、在块范围内、`end >= start`；`_clamp_words` 无错误。R1200 b40 未单独跑 decode-score 诊断（b40 的 full-forward logits/scores 会接近显存上限）；其生成文本、时间戳和 aligner logits 均正常，未见 NaN/Inf 迹象。

诊断模式的 finite 检查：

| 检查 | 30 s FP16 | 30 s BF16 | 600 s b20 FP16 |
|---|---|---|---|
| prefill logits finite | 是 | 是 | 是 |
| prefill min/max | -22.63 / 35.28 | -22.38 / 35.25 | -26.80 / 38.28 |
| decode steps | 62 | 62 | 107 |
| decode 非 finite steps | 0 | 0 | 0 |
| decode score min/max | -27.44 / 32.06 | -27.50 / 32.25 | -36.47 / 37.81 |
| aligner logits finite | 是 | 是 | 是 |
| aligner logits min/max | -17.59 / 17.70 | -17.63 / 17.75 | -19.23 / 19.22 |
| 达到 max_new_tokens | 否 | 否 | 否 |

### 2.3 批量扩展（中文 FP16）

| batch | 音频 | ASR-only RTFx | ASR+aligner RTFx | ASR s | 峰值 alloc/reserved MB | 备注 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 30 s | 20.29 | 19.42 | 1.478 | 6,062 / 6,228 | 延迟主导 |
| 2 | 60 s | 37.72 | 34.33 | 1.591 | 6,286 / 6,646 | 延迟主导 |
| 20 | 600 s | 149.27 | 114.85 | 4.020 | 10,300 / 13,224 | 基线 batch |
| 24 | 720 s | 168.89 | 124.89 | 4.263 | 11,192 / 12,828 | |
| 32 | 960 s | 193.41 | 135.59 | 4.964 | 11,565 / 11,636 | |
| 40 | 1,200 s | **210.15** | 143.42 | 5.711 | 13,000 / 13,078 | 实测最大 batch |
| 48 | 1,440 s | — | — | — | — | OOM（aligner `conv2d1` 尝试 4.12 GiB） |

批量收益递减但仍明显：b20→b40 为 1.41×。用 log/饱和外推，b128 大约 **250–315 RTFx**；线性外推（476）不可信。这个外推只用于说明趋势，不是实测值。540 仍高于该范围。

### 2.4 Kernel 分解（torch profiler，单次 generate）

| 音频/batch | dtype | `aten::mm` | `aten::addmm` | GEMM 合计 / CUDA 总时间 | flash attention | `aten::cat` | 其它 |
|---|---|---:|---:|---:|---:|---:|---:|
| R600 b20 | FP16 | 974.9 ms | 543.8 ms | 1,518.6 / 2,416 ms = **62.9%** | 246.9 ms (10.2%) | 198.3 ms (8.2%) | mul/add/copy/elementwise 等 |
| R600 b20 | BF16 | 969.2 ms | 543.3 ms | 1,512.5 / 2,411 ms = **62.7%** | 247.7 ms (10.3%) | 199.1 ms (8.3%) | 同上 |
| R1200 b40 | FP16 | 1,325.7 ms | 1,085.7 ms | 2,411.4 / 4,032 ms = **59.8%** | 514.3 ms (12.8%) | 395.5 ms (9.8%) | 同上 |

FP16 与 BF16 的 kernel 组成几乎相同；FP16 profile 中实际跑的是 `ampere_fp16_s16816gemm_fp16_...`，但 compute type 仍是 FP32 累加（见 §3.1）。

### 2.5 FP16/FP16 累加微基准（cuBLAS，4096³）

独立 CUDA 程序（`/tmp/fp16-probe/gemm_fp16acc.cu`，sm_86，CUDA 12.8）：

| 模式 | 时间 | TFLOPS | 相对 FP32 累加 |
|---|---:|---:|---:|
| FP16 权重 / FP32 累加 | 2.405 ms | 57.1 | 1.00× |
| FP16 权重 / FP16 累加 | 1.203 ms | 114.3 | **2.00×** |

`CUBLAS_COMPUTE_16F` 达到 3090 官方 FP16/FP16 累加峰值 142.3 TFLOPS 的 **80.3%**。数值误差（K=2048，M=256，N=2048，激活 std=1，权重 std=0.02，对照 FP32 累加）：max abs **0.007812**，mean abs **0.000783**，max rel 2.596（仅出现在接近 0 的输出），**无 NaN/Inf**。这证明硬件和 cuBLAS 能做 FP16 累加，且单层误差可控；不代表多层模型端到端误差。

### 2.6 FP16/FP16 累加端到端补丁（非产品代码）

为验证“如果 GEMM 真用 FP16 累加会怎样”，用 ctypes 加载一个 `CUBLAS_COMPUTE_16F` 的 linear 实现，替换所有 `nn.Linear.forward`（仅 inference；attention/conv 仍是 FP32 累加）。这是混合精度路径，不是完整 FP16-accum 引擎。

| 音频 | batch | ASR-only RTFx | ASR s | aligner forward s | CER | 时间戳 max | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---|
| R600 zh | 20 | 141.22 | 4.249 | 0.680 | 0.229% | 0.160 s | 无 |
| V600EN | 20 | 111.09 | 5.401 | 0.607 | 0.153% | 0.240 s | 无 |

比默认 FP16 慢（R600 149.27→141.22，英文 123.34→111.09），原因是每层 `nn.Linear` 都经过 Python/ctypes 调用和一次 `torch.empty`，decode 的 100+ 步 × 140+ 层把开销放大；aligner 的大 GEMM 则变快（0.815→0.680 s）。这是实现开销，不是 FP16 累加本身的性能结论。质量仍然通过；30 s 诊断显示 prefill/decode/aligner logits 全部 finite（62 个 decode step，0 个非 finite），未达到 `max_new_tokens`。

---

## 3. 分析

### 3.1 为什么 FP16 权重没有加速

`torch.matmul` 在 FP16 输入上调用 `cublasGemmEx`，cuBLAS 日志明确显示：

```text
computeType: type=cublasComputeType_t; val=CUBLAS_COMPUTE_32F(68)
algo: type=SOME TYPE; val=CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)
```

`torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction` 为 `True`，但把它设为 `False` 对 4096³ 矩阵乘的耗时没有影响（2.403 ms vs 2.399 ms）；它没有把 compute type 切到 `CUBLAS_COMPUTE_16F`。因此：

- **FP16 权重 + PyTorch 默认路径 = FP16 存储 + FP32 累加**，峰值与 BF16 相同（71.2 TFLOPS），实测 57–60 TFLOPS。
- 真正的 FP16/FP16 累加需要显式 `CUBLAS_COMPUTE_16F`（cuBLAS）或 CUTLASS FP16 accumulate；这正是自研引擎可以做的，也是唯一能拿到 142.3 TFLOPS 峰值的路径。
- 这也解释了为什么 600 s/1200 s 的 FP16 与 BF16 吞吐几乎相同，以及为什么“只把 dtype 改成 FP16”不会带来任何 540 收益。

### 3.2 540 差距

以实测 ASR-only 为准（不改变 540 目标）：

| 口径 | 实测 RTFx | 距 540 | 说明 |
|---|---:|---:|---|
| R600 zh b20 FP16 | 149.27 | **3.62×** | 与基线同 batch |
| R600 zh b20 BF16 | 150.57 | 3.59× | 对照 |
| R1200 zh b40 FP16 | **210.15** | **2.57×** | 当前环境最大 batch |
| R1200 zh b40 BF16 | 212.84 | 2.54× | 对照 |
| V600EN b20 FP16 | 123.34 | **4.38×** | 英文 token 更多 |
| ZH30 b1 FP16 | 20.29 | 26.6× | 延迟主导，不作 540 依据 |

**FP16 累加能补多少？** 按 b20/b40 profile，GEMM（`aten::mm` + `aten::addmm`）占 GPU 时间 60%–63%。即使 GEMM 完美 2×：

- b40：`210.15 / (0.40 + 0.60/2) = 300.2` RTFx（GEMM 60%）；若 GEMM 65%，`210.15 / 0.675 = 311.3`；若 70%，`323.3`。
- b20：`149.27 / (0.371 + 0.315) = 217.8` RTFx。

也就是说，**FP16 累加在最好的 b40 情况下也只能到约 300–323 RTFx，仍差 540 的 1.67–1.80×**。这还没有扣除 attention、`aten::cat`、elementwise、memcpy 和 decode 的带宽下限。要让 540 成立，自研引擎必须在 FP16 累加之外再拿到约 1.7–1.8× 的非 GEMM/调度效率；候选手段是 batch 128、CUDA graphs、Paged KV、融合 cat/elementwise、减少每步 kernel launch、把 mel/预处理移出关键路径。**这些都不是 dtype 改动能自动带来的。**

用可行性报告的 440 TFLOP/h 估算（注意这是估算，不是本探针实测）：

- 540 RTFx = 1 h / 6.667 s = 66.0 TFLOPS = 142.3 峰值的 **46.4%**。
- 实测 b40 210.15 RTFx ≈ 25.7 TFLOPS = 142.3 的 **18.1%**；b20 149.27 ≈ 18.2 TFLOPS = **12.8%**。
- 从 18.1% 到 46.4% 需要 2.57× 效率；FP16 累加贡献约 1.43–1.54×，剩余约 1.67–1.80× 必须来自引擎。

**batch 128 能否补上剩余差距？** 本环境无法实测（b48 OOM）。用 b20→b40 的 1.41× 做参照，b128 的收益很可能递减；log/饱和外推给出 b128 约 250–315 RTFx（FP32 累加），再乘 FP16 累加的 1.43–1.54×，约 **360–485 RTFx**。这仍低于 540，且外推不确定性很大。只有再叠加融合算子/CUDA graphs 等 1.1–1.5×，才可能触及 540。**540 在 3090 上不是物理不可能，但本探针没有证明它；它依赖一个尚未实现、尚未测过的引擎。**

### 3.3 显存与批量

- FP16 与 BF16 权重都是 2 bytes/参数，权重总量约 5.83 GB；FP16 **不省显存**。
- 600 s b20 的 10,300 / 13,224 MB 与 BF16 基线完全一致；1200 s b40 为 12,999.9 / 13,078 MB。
- 当前 GPU 有约 9.6 GB 外部占用，free 约 14.1–14.5 GB；b48 在 aligner `conv2d1` OOM（4.12 GiB 单次分配），b40 是实测上限。
- b128 ASR-only 在干净 24 GB 上或许能放下（可行性报告估算 13.6–16.6 GB），但 ASR+aligner 同时 b128 需要 24.5–27.5 GB，放不下；必须顺序执行或缩 aligner batch。当前共享 GPU 上 b128 不可测。
- 因此，“FP16 让 batch 更大”不成立；batch 上限由 16-bit 权重和激活/KV 决定，与 BF16 相同。

### 3.4 质量边界

- FP16 权重（FP32 累加）在 30/60/600/1200 s、中英文上均通过：CER ≤ 0.191%，时间戳 p95 为 0、最大 0.24 s，无 NaN/Inf、无截断/重复。
- FP16 累加（混合路径）在 600 s 中英文上 CER 0.153%–0.229%，时间戳同样正常；单层 cuBLAS 微基准无 NaN、绝对误差 < 0.008。质量看起来可接受。
- **但这不是绝对质量证明**：BF16 是对照，不是 ground truth；FP16 累加只替换了 `nn.Linear`，attention/conv 仍 FP32 累加；只测了 2 个 600 s 片段（中/英）和 1 个 1200 s 片段。若要把 FP16 作为引擎精度路线，仍需 ground-truth 转写集上的人工/WER 评估，以及完整 FP16-accum kernel 的端到端验证。
- 没有观察到溢出/NaN 的迹象；FP16 的数值范围（logits 约 -37–38）远低于 65504，score 也远低于上限。真正的风险在多层 FP16 累加的舍入误差，而不是溢出。

---

## 4. 判定与下一步

### 4.1 判定

- **FP16 精度路线：可行（GO）。** 质量与 BF16 基本无差，显存不增加，代码路径无需改契约。
- **FP16 → ASR-only 540：本探针不成立（NO-GO on current evidence）。** 最好实测 210.15 RTFx（b40），差 2.57×；FP16 累加的乐观上限约 300–323 RTFx，仍差 1.67–1.80×。540 需要自研引擎在 dtype 之外再拿约 1.7–1.8×，本探针没有证明这部分。
- **条件式结论：** 如果决策是“FP16 能不能作为自研引擎的精度路线”，答案是 **可以**；如果决策是“FP16 能不能保证 540、可以据此开始写完整引擎”，答案是 **不能**。建议把 540 保留为目标，但设置一个**原型性能门控**：只有最小引擎原型在干净 24 GB（或 5090）上实测 ≥540（或明确的分阶段目标），才投入完整引擎。不要因为“FP16 峰值 142.3 TFLOPS”就默认 540 成立。

### 4.2 选项（不自行改目标）

1. **换 GPU（推荐度最高，工程风险最低）：** 5090 的 BF16 稠密峰值 209.5 TFLOPS，540 只需 31.5%；不需要 FP16 累加就能留出余量。代价是硬件预算。
2. **保留 3090，降低目标：** 以实测为准，ASR-only 300–320 RTFx 是 FP16 累加 + b40 的乐观上限；若要 540，必须接受“自研引擎原型可能达不到”的风险。目标调整需要产品决策，本报告不自行修改。
3. **做最小自研引擎原型（保留 540 的唯一 3090 路径）：** 只实现 GLM-ASR 的 encoder/decoder + CUTLASS FP16-accumulate GEMM + CUDA graphs + Paged KV + batch 128（或分阶段 batch），先在干净 24 GB 上测 ASR-only。若原型达不到 540 的 80%–90%，就应换 GPU 或降目标。这个原型的成本远低于完整引擎，且能直接回答本探针无法回答的 batch-128/融合算子问题。
4. **INT8：** 理论上 3090 INT8 峰值 284.7 TOPS，540 只需 23.2%，但 GLM-ASR 没有官方 INT8 checkpoint，vLLM 也没有该架构 INT8 路径；需要自研量化和校准，质量风险最高。不建议作为第一选择。

### 4.3 自研引擎要补什么（量化）

以 b40 实测 210.15 RTFx 为起点，目标 540：

- 纯 FP16 累加（GEMM 2×，60% 占比）：约 300 RTFx，还需 **1.80×**。
- 若 GEMM 占比 70%：约 323 RTFx，还需 **1.67×**。
- 若 batch 128 再加 1.2–1.5×：约 360–485 RTFx，还需 **1.11–1.50×**。
- 这最后的 1.1–1.5× 必须来自非 GEMM 路径：decode 是带宽约束（b40 每步要读约 4.5 GB 权重；107 步 × 4.5 GB / 936 GB/s ≈ 0.51 s 下限），需要更大 batch 分摊、Paged KV、CUDA graphs、融合 cat/elementwise、减少 kernel launch。FP16 累加不会改善这些。
- 换言之：**540 的必要条件是 FP16 累加 + 引擎工程，充分条件尚未证明。**

---

## 5. 证据索引

`bench/results/fp16-probe/` 下的关键文件：

| 文件 | 内容 |
|---|---|
| `zh30_{fp16,bf16}_b1_hot.json` | 30 s 热吞吐与质量 |
| `zh60_{fp16,bf16}_b2_hot.json` | 60 s 热吞吐与质量 |
| `r600_{fp16,bf16}_b20_hot.json` | 600 s 中文 b20 主对照 |
| `v600en_{fp16,bf16}_b20_hot.json` | 600 s 英文 b20 主对照 |
| `r1200_{fp16,bf16}_b40_hot.json` | 1200 s 中文 b40 最大 batch 对照 |
| `r720_fp16_b24_hot.json`、`r960_fp16_b32_hot.json` | 批量扩展 |
| `r600_fp16acc_b20_hot.json`、`v600en_fp16acc_b20_hot.json`、`zh30_fp16acc_b1_hot.json` | FP16/FP16 累加混合补丁 |
| `zh30_{fp16,bf16}_b1_diag.json`、`r600_fp16_b20_diag.json`、`zh30_fp16acc_b1_diag.json` | finite logits/scores、encoder/decoder 分解 |
| `r600_{fp16,bf16}_b20_profile.json`、`r1200_fp16_b40_profile.json` | torch profiler kernel 分解 |
| `compare_*.json` | CER、时间戳、词数、clamp 错误 |
| `gemm_fp16acc_output.txt`、`cublas_compute_type.txt`、`r1440_fp16_b48_oom.txt` | cuBLAS 微基准输出、compute type 日志、b48 OOM 现场 |
| `summary_hot.json` | 上述热跑的统一摘要 |
| `pilot_*.json` | 早期英文对齐试跑，已弃用，仅留档 |

复跑命令（示例；`PY`/`OUT` 按需替换）：

```bash
cd /root/VideoLingo-refactor
PY=/root/VideoLingo/.venv/bin/python
OUT=bench/results/fp16-probe

# 30 s
$PY /tmp/fp16-probe/probe.py --audio /tmp/fp16-probe/zh_30s.wav \
  --glm-dtype fp16 --batch-size 1 --mode hot --warmups 1 --runs 3 \
  --language Chinese --output $OUT/zh30_fp16_b1_hot.json

# 600 s 中文
$PY /tmp/fp16-probe/probe.py --audio /tmp/fp16-probe/pi_600s.wav \
  --glm-dtype fp16 --batch-size 20 --mode hot --warmups 1 --runs 3 \
  --language Chinese --output $OUT/r600_fp16_b20_hot.json

# 1200 s b40
$PY /tmp/fp16-probe/probe.py --audio /tmp/fp16-probe/pi_1200s.wav \
  --glm-dtype fp16 --batch-size 40 --mode hot --warmups 1 --runs 2 \
  --language Chinese --output $OUT/r1200_fp16_b40_hot.json

# 诊断
$PY /tmp/fp16-probe/probe.py --audio /tmp/fp16-probe/pi_600s.wav \
  --glm-dtype fp16 --batch-size 20 --mode diag --language Chinese \
  --output $OUT/r600_fp16_b20_diag.json

# 质量对比
$PY /tmp/fp16-probe/compare.py --fp16 $OUT/r600_fp16_b20_hot.json \
  --bf16 $OUT/r600_bf16_b20_hot.json --output $OUT/compare_r600.json

# kernel profile
$PY /tmp/fp16-probe/profile_asr.py --audio /tmp/fp16-probe/pi_600s.wav \
  --dtype fp16 --batch-size 20 --output $OUT/r600_fp16_b20_profile.json

# FP16/FP16 累加微基准
cd /tmp/fp16-probe
/usr/local/cuda-12.8/bin/nvcc -O3 -arch=sm_86 -o gemm_fp16acc gemm_fp16acc.cu \
  -I/usr/local/cuda-12.8/include -L/usr/local/cuda-12.8/lib64 -lcublas
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 ./gemm_fp16acc
```

关键环境输出：

```text
torch 2.11.0+cu129, transformers 5.14.1, CUDA 12.9
GPU 0: NVIDIA GeForce RTX 3090, 24,576 MiB, driver 570.144
FP16 reduced precision reduction: True (does not change compute type)
cuBLAS: computeType = CUBLAS_COMPUTE_32F(68)
FP16/FP32acc 4096^3: 57.1 TFLOPS
FP16/FP16acc 4096^3: 114.3 TFLOPS (2.00x)
```