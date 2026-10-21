# RTFx 基线（T2a）

关联：Epic #1；T2a #4；spec #16；工具审计 `docs/agents/tool-audit.md`。
实现 commit：`ff1233a`。分支：`feat/t2a-rtfx-baseline`。
本任务只建立权重与性能基线，不实现 transcribe 阶段（T4 #8），不跑全流程。

## 范围

本任务负责：

- 从 ModelScope 获取官方 GLM-ASR-Nano-2512 与 Qwen3-ForcedAligner-0.6B-hf 权重。
- 提供一个可复跑的基准工具 `bench.asr_rtfx`，测量 30 秒、1 分钟、10 分钟音频的冷/热 RTFx、显存峰值与批量大小。
- 记录缺依赖、缺权重、缺输入时的错误行为。

本任务不负责：

- `pipeline.runner` 的 transcribe 阶段与 transcript JSON 契约（T4）。
- 批量、分块、注意力实现、对齐分组、流水线重叠的优化（T2b）。
- 是否继续下游阶段的决策（T2c，人类门）。
- 全流程运行。

## 权重

| 模型 | ModelScope ID | 本地路径 | safetensors 大小 | 授权 |
|------|---------------|----------|------------------|------|
| GLM-ASR-Nano-2512 | `ZhipuAI/GLM-ASR-Nano-2512` | `/root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512` | 4,515,776,712 B (4.21 GiB) | 权重 MIT；上游代码 Apache-2.0 |
| Qwen3-ForcedAligner-0.6B-hf | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | `/root/.cache/modelscope/Qwen/Qwen3-ForcedAligner-0___6B-hf` | 1,835,545,960 B (1.71 GiB) | Apache-2.0 |

- 下载日期：2026-10-21。ModelScope 未固定 revision（默认 `master`），以文件大小作为同一性检查。
- 权重在仓库外缓存，不进入版本库。
- 下载命令：

  ```bash
  PYTHON=/root/VideoLingo/.venv/bin/python
  $PYTHON -m bench.asr_rtfx download --cache-dir /root/.cache/modelscope
  ```

- 权重来源页面：<https://modelscope.cn/models/ZhipuAI/GLM-ASR-Nano-2512>、<https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B-hf>。

## 前置条件

- Python：`/root/VideoLingo/.venv/bin/python`（3.10.19）。
- GPU：NVIDIA GeForce RTX 3090，24 GB（torch 报告 24132.2 MB）。
- 依赖：torch 2.11.0+cu129、transformers 5.14.1、modelscope 1.27.0、soundfile、numpy。
- `transformers` 必须包含 `GlmAsrForConditionalGeneration` 与 `qwen3_asr` 的 `Qwen3ASRForTokenClassification`。当前 5.14.1 已包含；更旧版本会由 preflight 明确拒绝。
- 不需要 `torchcodec`：基准输入是 16 kHz mono WAV，由 soundfile 读取。
- 磁盘：权重合计约 6.35 GB。

## 夹具

- 已提交：`tests/fixtures/audio/zh_30s.wav`。来源 `zai-org/GLM-ASR` 的 `examples/example_zh.wav`（Apache-2.0），6 秒片段循环 5 次并裁到 30 秒，16 kHz mono PCM s16le。来源与生成命令见 `tests/fixtures/audio/README.md`。
- 60 秒与 600 秒夹具不提交，由已提交的 30 秒文件确定性生成：

  ```bash
  mkdir -p /tmp/t2a-audio
  cp tests/fixtures/audio/zh_30s.wav /tmp/t2a-audio/zh_30s.wav
  ffmpeg -y -stream_loop 1 -i tests/fixtures/audio/zh_30s.wav -t 60 \
    -ar 16000 -ac 1 -c:a pcm_s16le /tmp/t2a-audio/zh_60s.wav
  ffmpeg -y -stream_loop 19 -i tests/fixtures/audio/zh_30s.wav -t 600 \
    -ar 16000 -ac 1 -c:a pcm_s16le /tmp/t2a-audio/zh_600s.wav
  ```

- 夹具是循环语音，只用于稳定吞吐测量，不用于评估 ASR 质量。

## 口径

- `chunk_seconds = 30`，`batch_size = 1`，`language = Chinese`，`max_new_tokens = 512`。
- 每个 30 秒块先整块转写，再整块做强制对齐；对齐时间戳必须非空、单调、在该块时长 + 0.5 秒内，否则基准失败。
- **冷启动 RTFx** = 音频时长 / (模型加载 + 一次完整推理)。包含模型加载时间，不含 Python 导入与应用启动。
- **热启动 RTFx** = 音频时长 / 一次完整推理。模型已加载，先做 1 次 warmup（只跑第一个 30 秒块），再连续测量 3 次并取中位数。
- **纯推理 RTFx** = 音频时长 / 推理耗时，不含模型加载；冷启动表中单独列出，便于与热启动对比。
- 显存峰值取 `torch.cuda.max_memory_allocated()` 与 `max_memory_reserved()`，单位 MB。冷启动包含加载与推理，热启动只包含测量运行。
- RTFx 只测量模型计算，不测量进程启动、音频解码、报告写入。

## 命令

```bash
cd /root/VideoLingo-refactor
PYTHON=/root/VideoLingo/.venv/bin/python
CACHE=/root/.cache/modelscope
OUT=bench/results/t2a

$PYTHON -m bench.asr_rtfx preflight --cache-dir "$CACHE" \
  --audio /tmp/t2a-audio/zh_30s.wav \
  --audio /tmp/t2a-audio/zh_60s.wav \
  --audio /tmp/t2a-audio/zh_600s.wav

# 冷启动：每个音频一条命令
$PYTHON -m bench.asr_rtfx run --audio /tmp/t2a-audio/zh_30s.wav \
  --cache-dir "$CACHE" --mode cold --batch-size 1 --runs 1 \
  --output "$OUT/zh-30s-cold.json"
$PYTHON -m bench.asr_rtfx run --audio /tmp/t2a-audio/zh_60s.wav \
  --cache-dir "$CACHE" --mode cold --batch-size 1 --runs 1 \
  --output "$OUT/zh-60s-cold.json"
$PYTHON -m bench.asr_rtfx run --audio /tmp/t2a-audio/zh_600s.wav \
  --cache-dir "$CACHE" --mode cold --batch-size 1 --runs 1 \
  --output "$OUT/zh-600s-cold.json"

# 热启动：1 次 warmup + 3 次测量
$PYTHON -m bench.asr_rtfx run --audio /tmp/t2a-audio/zh_30s.wav \
  --cache-dir "$CACHE" --mode hot --batch-size 1 --warmups 1 --runs 3 \
  --output "$OUT/zh-30s-hot.json"
$PYTHON -m bench.asr_rtfx run --audio /tmp/t2a-audio/zh_60s.wav \
  --cache-dir "$CACHE" --mode hot --batch-size 1 --warmups 1 --runs 3 \
  --output "$OUT/zh-60s-hot.json"
$PYTHON -m bench.asr_rtfx run --audio /tmp/t2a-audio/zh_600s.wav \
  --cache-dir "$CACHE" --mode hot --batch-size 1 --warmups 1 --runs 3 \
  --output "$OUT/zh-600s-hot.json"
```

## 结果

环境：RTX 3090（24 GB）、torch 2.11.0+cu129、transformers 5.14.1、modelscope 1.27.0。
模型：GLM-ASR-Nano-2512 + Qwen3-ForcedAligner-0.6B-hf。批量大小均为 1。

| 音频 | 模式 | 次数 | RTFx 中位 | RTFx 范围 | 纯推理 RTFx | 加载 s | ASR 中位 s | 对齐中位 s | 显存峰值 alloc/reserved MB |
|------|------|------|-----------|-----------|-------------|--------|------------|------------|---------------------------|
| 30 s | 冷 | 1 | 4.64 | — | 12.65 | 4.10 | 2.15 | 0.23 | 6062 / 6228 |
| 30 s | 热 | 3 | 17.32 | 15.81–17.38 | 17.32 | 4.02 | 1.67 | 0.07 | 6062 / 6228 |
| 60 s | 冷 | 1 | 7.37 | — | 14.75 | 4.08 | 3.86 | 0.21 | 6062 / 6228 |
| 60 s | 热 | 3 | 16.54 | 16.29–17.38 | 16.54 | 4.06 | 3.33 | 0.23 | 6062 / 6228 |
| 600 s | 冷 | 1 | 14.58 | — | 16.18 | 4.09 | 34.61 | 2.46 | 6062 / 6228 |
| 600 s | 热 | 3 | 16.81 | 16.68–16.88 | 16.81 | 4.05 | 33.96 | 1.86 | 6062 / 6228 |

观察：

- 热启动 RTFx 稳定在 16.5–17.3，低于 20 的可用线。这是 T2a 基线结果，不是 T2c 的决策。
- ASR 占总推理时间的约 90%–97%；ForcedAligner 只占约 4%–10%（冷启动 30 秒最高）。优化重点在 ASR。
- 模型加载约 4.0–4.1 秒；冷启动差异主要来自加载。
- 显存峰值约 6.06 GB allocated / 6.23 GB reserved，批量 1。提升批量还有显存余量，但本任务不调优。
- 600 秒音频按 30 秒切成 20 块，处理时间近似线性；基线没有跨块重叠或流水线重叠。

## 失败判定

以下任一情况为 FAIL：

- preflight 未返回 0，或未打印两个模型的路径与 safetensors 大小。
- 缺权重时进程以 0 退出，或没有指出缺失的模型 ID 与缓存目录。
- 目标 JSON 缺失，或 `runs` 为空，或没有 `rtfx`、`vram_peak_allocated_mb`、`batch_size`。
- 对齐结果为空、时间戳回退或越界；基准没有报错而是继续记录数字。
- 运行命令实际执行了 ingest、segment、translate、subtitle、burn 或全流程。
- 没有可复现的命令与证据。
- 实现 agent 与验证 agent 是同一个 agent。

## 证据

- `bench/results/t2a/preflight.json`：环境、权重路径与大小、三个音频的采样率与时长。
- `bench/results/t2a/zh-{30s,60s,600s}-{cold,hot}.json`：每次运行的耗时、ASR/对齐分解、RTFx、显存、词数。
- 数字来自 2026-10-21 的实测；GPU 上另有其它进程占用约 8–10 GB 显存（验证时为 9.5 GB），但不影响本进程的 allocated/reserved 统计。

## 独立验证

由未参与实现的 agent 执行。建议复核：

1. 用同一缓存运行 preflight，确认路径、大小、版本一致。
2. 用同一夹具与命令复跑热启动 30 秒与 600 秒，热启动 RTFx 中位数应在报告值的 ±10% 内；冷启动在 ±25% 内；显存峰值在 ±5%（约 ±300 MB）内。
3. 用空缓存目录运行 preflight，确认退出码 2 且错误包含模型 ID。
4. 检查证据 JSON 的 `words`、时间戳校验结果与 `batch_size`。
5. 确认没有跑全流程。

## 验证记录

- Ticket: #4
- Stage: T2a（性能基线，非阶段）
- Implementer: 实现会话（本分支作者）
- Verifier: 独立验证 agent（未参与实现）
- Commit: ff46d84（HEAD；实现 ff1233a）
- Scope: 权重可用性 + 30s/60s/600s 冷/热 RTFx 基线
- Preconditions: RTX 3090、旧 venv、ModelScope 缓存 `/root/.cache/modelscope`；60s/600s 音频由夹具按文档生成
- Config path: 不适用（benchmark CLI）
- Fixtures: `tests/fixtures/audio/zh_30s.wav`；`/tmp/t2a-audio/zh_60s.wav`、`zh_600s.wav`
- Command: 见“命令”小节；验证使用同样的 preflight 与 run 参数，输出写到 `/tmp/verify-rtfx`
- Expected: 热启动 RTFx 中位数在报告值 ±10% 内，冷启动 ±25% 内，显存 ±5% 内；空缓存 preflight 退出 2
- Result: PASS。30s 热 17.115（报告 17.316，-1.16%）；600s 热 16.873（报告 16.815，+0.35%）；30s 冷 4.708（报告 4.636，+1.54%）；显存 6062/6228 MB，与报告一致；空缓存退出 2 并指出模型 ID；六个证据 JSON 与报告表一致；未跑全流程
- Exit code: 0（基准）；2（空缓存 preflight）
- Evidence: `bench/results/t2a/`；验证输出在 `/tmp/verify-rtfx`
- Verdict: PASS