# RTFx 优化（T2b）

关联：Epic #1；T2b #5；T2c #6；spec #16；T2a 基线 `docs/agents/rtfx-baseline.md`。
实现 commit：`a74a3c8`。分支：`feat/t2b-performance`。

## 验收口径

**20 RTFx 不是 T2b 的验收目标，也不是通过线。** 它只是 T2c 报告中的历史可用参考线之一。T2b 以真实语音测试集上的相对提升、绝对 RTFx、显存与可复现性为准。

本任务不改变 transcribe 阶段契约：输入仍是 16 kHz mono `media/audio.wav`，输出仍是 `transcript.json`（language、segments、词级时间戳）。T2b 只改批量策略与默认值。

## 测试集

真实语音来自本机旧归档 `history/` 的用户媒体，不提交到仓库；测试集由固定命令生成，源文件 SHA-256 如下。合成夹具是已提交的 `tests/fixtures/audio/zh_30s.wav`（循环 6 秒语音），只作与 T2a 的可比控制组。

| ID | 来源 | 时长 | 词数 | 语言 | 说明 |
|----|------|------|------|------|------|
| R30 | 中文新闻配音 | 30 s | 83 | zh | `-ss 15 -t 30` |
| R60 | 中文新闻配音 | 60 s | 150 | zh | `-ss 15 -t 60` |
| R480 | 中文新闻配音 | 480 s | 1793 | zh | `-ss 15 -t 480` |
| R600 | Pi 视频中文配音 | 600 s | 2695 | zh | `-ss 15 -t 600` |
| S600 | 合成循环夹具 | 600 s | 2200 | zh | 与 T2a 同一夹具 |

- 中文新闻源：`/root/VideoLingo/history/每日必看吳乃仁僅管收12天比教召少陳昭姿誰給台糖壓力請辭了嗎姜至剛耍嘴皮委員心裡答案跟我一樣20260810/normalized_dub.wav`，506.88 s，SHA-256 `7a2f84c216ca3a06f941a933f9bf9a590aa252104cdde98050c7bf95bb4e9e11`。
- Pi 中文配音源：`/root/VideoLingo/history/Pi-to-Pi_-Two-Way-Agent-Orchestration-with-the-Pi-Coding-Agent_1080p/normalized_dub.wav`，2091.61 s，SHA-256 `e2dff1fe9b33970b6223d19c61644452801d4ec30aa9152beb5ea3bd5e2788ad`。
- 生成命令（等长裁剪，16 kHz mono）：

  ```bash
  ffmpeg -y -ss 15 -i <source.wav> -t <duration> \
    -ar 16000 -ac 1 -c:a pcm_s16le /tmp/t2b-real/<id>.wav
  ```

- 所有真实测试显式传 `--language Chinese`；生产 stage 在 R600 上自动检测为 `zh`。
- R600 是中文配音，不是英文；首次实验曾按英文对齐，已用中文重跑，报告只采用重跑数据。

## 口径

- **热 RTFx** = 音频时长 / 推理耗时；模型已加载，1 次 warmup，批量 sweep 2 次测量取中位数、短音频 3 次。不含模型加载与 ModelScope 路径解析。
- **冷阶段壁钟** = runner 从启动到写完 transcript 的实际时间，包含模型解析、加载和推理。短音频由加载主导，单独报告，不与热 RTFx 混用。
- 批量与分块通过 `bench.asr_rtfx` 的 `--batch-size`、`--chunk-seconds` 控制；显存取 `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`。
- GPU 上另有约 9.5 GB 被其它进程占用；显存数字是本进程峰值，不是整卡占用。

## 批量：主要收益

R600 真实 10 分钟，30 秒分块：

| batch | 热 RTFx | 范围 | 显存 alloc/reserved MB | ASR s | 对齐 s |
|-------|---------|------|------------------------|-------|--------|
| 1 | 14.37 | 14.30–14.43 | 6062 / 6228 | 40.15 | 1.81 |
| 2 | 25.07 | 24.95–25.19 | 6287 / 6646 | 22.71 | 1.33 |
| 4 | 43.81 | 43.54–44.07 | 6733 / 7266 | 12.36 | 1.25 |
| 8 | 62.41 | 61.96–62.87 | 7624 / 8838 | 8.51 | 1.17 |
| 16 | 83.26 | 83.17–83.35 | 9408 / 11754 | 6.05 | 1.14 |
| 20 | **120.10** | 118.57–121.64 | 10300 / 13224 | 3.92 | 1.14 |

- 20 是本段音频的块数，batch > 20 与 20 等价。
- 合成 S600 的曲线一致：1→16.81、2→30.36、4→54.82、8→80.20、16→103.26、20→147.95。
- 480 秒 R480：1→15.31、8→73.69、16→111.11（16 块，batch ≥16 等价）。
- 60 秒 R60：1→21.97、2→33.70。
- 30 秒只有 1 块，批量无效：20.09。
- **相对 batch=1（与 T2a 相同口径）**：R600 14.37→120.10（8.36x）；R480 15.31→111.11（7.26x）；S600 16.81→147.95（8.80x）。S600 的 batch=1 是 T2a 已发布基线；R600/R480 的 batch=1 由 T2b 按同一口径实测。

## 分块

R600，30 秒块是吞吐最优：

| 分块 | 块数 | batch | 热 RTFx | 显存 alloc/reserved MB | 词数 |
|------|------|-------|---------|------------------------|------|
| 15 s | 40 | 20 | 99.35 | 8070 / 9994 | 2680 |
| 30 s | 20 | 20 | **120.10** | 10300 / 13224 | 2695 |
| 60 s | 10 | 10 | 82.00 | 10299 / 13222 | 2676 |

15 秒块增加批次数与对齐开销；60 秒块降低并行度。**默认保持 30 秒。**

## 其它轴

| 项 | 结论 | 证据 |
|----|------|------|
| 注意力实现 | 两个模型都已是 SDPA、bf16；未安装 flash-attn，无法安全切换。无代码改动。 | 运行时 `_attn_implementation=sdpa` |
| 对齐分组 | 批量 20 时 20 个块一次对齐，对齐 1.14 s / 总推理 5.06 s（22.5%）。已按最大块组分组，无进一步分组收益。 | R600 b20 |
| 流水线重叠 | 未采用。理论上 ASR/对齐完全重叠的上限约 5.06→3.92 s（1.29x），但需把 ASR 拆成更小批；batch 16 的 ASR 已 6.05 s，拆分损失大于重叠收益。 | 批量曲线 |
| 生成上限 | `max_new_tokens` 128/256/512 在 R30 上词数均为 83，耗时无实质变化（EOS 提前结束）。保留 512。 | `zh_real_30s-t128/t256` |
| 模型加载 | 未优化。热 RTFx 口径不含加载；R600 冷阶段壁钟 15.18 s（日志内 13.87 s，含解析与加载）。 | 生产 stage 实测 |

## 生产默认

- `config.example.yaml` 与 factory 默认 `asr.batch_size: 20`。
- ASR 与对齐都走自适应批量；CUDA OOM 时自动减半（20→10→5→2→1），打印警告后重试，不改变阶段契约。
- 30 秒块、`max_new_tokens: 512`、0 重叠保持不变。
- R600 生产 stage 实测：`chunks=20 chunk_seconds=30 batch_size=20 segments=20 words=2695 time=13.87s`，wall 15.18 s，language=zh；transcript 时间戳单调且 `end > start`。

## 对 T2c 的输入

| 参照线 | 结果 | 说明 |
|--------|------|------|
| 20 RTFx | R30 20.09；R60 33.70；R480 111.11；R600 120.10 | **不是验收目标**；短音频是单块延迟下限 |
| 50 RTFx | 8–10 分钟真实中文已超过（111–120）；60 秒为 33.70 | 长音频达到目标 |
| 540 RTFx | 未达到；最好 147.95（合成）/ 120.10（真实） | 差距来自自回归 ASR 解码，本任务不引入未验证推理栈 |

冷阶段壁钟对短音频不友好（模型加载约 7–10 s）；长视频可摊薄。T2c 需要同时看热 RTFx 与冷壁钟。

## 复跑

```bash
cd /root/VideoLingo-refactor
PYTHON=/root/VideoLingo/.venv/bin/python
CACHE=/root/.cache/modelscope

# 真实 10 分钟，batch 20，热启动
$PYTHON -m bench.asr_rtfx run \
  --audio /tmp/t2b-real/en_real_600s.wav \
  --mode hot --batch-size 20 --language Chinese \
  --warmups 1 --runs 3 --cache-dir "$CACHE"
```

- 复盘批量曲线时改 `--batch-size`；复盘分块时加 `--chunk-seconds`。
- 真实音频不随仓库提交；缺失时按测试集章节的源路径与命令重建，源 SHA-256 必须一致。
- 合成 S600 由已提交的 `tests/fixtures/audio/zh_30s.wav` 循环生成，可完全复现。

## 证据

- `bench/results/t2b/`：R30/R60/R480/R600 批量、分块、生成上限，以及 S600 批量 2–20 的 JSON。
- `bench/results/t2a/zh-600s-hot.json`：S600 的 batch=1 基线。
- 数字来自 2026-10-21，GPU 为 RTX 3090 24 GB；其它进程占用约 9.5 GB。

## 失败判定

- 热 RTFx 复跑中位数偏离报告超过 ±10%，或显存峰值偏离超过 ±5%（约 ±300 MB）。
- 阶段契约改变（transcript 字段、时间戳规则、输入路径）。
- 默认批量改动导致静默截断、丢块或 transcript 不完整。
- OOM 回退未生效，或以 0 退出但产物不完整。
- 没有可复现命令/证据，或验证 agent 与实现 agent 相同。

## 独立验证

由未参与实现的 agent 执行：

1. 复跑 S600 batch 1、20，确认 RTFx 与显存落在容差内；S600 可完全复现。
2. 复跑 R600 batch 20（真实音频存在时），确认 120.10 附近；缺失真实音频则标记“未验证”。
3. 确认生产默认 batch=20、OOM 减半单测通过、阶段契约未变。
4. 跑全部测试，确认当前全套 57 passed。
5. 确认没有跑下游阶段或全流程。

## 验证记录

- Ticket: #5
- Stage: 性能（非阶段）
- Implementer: 实现会话
- Verifier: 独立验证 agent（未参与实现）
- Commit: 5b1ab64（实现 a74a3c8）
- Scope: 批量、分块、注意力、对齐分组、流水线重叠、生成上限
- Preconditions: RTX 3090 24 GB、旧 venv、ModelScope 缓存；GPU 另有约 9.5 GB 占用
- Config path: 不适用（benchmark CLI）；生产默认 `config.example.yaml`
- Fixtures: S600 由已提交夹具生成；R600 本地真实音频（源 SHA-256 已记录）
- Command: 见“复跑”小节；验证写入 `/tmp`
- Expected: S600 b1≈16.81、b20≈147.95；R600 b20≈120.10；生产默认 batch20；当前全套 57 tests（T2b 验证时 54；尾块修复后 57）
- Result: PASS
  - S600 b1：16.98（+1.0%），6062/6228 MB，2200 words
  - S600 b20：152.05（+2.8%），10293/13224 MB，2200 words
  - R600 b20：121.89（+1.5%），10300/13224 MB，2695 words
  - 生产 stage：exit 0，wall 15.08 s；日志 `chunks=20 chunk_seconds=30 batch_size=20 segments=20 words=2695`；language=zh；时间戳单调、`end > start`、0–599.920 s；只写 transcript.json
  - 默认 batch=20；OOM 8→4→2 回退单测通过；报告表与证据 JSON 一致
  - T2b 验证时 54 tests passed；尾块修复合并后当前全套 57；无下游或全流程；仓库未修改
- Exit code: 0
- Evidence: 上述 RTFx/显存/日志；`bench/results/t2b/`
- Verdict: PASS

验证备注：20 RTFx 未被当作验收线；通过依据是真实测试集上的相对提升与绝对吞吐。