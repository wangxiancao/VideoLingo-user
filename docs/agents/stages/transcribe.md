# 阶段验证：transcribe

## 范围

关联：Epic #1；T4 #8；spec #16；实现 commit `7b1c34c`。

本阶段负责：

- `media/audio.wav` → `transcript.json`。
- 官方 `ZhipuAI/GLM-ASR-Nano-2512` 转写 + `Qwen/Qwen3-ForcedAligner-0.6B-hf` 词级对齐。
- 30 秒分块、按配置批量、块偏移合并；当前不做块重叠。
- 每个非静音块生成一个 transcript segment，保留段文本和词级时间戳。
- 语言：配置显式指定，或从转写文本脚本自动判断（zh/en/ja/ko/ru）。

本阶段不负责：

- 自然句重切（T5 segment）。
- 翻译、字幕、烧录。
- 批量、分块、注意力等性能优化（T2b）；本阶段只要求契约正确。
- 全流程运行。

## 前置条件

- Python：`/root/VideoLingo/.venv/bin/python`（3.10.19）。
- GPU：RTX 3090；CUDA 可用。`torch` 2.11.0+cu129、`transformers` 5.14.1、`modelscope` 1.27.0、`soundfile`。
- 权重：T2a 已下载到 `/root/.cache/modelscope`；GLM-ASR 4,515,776,712 B（MIT），ForcedAligner 1,835,545,960 B（Apache-2.0）。权重不进入仓库。
- 输入音频必须是 16 kHz、单声道、16-bit PCM WAV；由 ingest 产出。
- 配置只读；使用 `asr.*`：

  ```yaml
  asr:
    backend: 'glm_asr'          # 仅支持 glm_asr，未知值显式失败
    language: 'auto'            # auto 或 zh/en/yue/fr/de/it/ja/ko/pt/ru/es
    chunk_seconds: 30
    batch_size: 1
    max_new_tokens: 512
    modelscope_cache: '~/.cache/modelscope'
  ```

- `language: auto` 只按脚本判断：汉字→zh，假名→ja，韩文→ko，西里尔→ru，拉丁→en。法语、德语、西语、意语、葡语与粤语请显式设置。

## 夹具

- `tests/fixtures/audio/zh_30s.wav`：T2a 提交的 30 秒普通话夹具，16 kHz mono；来源与生成命令见 `tests/fixtures/audio/README.md`。
- 60 秒回归夹具不提交，由 30 秒夹具生成：

  ```bash
  ffmpeg -y -stream_loop 1 -i tests/fixtures/audio/zh_30s.wav -t 60 \
    -ar 16000 -ac 1 -c:a pcm_s16le /tmp/zh_60s.wav
  ```

- 夹具是循环语音；只验证阶段契约，不评估 ASR 文本质量。

## 命令

```bash
cd /root/VideoLingo-refactor
PYTHON=/root/VideoLingo/.venv/bin/python
WORK=$(mktemp -d)
cp config.example.yaml "$WORK/config.yaml"
mkdir -p "$WORK/media"
cp tests/fixtures/audio/zh_30s.wav "$WORK/media/audio.wav"

$PYTHON -m pipeline.runner \
  --stages transcribe \
  --workdir "$WORK" \
  --config "$WORK/config.yaml"

# 契约检查
$PYTHON - <<PY
import json
data = json.load(open("$WORK/transcript.json"))
print(data["schema_version"], data["language"], len(data["segments"]))
words = [word for segment in data["segments"] for word in segment["words"]]
print(len(words))
print(all(pair["start"] <= nxt["start"] for pair, nxt in zip(words, words[1:])))
print(all(word["end"] > word["start"] for word in words))
print(max(word["end"] for word in words))
PY
```

60 秒与 batch=2：

```bash
cp /tmp/zh_60s.wav "$WORK/media/audio.wav"
# 可选：把 config.yaml 里的 asr.batch_size 改成 2
$PYTHON -m pipeline.runner --stages transcribe --workdir "$WORK" --config "$WORK/config.yaml"
```

失败路径：

```bash
# 缺权重：退出 1，指出 stage 与模型 ID，不写 transcript
EMPTY=$(mktemp -d)
printf "asr:\n  modelscope_cache: $EMPTY\n" > "$WORK/empty.yaml"
$PYTHON -m pipeline.runner --stages transcribe --workdir "$WORK" --config "$WORK/empty.yaml"

# 缺音频：退出 2
NOMEDIA=$(mktemp -d)
cp config.example.yaml "$NOMEDIA/config.yaml"
$PYTHON -m pipeline.runner --stages transcribe --workdir "$NOMEDIA" --config "$NOMEDIA/config.yaml"
```

## 期望

- 退出码 0；只生成 `transcript.json`，不生成下游产物。
- `schema_version = 1`；`language` 非空；`segments` 非空。
- 每个 segment 有非空 `text` 与非空 `words`；segment 时间戳单调、不重叠。
- 词级时间戳满足 `0 <= start < end <= 音频时长 + 0.5`，起点单调，顺序稳定。
- 非静音块对应一个 segment；块偏移正确（60 秒音频第二段从约 30 秒开始）。
- 对齐器返回的零长度单元扩展为 1 ms，保留词数，不产出 `end <= start`。
- 静音块被跳过并打印提示；全部静音时显式失败，不写空 transcript。
- `asr.language: auto` 在本夹具上得到 `zh`；显式语言覆盖优先。
- 未知 backend、缺权重、坏音频、缺音频均非零退出并指出阶段。
- 不运行 segment/translate/subtitle/burn 或全流程。

## 失败判定

以下任一情况为 FAIL：

- 退出码 0 但 `transcript.json` 缺失、为空，或 `segments` / `words` 为空。
- 时间戳回退、零长度、越过音频时长 + 0.5 秒，或块偏移未叠加。
- 语言字段为空，或显式语言被静默忽略。
- 缺权重/坏音频仍以 0 退出，或没有指出阶段。
- 失败后留下半成品 transcript。
- 运行命令实际执行了下游阶段或全流程。
- 没有可复现命令或证据。
- 实现 agent 与验证 agent 是同一个 agent。

## 证据

实现会话实测（2026-10-21，RTX 3090，权重在 ModelScope 缓存）：

| 输入 | 结果 | 退出码 | 备注 |
|------|------|--------|------|
| 30 s zh_30s.wav | 1 segment，110 words，language=zh，0.72–29.921 s | 0 | 9 个零长度单元扩展为 1 ms；总耗时约 12.0 s（含权重解析 1.3 s、加载 7.3 s、ASR 2.1 s、对齐 0.2 s） |
| 60 s（30 s ×2） | 2 segments，220 words；第二段 30.72–59.921 s | 0 | 块偏移正确；验证实测约 13.8 s |
| 60 s，batch_size=2 | 2 segments，220 words | 0 | 批量路径正确 |
| 缺权重（空缓存） | 无 transcript | 1 | `error: stage 'transcribe': model weights for ZhipuAI/GLM-ASR-Nano-2512 are not available ...` |
| 缺音频 | 无 transcript | 2 | `stage 'transcribe' requires 'audio' ... (produced by 'ingest')` |

- 测试：`52 passed`（T1 29 + T2a 6 + T3 4 + T4 13）。
- 性能基线仍引用 T2a 报告：30 s 热启动 RTFx 17.32、60 s 16.54、显存峰值 6062/6228 MB、batch 1；T4 不重新调优。
- 未跑全流程。

## 独立验证

由未参与实现的 agent 执行。建议复核：

1. 30 秒夹具经 runner 跑通；检查 language、segment/word 数量、词时间戳单调且不超过 30.5 s、只生成 transcript。
2. 60 秒夹具检查两块与第二块约 30 秒的偏移。
3. 空缓存配置运行，确认退出 1、stderr 含 `stage 'transcribe'` 与模型 ID、无 transcript。
4. 缺音频运行，确认退出 2。
5. 跑 `pytest` 四个测试文件，确认 52 passed。
6. 确认没有跑下游阶段或全流程。
7. 若 GPU、权重或依赖不可用，标记为“未验证”，不要猜 PASS。

## 验证记录

- Ticket: #8
- Stage: transcribe
- Implementer: 实现会话
- Verifier: 独立验证 agent（未参与实现）
- Commit: 8288c35（HEAD；实现 7b1c34c）
- Scope: 30 s / 60 s 音频 → transcript JSON；GLM-ASR + ForcedAligner
- Preconditions: RTX 3090、旧 venv、ModelScope 缓存、CJK/英文夹具
- Config path: 临时复制的 `config.example.yaml`
- Fixtures: `tests/fixtures/audio/zh_30s.wav`；按文档生成的 `/tmp/zh_60s.wav`
- Command: 见“命令”小节；验证全部写入 `/tmp`，未改仓库
- Expected: 退出 0；language=zh；词时间戳单调、`end > start`、在 30.5/60.5 s 内；缺权重退出 1；缺音频退出 2
- Result: PASS
  - 30 s：退出 0，wall 12.02 s；1 segment 0.72–29.921 s，110 words；9 个零长度单元扩展为 1 ms；无零长度词
  - 60 s：退出 0，wall 13.76 s；2 segments（0.72–29.921、30.72–59.921），220 words；最大 end 59.921 s
  - 缺权重：退出 1，stderr 含 `stage 'transcribe'` 与 GLM 模型 ID，无 transcript
  - 缺音频：退出 2，stderr 指出 stage 与 producer
  - 测试：52 passed（T1 29 + T2a 6 + T3 4 + T4 13）
  - 仅运行 transcribe；无下游或全流程；仓库保持干净
- Exit code: 0（30/60 s）；1（缺权重）；2（缺音频）
- Evidence: 上述计数/时间范围/耗时；pytest 输出；空缓存错误
- Verdict: PASS