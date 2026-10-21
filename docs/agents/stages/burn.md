# 阶段验证：burn

## 范围

关联：Epic #1；T8 #12；spec #16；实现 commit 待填。

本阶段负责：

- `media/source.mp4` + `subs/target.srt` → `output.mp4`。
- 使用 FFmpeg 的 `subtitles` 滤镜与 Noto Sans CJK SC 字体烧录译文。
- 输出保持源视频时长与音视频流；失败显式报错，不生成黑色占位视频。

本阶段不负责：

- 字幕生成（T7）；`subs/source.srt` 作为契约输入存在，但本阶段渲染的是 target。
- 转码策略、硬件编码或多语言字幕选择。
- 全流程运行。

## 前置条件

- Python：`/root/VideoLingo/.venv/bin/python`。
- FFmpeg / ffprobe：4.2.7（本机）。
- 字体：`fc-match "Noto Sans CJK SC"` 必须匹配到 Noto Sans CJK；缺失时显式失败。
- 输入：5 秒 MP4（含音频）与合法 SRT；不需要 GPU/模型/LLM。
- 不新增配置键。

## 夹具

- `tests/fixtures/video/sample_5s.mp4`：H.264 320x240 + AAC，5.022 s。
- `tests/fixtures/srt/sample_5s_source.srt`：英文源字幕。
- `tests/fixtures/srt/sample_5s_target.srt`：中文译文 `中文字幕渲染测试`，0.5–4.5 s。

## 命令

```bash
cd /root/VideoLingo-refactor
PYTHON=/root/VideoLingo/.venv/bin/python
WORK=$(mktemp -d)
cp config.example.yaml "$WORK/config.yaml"
mkdir -p "$WORK/media" "$WORK/subs"
cp tests/fixtures/video/sample_5s.mp4 "$WORK/media/source.mp4"
cp tests/fixtures/srt/sample_5s_source.srt "$WORK/subs/source.srt"
cp tests/fixtures/srt/sample_5s_target.srt "$WORK/subs/target.srt"

$PYTHON -m pipeline.runner --stages burn \
  --workdir "$WORK" --config "$WORK/config.yaml"

ffprobe -v error -show_entries format=duration \
  -show_entries stream=codec_type,codec_name,width,height \
  -of json "$WORK/output.mp4"
ffmpeg -y -v error -ss 1.5 -i "$WORK/output.mp4" -frames:v 1 "$WORK/frame.png"
```

## 期望

- 退出 0；只生成 `output.mp4`。
- 输出含 H.264 视频流与 AAC 音频流；时长与源视频相差 ≤0.5 s。
- 中文字幕可见，CJK 字形正常，无方块。
- 抽帧 1.5 s 可见“中文字幕渲染测试”。
- 源视频损坏、缺流、字体缺失或 FFmpeg 失败时退出非零，且不保留 `output.mp4` 或临时输出。
- 不运行 ingest/transcribe/subtitle 或全流程。

## 失败判定

- 输出缺失/为空，或没有音视频流。
- 时长偏差 >0.5 s。
- 字幕不可见、CJK 显示为方块，或输出是黑屏占位。
- FFmpeg/字体失败仍以 0 退出，或留下半成品。
- 运行了前序阶段或全流程。
- 没有可复现命令/证据，或实现 agent 与验证 agent 相同。

## 证据

- 实测：`62 passed`（60 + T8 2）。
- Runner 输出：`output.mp4` 5.022 s，H.264 320x240 + AAC；抽帧 1.5 s 可见中文字幕。
- 损坏源视频：退出 1，无 `output.mp4`。

## 独立验证

由未参与实现的 agent 执行：

1. 用夹具运行 burn，确认退出 0、`output.mp4` 存在。
2. ffprobe 检查音视频流与时长差 ≤0.5 s。
3. 抽 1.5 s 帧并目视确认中文正常、无方块。
4. 用损坏的 source.mp4 运行，确认退出 1 且无输出。
5. 确认只跑 burn，未跑前序阶段或全流程。
6. 跑全部测试，确认 62 passed。

## 验证记录

```markdown
## 验证记录

- Ticket: #12
- Stage: burn
- Implementer: 实现会话
- Verifier:
- Commit:
- Scope: source.mp4 + target.srt → output.mp4
- Preconditions:
- Config path:
- Fixtures:
- Command:
- Expected:
- Result:
- Exit code:
- Evidence:
- Verdict: PASS / FAIL
```