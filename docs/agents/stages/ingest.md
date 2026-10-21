# 阶段验证：ingest

## 范围

关联：Epic #1；T3 #7；spec #16；实现 commit `c3ad043`。

本阶段负责：

- 本地文件或 http(s) URL → `media/source.mp4` 与 `media/audio.wav`。
- `source.mp4` 优先流复制；容器/编码不兼容时回退到 H.264/AAC 转码。
- `audio.wav` 固定为 16 kHz、单声道、16-bit PCM。
- 产物先写临时目录，校验通过后再替换正式路径，失败不留下半成品。

本阶段不负责：

- 人声分离 / Demucs。
- 转写、分段、翻译、字幕与烧录。
- 分辨率选择、播放列表、字幕下载。
- 全流程运行。

## 前置条件

- Python：`/root/VideoLingo/.venv/bin/python`（3.10.19）。
- FFmpeg / ffprobe：4.2.7（本机）。必需。
- URL 输入：`yt-dlp` 2026.07.04（旧 venv 已有）；远程站点可能需要网络与 cookies。可选 `node`，存在时用作 YouTube JS runtime。
- 不需要 GPU、模型权重或 API key。
- 配置只读；本阶段不新增配置键。可选读取 `youtube.cookies_path`，文件存在时传给 yt-dlp。

## 夹具

- `tests/fixtures/video/sample_5s.mp4`：FFmpeg `testsrc2` + 440 Hz 正弦波，5.022 秒，H.264 320x240 + AAC mono 48 kHz。合成素材，无第三方版权。
- 生成命令与来源见 `tests/fixtures/video/README.md`。
- URL 验证用本地 HTTP 服务提供同一文件，不依赖外网。

## 命令

```bash
cd /root/VideoLingo-refactor
PYTHON=/root/VideoLingo/.venv/bin/python
WORK=$(mktemp -d)
cp config.example.yaml "$WORK/config.yaml"

# 本地文件
$PYTHON -m pipeline.runner \
  --stages ingest \
  --workdir "$WORK/local" \
  --config "$WORK/config.yaml" \
  --source tests/fixtures/video/sample_5s.mp4

# URL：本地 HTTP 服务
$PYTHON -m http.server 8765 --bind 127.0.0.1 \
  --directory tests/fixtures/video >/tmp/t3-http.log 2>&1 &
SERVER=$!
sleep 1
$PYTHON -m pipeline.runner \
  --stages ingest \
  --workdir "$WORK/url" \
  --config "$WORK/config.yaml" \
  --source http://127.0.0.1:8765/sample_5s.mp4
kill "$SERVER"

# 产物检查
ffprobe -v error -show_entries format=duration \
  -show_entries stream=codec_type,sample_rate,channels \
  -of json "$WORK/local/media/audio.wav"
ffprobe -v error -show_entries format=duration \
  -of default=nw=1:nk=1 "$WORK/local/media/source.mp4"
```

失败路径：

```bash
# 无音轨视频：退出 1，stderr 包含 stage 'ingest' 与 audio
ffmpeg -y -v error -f lavfi -i color=c=blue:s=160x120:d=2 \
  -c:v libx264 -pix_fmt yuv420p -an "$WORK/silent.mp4"
$PYTHON -m pipeline.runner --stages ingest --workdir "$WORK/fail" \
  --config "$WORK/config.yaml" --source "$WORK/silent.mp4"

# 不支持的 scheme：退出 2
$PYTHON -m pipeline.runner --stages ingest --workdir "$WORK/fail2" \
  --config "$WORK/config.yaml" --source ftp://example.com/a.mp4
```

## 期望

- 退出码 0；`media/source.mp4` 与 `media/audio.wav` 均存在且非空。
- `audio.wav`：1 声道、16000 Hz、16-bit PCM、帧数大于 0。
- 音频时长与 `source.mp4` 时长相差不超过 0.5 秒；`source.mp4` 含视频流。
- `source.mp4` 保留源视频内容；本机夹具为流复制，URL 与本地产物大小一致。
- 工作区根下不残留 `.ingest-*` 临时目录。
- 失败时退出码非零，错误信息包含 `stage 'ingest'`；无音轨视频不产出音频。
- 不做人声分离；没有 demucs 调用。

## 失败判定

以下任一情况为 FAIL：

- 产物缺失、为空，或路径不是 `media/source.mp4` / `media/audio.wav`。
- 音频不是 16 kHz 单声道 16-bit PCM。
- 音频与视频时长相差超过 0.5 秒。
- 无音轨输入仍以 0 退出，或没有指出阶段。
- URL 下载失败仍以 0 退出，或静默改用未声明来源。
- 失败后留下 `.ingest-*` 临时目录或半成品音频。
- 运行命令实际执行了 transcribe、segment、translate、subtitle、burn 或全流程。
- 没有可复现命令或证据。
- 实现 agent 与验证 agent 是同一个 agent。

## 证据

实现会话实测（2026-10-21）：

| 输入 | 退出码 | source.mp4 | audio.wav | 视频时长 | 音频时长 | 音频格式 |
|------|--------|------------|-----------|----------|----------|----------|
| 本地 `sample_5s.mp4` | 0 | 225,831 B | 160,504 B | 5.022 s | 5.013 s | 1 ch / 16000 Hz / 16-bit |
| 本地 HTTP URL | 0 | 225,831 B | 160,504 B | 5.022 s | 5.013 s | 1 ch / 16000 Hz / 16-bit |
| 无音轨视频 | 1 | 不产出 | 不产出 | — | — | — |
| `ftp://` URL | 2 | 不产出 | 不产出 | — | — | — |

- 无音轨错误：`error: stage 'ingest': source has no audio stream: ...`
- 不支持 scheme 错误：`error: stage 'ingest': unsupported source URL scheme: ftp`
- 测试：`39 passed`（T1 29 + T2a 6 + T3 4）。
- 远程 YouTube / 外网 URL 未验证；本地 HTTP URL 已验证下载路径。

## 独立验证

由未参与实现的 agent 执行。建议复核：

1. 用本地文件与本地 HTTP URL 各跑一次，检查产物路径、大小、时长与 WAV 头。
2. 跑无音轨视频，确认退出 1 且 stderr 含 `stage 'ingest'` 与 `audio`。
3. 跑 `ftp://`，确认退出 2。
4. 确认工作区没有 `.ingest-*` 残留。
5. 确认没有调用 demucs、没有跑下游阶段或全流程。

## 验证记录

```markdown
## 验证记录

- Ticket: #7
- Stage: ingest
- Implementer: 实现会话
- Verifier:
- Commit: c3ad043
- Scope: 本地文件 + URL → 规范化视频与 16k mono 音频
- Preconditions:
- Config path: 临时复制的 config.example.yaml
- Fixtures: tests/fixtures/video/sample_5s.mp4
- Command:
- Expected:
- Result:
- Exit code:
- Evidence:
- Verdict: PASS / FAIL
```