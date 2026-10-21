# Audio fixtures

## zh_30s.wav

- Source: `examples/example_zh.wav` from `zai-org/GLM-ASR`, Apache-2.0.
- Upstream URL: https://raw.githubusercontent.com/zai-org/GLM-ASR/main/examples/example_zh.wav
- Transform: the 6.0 s clip is repeated 5 times, trimmed to 30.0 s, 16 kHz mono PCM s16le.
- Command:

  ```bash
  curl -fsSL -o /tmp/example_zh.wav \
    https://raw.githubusercontent.com/zai-org/GLM-ASR/main/examples/example_zh.wav
  ffmpeg -y -stream_loop 4 -i /tmp/example_zh.wav -t 30 \
    -ar 16000 -ac 1 -c:a pcm_s16le tests/fixtures/audio/zh_30s.wav
  ```

- Language: Mandarin Chinese. The audio is a loop, so the transcript repeats; this is intentional for a stable performance fixture, not an ASR quality fixture.

Longer benchmark clips are generated from this file and are not committed:

```bash
ffmpeg -y -stream_loop 1 -i tests/fixtures/audio/zh_30s.wav -t 60 \
  -ar 16000 -ac 1 -c:a pcm_s16le /tmp/zh_60s.wav
ffmpeg -y -stream_loop 19 -i tests/fixtures/audio/zh_30s.wav -t 600 \
  -ar 16000 -ac 1 -c:a pcm_s16le /tmp/zh_600s.wav
```