# Video fixtures

## sample_5s.mp4

- Synthetic, no third-party rights. H.264 320x240 at 24 fps, AAC mono 48 kHz, 5.022 s.
- Source: FFmpeg `testsrc2` video and a 440 Hz sine tone.
- Command:

  ```bash
  ffmpeg -y -f lavfi -i testsrc2=size=320x240:rate=24 \
    -f lavfi -i sine=frequency=440:sample_rate=48000 \
    -t 5 -c:v libx264 -preset veryfast -pix_fmt yuv420p \
    -c:a aac -b:a 96k -shortest -movflags +faststart \
    tests/fixtures/video/sample_5s.mp4
  ```

- Used by ingest verification for both a local path and a local HTTP URL.
- It must stay short and small; do not replace it with real footage.