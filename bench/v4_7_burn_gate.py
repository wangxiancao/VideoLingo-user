"""V4-7 O3 burn gate: libx264 vs h264_nvenc on the same subtitle burn.

Burns one sample with both encoder paths (mirroring the stage's ffmpeg
invocation exactly), measures wall-clock and output size, then scores the
nvenc output against the x264 output with ffmpeg's SSIM filter (how much
the GPU encode deviates from the CPU reference render). One-off evidence
harness per spec #54 / ticket #57; not product code.

Usage:
    .venv/bin/python bench/v4_7_burn_gate.py \
        --source <video.mp4> --srt <target.srt> --out-dir bench/results/v4-7
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

from pipeline.stages.burn import (  # noqa: E402
    FONT_NAME,
    FONT_SIZE,
    _encoder_args,
    _escape_filter_path,
)

SSIM_FLOOR = 0.98


class _FakeConfig(dict):
    def __init__(self, encoder):
        super().__init__({"burn.encoder": encoder})
    def get(self, key, default=None):  # config-like dotted access
        return super().get(key, default)


def burn(source: Path, srt: Path, output: Path, encoder: str):
    video_filter = (
        f"subtitles=filename='{_escape_filter_path(srt)}':"
        f"force_style='FontName={FONT_NAME},FontSize={FONT_SIZE},"
        "Outline=1,Shadow=0,Alignment=2,MarginV=24'"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vf", video_filter,
        "-map", "0:v:0", "-map", "0:a:0",
        *_encoder_args(_FakeConfig(encoder)),
        "-c:a", "copy", "-movflags", "+faststart", str(output),
    ]
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    wall = time.perf_counter() - started
    if result.returncode != 0:
        raise SystemExit(f"burn failed with {encoder}: {result.stderr[-500:]}")
    return {"encoder": encoder, "wall_s": round(wall, 1),
            "bytes": output.stat().st_size}


def ssim(a: Path, b: Path) -> float:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(a), "-i", str(b),
         "-lavfi", "ssim", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    line = [l for l in result.stderr.splitlines() if "SSIM" in l]
    if not line:
        raise SystemExit("no SSIM summary in ffmpeg output")
    text = line[-1]
    import re
    match = re.search(r"All:([0-9.]+)", text)
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--srt", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x264_path = out / "gate-x264.mp4"
    nvenc_path = out / "gate-nvenc.mp4"

    x264 = burn(Path(args.source), Path(args.srt), x264_path, "libx264")
    nvenc = burn(Path(args.source), Path(args.srt), nvenc_path, "h264_nvenc")
    score = ssim(x264_path, nvenc_path)
    report = {
        "schema_version": 1, "tool": "v4_7_burn_gate",
        "source": str(args.source), "libx264": x264, "h264_nvenc": nvenc,
        "ssim_vs_x264": score, "ssim_floor": SSIM_FLOOR,
        "speedup": round(x264["wall_s"] / nvenc["wall_s"], 2),
        "pass": score >= SSIM_FLOOR,
    }
    (out / "burn-gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
