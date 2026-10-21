import json
import os
import shutil
import subprocess
from pathlib import Path

from pipeline.contracts import StageError

FONT_NAME = "Noto Sans CJK SC"
FONT_SIZE = 24
MAX_DURATION_DRIFT = 0.5


def _require_tool(name):
    path = shutil.which(name)
    if path is None:
        raise StageError(f"required tool not found on PATH: {name}")
    return path


def _run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise StageError(f"cannot run {command[0]}: {exc}") from exc


def _tail(text):
    return (text or "").strip()[-1500:]


def _probe(path):
    result = _run(
        [
            _require_tool("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise StageError(f"ffprobe cannot read {path}: {_tail(result.stderr)}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise StageError(f"ffprobe returned invalid JSON for {path}") from exc


def _streams(probe):
    streams = probe.get("streams")
    return streams if isinstance(streams, list) else []


def _duration(probe):
    try:
        duration = float((probe.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        raise StageError("media duration is unavailable") from None
    if duration <= 0:
        raise StageError("media duration must be positive")
    return duration


def _check_font():
    fc_match = shutil.which("fc-match")
    if fc_match is None:
        return
    result = _run([fc_match, FONT_NAME])
    if result.returncode != 0 or "Noto Sans CJK" not in result.stdout:
        raise StageError(f"CJK font is not available: {FONT_NAME}")


def _escape_filter_path(path):
    text = str(path)
    for char in ("\\", ":", "'", "[", "]", ","):
        text = text.replace(char, "\\" + char)
    return text


def run(ctx) -> None:
    source = ctx.path("source_video")
    subtitle = ctx.path("srt_target")
    output = ctx.path("output_video")
    if output.exists():
        output.unlink()

    _require_tool("ffmpeg")
    _check_font()
    source_probe = _probe(source)
    codecs = {stream.get("codec_type") for stream in _streams(source_probe)}
    if "video" not in codecs:
        raise StageError(f"source video has no video stream: {source}")
    if "audio" not in codecs:
        raise StageError(f"source video has no audio stream: {source}")
    source_duration = _duration(source_probe)

    temp = output.with_name(".output.tmp.mp4")
    if temp.exists():
        temp.unlink()
    filter_path = _escape_filter_path(subtitle)
    video_filter = (
        f"subtitles=filename='{filter_path}':"
        f"force_style='FontName={FONT_NAME},FontSize={FONT_SIZE},"
        "Outline=1,Shadow=0,Alignment=2,MarginV=24'"
    )
    result = _run(
        [
            _require_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            video_filter,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(temp),
        ]
    )
    if result.returncode != 0 or not temp.is_file() or temp.stat().st_size == 0:
        if temp.exists():
            temp.unlink()
        raise StageError(f"cannot burn subtitles: {_tail(result.stderr)}")

    try:
        output_probe = _probe(temp)
        output_codecs = {
            stream.get("codec_type") for stream in _streams(output_probe)
        }
        if not {"video", "audio"}.issubset(output_codecs):
            raise StageError("burned video is missing a video or audio stream")
        output_duration = _duration(output_probe)
        if abs(output_duration - source_duration) > MAX_DURATION_DRIFT:
            raise StageError(
                f"burned duration {output_duration:.3f}s differs from source "
                f"{source_duration:.3f}s"
            )
        os.replace(temp, output)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise