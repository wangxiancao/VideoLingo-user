import json
import os
import shutil
import subprocess
from pathlib import Path

from pipeline.contracts import StageError

FONT_NAME = "Noto Sans CJK SC"
FONT_SIZE = 24
MAX_DURATION_DRIFT = 0.5
DEFAULT_ENCODER = "libx264"


def _encoder_args(config):
    """Video codec args for the burn, from `burn.encoder` (V4-7 O3).

    libx264 keeps the historical veryfast/crf18 path byte-for-byte;
    h264_nvenc uses the highest-quality preset with quality-based VBR
    (crf18-equivalent) for a large speedup at a small quality cost, gated
    by SSIM in the O3 evidence run. This box's ffmpeg SDK predates the
    p1-p7 preset names, so the classic "slow" (=hq 2 passes) is used.
    """
    encoder = str(config.get("burn.encoder", DEFAULT_ENCODER)).strip().lower()
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "slow",
                "-rc", "vbr", "-cq", "19", "-b:v", "0"]
    raise StageError(
        f"burn.encoder must be libx264 or h264_nvenc, got {encoder!r}"
    )


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


def _burn_single(source, video_filter, temp, encoder_args):
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
            *encoder_args,
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


def _chunk_count(config, duration):
    """burn.chunks, bounded so short videos never spawn useless workers."""
    raw = config.get("burn.chunks", 1)
    try:
        chunks = int(raw)
    except (TypeError, ValueError):
        raise StageError(f"burn.chunks must be an integer, got {raw!r}") from None
    if chunks < 1 or chunks > 8:
        raise StageError(f"burn.chunks must be 1..8, got {chunks}")
    if chunks > 1 and duration < chunks * 30:
        chunks = max(1, int(duration // 30))
    return chunks


def _keyframe_times(source):
    """Video keyframe timestamps (seconds), for exact tiling cuts."""
    result = _run(
        [
            _require_tool("ffprobe"), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,flags",
            "-of", "csv=p=0", str(source),
        ]
    )
    if result.returncode != 0:
        raise StageError(
            f"cannot list keyframes of {source}: {_tail(result.stderr)}"
        )
    times = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and "K" in parts[1].upper():
            try:
                times.append(float(parts[0]))
            except ValueError:
                continue
    if not times:
        raise StageError(f"no keyframes found in {source}")
    return times


def _burn_chunked(source, subtitle_path, video_filter, temp, chunks, encoder_args):
    """Keyframe-cut the source into N tiles, burn them in parallel, concat.

    Boundaries snap to exact keyframe timestamps so the stream-copy cuts
    tile without overlap; cuts keep ABSOLUTE timestamps (-copyts) so the
    one unmodified SRT renders correct cues in every tile; each burned tile
    is rebased to zero before a stream-copy concat. Subtitle rendering,
    decode and encode then run N-wide instead of serially.
    """
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    probe = _probe(source)
    duration = _duration(probe)
    keyframes = _keyframe_times(source)
    span = duration / chunks
    boundaries = [0.0]
    for index in range(1, chunks):
        target = span * index
        boundary = next((kf for kf in keyframes if kf >= target), None)
        if boundary is None or boundary <= boundaries[-1]:
            break  # fewer useful tiles than requested; burn what we have
        boundaries.append(boundary)

    with tempfile.TemporaryDirectory(prefix="burn-chunks-") as workdir:
        work = Path(workdir)

        # One segment-muxer pass tiles exactly at keyframes (no overlap, no
        # gap) while keeping absolute timestamps, so the unmodified SRT
        # renders correct cues in every tile.
        cut_times = ",".join(f"{kf:.3f}" for kf in boundaries[1:])
        result = _run(
            [
                _require_tool("ffmpeg"), "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(source),
                "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
                "-f", "segment", "-segment_times", cut_times,
                "-reset_timestamps", "0",
                str(work / "cut-%03d.mp4"),
            ]
        )
        if result.returncode != 0:
            raise StageError(
                f"cannot cut burn chunks: {_tail(result.stderr)}"
            )
        cuts = sorted(work.glob("cut-*.mp4"))
        if len(cuts) != len(boundaries):
            raise StageError(
                f"segment cut produced {len(cuts)} tiles, expected "
                f"{len(boundaries)}"
            )

        def burn(index, cut_path):
            burned = work / f"burned-{index}.mp4"
            result = _run(
                [
                    _require_tool("ffmpeg"), "-hide_banner", "-loglevel",
                    "error", "-y", "-i", str(cut_path),
                    "-vf", f"{video_filter},setpts=PTS-STARTPTS",
                    "-af", "asetpts=PTS-STARTPTS",
                    "-map", "0:v:0", "-map", "0:a:0",
                    *encoder_args,
                    "-c:a", "aac", "-b:a", "192k",
                    str(burned),
                ]
            )
            if result.returncode != 0 or not burned.is_file() \
                    or burned.stat().st_size == 0:
                raise StageError(
                    f"cannot burn chunk {index}: {_tail(result.stderr)}"
                )
            return burned

        with ThreadPoolExecutor(max_workers=chunks) as pool:
            burned_files = list(pool.map(
                lambda pair: burn(pair[0], pair[1]), enumerate(cuts)))

        listing = work / "concat.txt"
        listing.write_text(
            "\n".join(f"file '{path.name}'" for path in burned_files),
            encoding="utf-8",
        )
        result = _run(
            [
                _require_tool("ffmpeg"), "-hide_banner", "-loglevel", "error",
                "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", "-movflags", "+faststart", str(temp),
            ]
        )
        if result.returncode != 0 or not temp.is_file() or temp.stat().st_size == 0:
            if temp.exists():
                temp.unlink()
            raise StageError(
                f"cannot concatenate burned chunks: {_tail(result.stderr)}"
            )


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
    chunks = _chunk_count(ctx.config, source_duration)

    temp = output.with_name(".output.tmp.mp4")
    if temp.exists():
        temp.unlink()
    filter_path = _escape_filter_path(subtitle)
    video_filter = (
        f"subtitles=filename='{filter_path}':"
        f"force_style='FontName={FONT_NAME},FontSize={FONT_SIZE},"
        "Outline=1,Shadow=0,Alignment=2,MarginV=24'"
    )
    if chunks > 1:
        _burn_chunked(source, subtitle, video_filter, temp, chunks,
                      _encoder_args(ctx.config))
    else:
        _burn_single(source, video_filter, temp, _encoder_args(ctx.config))

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