import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from urllib.parse import urlparse

from pipeline.contracts import MissingInputError, StageError, UsageError

MEDIA_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".flv",
    ".wmv",
    ".m4v",
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".opus",
    ".ogg",
}


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


def _probe(path):
    result = _run(
        [
            _require_tool("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,sample_rate,channels,width,height",
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


def _tail(text):
    text = (text or "").strip()
    return text[-1500:]


def _streams(probe):
    streams = probe.get("streams")
    return streams if isinstance(streams, list) else []


def _has_video(probe):
    return any(stream.get("codec_type") == "video" for stream in _streams(probe))


def _has_audio(probe):
    return any(stream.get("codec_type") == "audio" for stream in _streams(probe))


def _duration(probe):
    value = (probe.get("format") or {}).get("duration")
    try:
        duration = float(value)
    except (TypeError, ValueError):
        raise StageError("media duration is unavailable") from None
    if duration <= 0:
        raise StageError("media duration must be positive")
    return duration


def _replace(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    try:
        source.replace(target)
    except OSError as exc:
        raise StageError(f"cannot replace {target}: {exc}") from exc


def _normalize_video(source, target):
    ffmpeg = _require_tool("ffmpeg")
    target = Path(target)
    if target.exists():
        target.unlink()

    copy_result = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    if copy_result.returncode == 0 and target.is_file() and target.stat().st_size > 0:
        return

    if target.exists():
        target.unlink()
    transcode_result = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    if transcode_result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        detail = _tail(transcode_result.stderr) or _tail(copy_result.stderr)
        raise StageError(f"cannot normalize video to MP4: {detail}")


def _extract_audio(source, target):
    ffmpeg = _require_tool("ffmpeg")
    target = Path(target)
    if target.exists():
        target.unlink()
    result = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    )
    if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        raise StageError(f"cannot extract 16 kHz mono audio: {_tail(result.stderr)}")


def _download(url, directory, cookies_path=None):
    try:
        import yt_dlp
    except ImportError as exc:
        raise StageError(
            "yt-dlp is required for URL sources; install it into the pipeline interpreter"
        ) from exc

    options = {
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(Path(directory) / "download.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": True,
        "socket_timeout": 30,
    }
    if shutil.which("node"):
        options["js_runtimes"] = {"node": {}}
    if cookies_path:
        cookie_file = Path(str(cookies_path)).expanduser()
        if cookie_file.is_file():
            options["cookiefile"] = str(cookie_file)

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        raise StageError(f"download failed for {url}: {_tail(str(exc))}") from exc

    files = [
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
    ]
    if not files:
        raise StageError(f"download produced no media file for {url}")
    return max(files, key=lambda path: path.stat().st_size)


def _validate_outputs(video_path, audio_path):
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise StageError(f"normalized video is missing or empty: {video_path}")
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise StageError(f"extracted audio is missing or empty: {audio_path}")

    probe = _probe(video_path)
    if not _has_video(probe):
        raise StageError(f"normalized video has no video stream: {video_path}")
    video_duration = _duration(probe)

    try:
        with wave.open(str(audio_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
    except (OSError, wave.Error) as exc:
        raise StageError(f"cannot read extracted audio {audio_path}: {exc}") from exc

    if channels != 1:
        raise StageError(f"extracted audio must be mono, found {channels} channels")
    if sample_rate != 16000:
        raise StageError(f"extracted audio must be 16 kHz, found {sample_rate} Hz")
    if sample_width != 2:
        raise StageError(f"extracted audio must be 16-bit PCM, found {sample_width * 8}-bit")
    if frames <= 0:
        raise StageError("extracted audio is empty")

    audio_duration = frames / sample_rate
    if abs(audio_duration - video_duration) > 0.5:
        raise StageError(
            f"audio duration {audio_duration:.3f}s does not match video duration {video_duration:.3f}s"
        )


def run(ctx) -> None:
    source = ctx.source
    if not source:
        raise MissingInputError("ingest requires --source")
    if "://" in source:
        scheme = urlparse(source).scheme.lower()
        if scheme not in ("http", "https"):
            raise UsageError(f"unsupported source URL scheme: {scheme or source}")
    else:
        local = Path(source)
        if not local.is_file():
            raise MissingInputError(f"source not found: {local}")

    ctx.workdir.mkdir(parents=True, exist_ok=True)
    cookies_path = ctx.config.get("youtube.cookies_path")
    with tempfile.TemporaryDirectory(prefix=".ingest-", dir=ctx.workdir) as temp_dir:
        if "://" in source:
            input_path = _download(source, temp_dir, cookies_path)
        else:
            input_path = Path(source).resolve()

        probe = _probe(input_path)
        if not _has_video(probe):
            raise StageError(f"source has no video stream: {input_path}")
        if not _has_audio(probe):
            raise StageError(f"source has no audio stream: {input_path}")

        temp_video = Path(temp_dir) / "source.mp4"
        temp_audio = Path(temp_dir) / "audio.wav"
        _normalize_video(input_path, temp_video)
        _extract_audio(temp_video, temp_audio)
        _validate_outputs(temp_video, temp_audio)

        _replace(temp_video, ctx.path("source_video"))
        _replace(temp_audio, ctx.path("audio"))