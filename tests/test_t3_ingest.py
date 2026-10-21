import functools
import http.server
import subprocess
import threading
import wave
from pathlib import Path

import pytest

from pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_FIXTURE = REPO_ROOT / "tests/fixtures/video/sample_5s.mp4"


def _write_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    return path


def _wav_info(path):
    with wave.open(str(path), "rb") as wav:
        return {
            "channels": wav.getnchannels(),
            "sample_rate": wav.getframerate(),
            "sample_width": wav.getsampwidth(),
            "frames": wav.getnframes(),
            "duration": wav.getnframes() / wav.getframerate(),
        }


def _probe_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


@pytest.fixture
def http_fixture_server():
    handler = functools.partial(_QuietHandler, directory=str(VIDEO_FIXTURE.parent))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/{VIDEO_FIXTURE.name}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_ingest_local_file_creates_normalized_media(tmp_path):
    config = _write_config(tmp_path)
    workdir = tmp_path / "work"

    assert run("ingest", workdir, config, source=str(VIDEO_FIXTURE)) == 0

    source = workdir / "media/source.mp4"
    audio = workdir / "media/audio.wav"
    assert source.is_file() and source.stat().st_size > 0
    assert audio.is_file() and audio.stat().st_size > 0

    info = _wav_info(audio)
    assert info["channels"] == 1
    assert info["sample_rate"] == 16000
    assert info["sample_width"] == 2
    assert abs(info["duration"] - _probe_duration(source)) <= 0.5
    assert _probe_duration(source) == pytest.approx(_probe_duration(VIDEO_FIXTURE), abs=0.5)


def test_ingest_url_downloads_and_normalizes(tmp_path, http_fixture_server):
    config = _write_config(tmp_path)
    workdir = tmp_path / "work"

    assert run("ingest", workdir, config, source=http_fixture_server) == 0

    assert (workdir / "media/source.mp4").is_file()
    info = _wav_info(workdir / "media/audio.wav")
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1


def test_ingest_video_without_audio_fails_with_stage_name(tmp_path, capsys):
    config = _write_config(tmp_path)
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:d=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(silent),
        ],
        check=True,
    )
    workdir = tmp_path / "work"

    assert run("ingest", workdir, config, source=str(silent)) == 1
    error = capsys.readouterr().err
    assert "stage 'ingest'" in error
    assert "audio" in error.lower()
    assert not (workdir / "media/audio.wav").exists()


def test_ingest_rejects_unsupported_url_scheme(tmp_path):
    config = _write_config(tmp_path)

    assert run("ingest", tmp_path / "work", config, source="ftp://example.com/video.mp4") == 2