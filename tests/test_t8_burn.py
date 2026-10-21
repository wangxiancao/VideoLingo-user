import json
import shutil
import subprocess
from pathlib import Path

from pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO = REPO_ROOT / "tests/fixtures/video/sample_5s.mp4"
SOURCE_SRT = REPO_ROOT / "tests/fixtures/srt/sample_5s_source.srt"
TARGET_SRT = REPO_ROOT / "tests/fixtures/srt/sample_5s_target.srt"


def _config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    return path


def _workdir(tmp_path):
    workdir = tmp_path / "work"
    media = workdir / "media"
    subs = workdir / "subs"
    media.mkdir(parents=True)
    subs.mkdir(parents=True)
    shutil.copyfile(VIDEO, media / "source.mp4")
    shutil.copyfile(SOURCE_SRT, subs / "source.srt")
    shutil.copyfile(TARGET_SRT, subs / "target.srt")
    return workdir


def _probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_burn_writes_a_video_with_subtitles(tmp_path):
    workdir = _workdir(tmp_path)

    assert run("burn", workdir, _config(tmp_path)) == 0

    output = workdir / "output.mp4"
    assert output.is_file() and output.stat().st_size > 0
    output_probe = _probe(output)
    source_probe = _probe(VIDEO)
    assert {stream["codec_type"] for stream in output_probe["streams"]} == {"video", "audio"}
    assert abs(
        float(output_probe["format"]["duration"]) - float(source_probe["format"]["duration"])
    ) <= 0.5


def test_burn_rejects_corrupt_video_without_output(tmp_path, capsys):
    workdir = _workdir(tmp_path)
    (workdir / "media/source.mp4").write_bytes(b"not a video")

    assert run("burn", workdir, _config(tmp_path)) == 1
    error = capsys.readouterr().err
    assert "stage 'burn'" in error
    assert "ffprobe" in error.lower() or "cannot read" in error.lower()
    assert not (workdir / "output.mp4").exists()