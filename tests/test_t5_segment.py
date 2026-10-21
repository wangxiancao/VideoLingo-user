import json
import shutil
from pathlib import Path

from pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT = REPO_ROOT / "tests/fixtures/transcript/sample_6s.json"
EXAM = REPO_ROOT / "tests/fixtures/llm/segment_exam.json"
GAP_EXAM = REPO_ROOT / "tests/fixtures/llm/segment_exam_gap.json"


def _config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    return path


def _workdir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copyfile(TRANSCRIPT, workdir / "transcript.json")
    return workdir


def test_segment_uses_exam_answers_and_preserves_words(tmp_path):
    workdir = _workdir(tmp_path)

    assert run(
        "segment",
        workdir,
        _config(tmp_path),
        llm_fixture=EXAM,
    ) == 0

    data = json.loads((workdir / "segments.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert len(data["segments"]) == 3
    assert [segment["id"] for segment in data["segments"]] == [1, 2, 3]
    assert [segment["text"] for segment in data["segments"]] == [
        "Hello world.",
        "This is a test.",
        "Another sentence follows here.",
    ]
    assert [segment["start"] for segment in data["segments"]] == [0.0, 1.0, 3.0]
    assert [segment["end"] for segment in data["segments"]] == [1.0, 3.0, 5.0]

    words = [word for segment in data["segments"] for word in segment["words"]]
    assert [word["text"] for word in words] == [
        "Hello",
        "world.",
        "This",
        "is",
        "a",
        "test.",
        "Another",
        "sentence",
        "follows",
        "here.",
    ]
    assert all(word["end"] > word["start"] for word in words)


def test_segment_rejects_uncovered_words(tmp_path, capsys):
    workdir = _workdir(tmp_path)

    assert run(
        "segment",
        workdir,
        _config(tmp_path),
        llm_fixture=GAP_EXAM,
    ) == 1
    error = capsys.readouterr().err
    assert "stage 'segment'" in error
    assert "coverage" in error.lower() or "cover" in error.lower()
    assert not (workdir / "segments.json").exists()