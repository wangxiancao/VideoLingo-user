"""T5 segment stage tests: rule-based segmentation (V4-3, #32 decision).

Boundaries come from terminal punctuation, silence gaps and length caps —
deterministic, no LLM. The sample transcript is gap-less with punctuated
words, so the expected boundaries match what the old LLM exam produced.
"""

import json
import shutil
from pathlib import Path

from pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT = REPO_ROOT / "tests/fixtures/transcript/sample_6s.json"


def _config(tmp_path, extra=""):
    path = tmp_path / "config.yaml"
    path.write_text(extra or "{}\n", encoding="utf-8")
    return path


def _workdir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copyfile(TRANSCRIPT, workdir / "transcript.json")
    return workdir


def _transcript_with_words(words, language="en"):
    return {
        "schema_version": 1,
        "language": language,
        "segments": [
            {
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": " ".join(word["text"] for word in words),
                "words": words,
            }
        ],
    }


def _write_transcript(workdir, payload):
    (workdir / "transcript.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_segment_rules_cut_on_punctuation_and_preserve_words(tmp_path):
    workdir = _workdir(tmp_path)

    assert run("segment", workdir, _config(tmp_path)) == 0

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
        "Hello", "world.", "This", "is", "a", "test.",
        "Another", "sentence", "follows", "here.",
    ]
    assert all(word["end"] > word["start"] for word in words)


def test_segment_cuts_on_silence_gap(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    words = [
        {"start": 0.0, "end": 0.5, "text": "words"},
        {"start": 0.5, "end": 1.0, "text": "flow"},
        {"start": 2.0, "end": 2.5, "text": "then"},
        {"start": 2.5, "end": 3.0, "text": "silence"},
    ]
    _write_transcript(workdir, _transcript_with_words(words))

    assert run("segment", workdir, _config(tmp_path)) == 0
    data = json.loads((workdir / "segments.json").read_text(encoding="utf-8"))
    assert [segment["text"] for segment in data["segments"]] == [
        "words flow", "then silence"
    ]


def test_segment_caps_sentence_length(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    words = [
        {"start": 0.1 * index, "end": 0.1 * index + 0.05, "text": f"w{index}"}
        for index in range(30)
    ]
    _write_transcript(workdir, _transcript_with_words(words))

    assert run(
        "segment", workdir, _config(tmp_path, "segmentation:\n  max_words: 10\n")
    ) == 0
    data = json.loads((workdir / "segments.json").read_text(encoding="utf-8"))
    sizes = [len(segment["words"]) for segment in data["segments"]]
    assert sizes == [10, 10, 10]
    joined = [word for segment in data["segments"] for word in segment["words"]]
    assert joined == words


def test_segment_respects_pause_config(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    words = [
        {"start": 0.0, "end": 0.5, "text": "a"},
        {"start": 0.6, "end": 1.1, "text": "b"},
        {"start": 1.2, "end": 1.7, "text": "c"},
    ]
    _write_transcript(workdir, _transcript_with_words(words))

    assert run(
        "segment", workdir, _config(tmp_path, "segmentation:\n  pause_s: 0.05\n")
    ) == 0
    data = json.loads((workdir / "segments.json").read_text(encoding="utf-8"))
    assert [segment["text"] for segment in data["segments"]] == ["a", "b", "c"]


def test_segment_joins_cjk_without_spaces(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    words = [
        {"start": 0.0, "end": 0.4, "text": "今天"},
        {"start": 0.4, "end": 0.8, "text": "天气"},
        {"start": 2.0, "end": 2.4, "text": "很好"},
    ]
    _write_transcript(workdir, _transcript_with_words(words, language="zh"))

    assert run("segment", workdir, _config(tmp_path)) == 0
    data = json.loads((workdir / "segments.json").read_text(encoding="utf-8"))
    assert [segment["text"] for segment in data["segments"]] == ["今天天气", "很好"]


def test_segment_rejects_empty_transcript(tmp_path, capsys):
    workdir = tmp_path / "work"
    workdir.mkdir()
    _write_transcript(
        workdir,
        {"schema_version": 1, "language": "en", "segments": [
            {"start": 0.0, "end": 0.0, "text": "", "words": []}
        ]},
    )

    assert run("segment", workdir, _config(tmp_path)) == 1
    error = capsys.readouterr().err
    assert "stage 'segment'" in error
    # The input contract rejects the degenerate transcript before the stage
    # runs; either way the run fails explicitly and writes no output.
    assert "invalid" in error
    assert not (workdir / "segments.json").exists()
