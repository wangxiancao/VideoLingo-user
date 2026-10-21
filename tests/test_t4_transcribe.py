import json
import shutil
from pathlib import Path

import pytest

from pipeline.asr.contracts import TranscriptSegment, Transcription, Word
from pipeline.asr.language import detect_language, resolve_language
from pipeline.contracts import ConfigError, StageError
from pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_FIXTURE = REPO_ROOT / "tests/fixtures/audio/zh_30s.wav"


def _write_config(tmp_path, text="{}\n"):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _prepare_workdir(tmp_path):
    workdir = tmp_path / "work"
    media = workdir / "media"
    media.mkdir(parents=True)
    shutil.copyfile(AUDIO_FIXTURE, media / "audio.wav")
    return workdir


def _fake_result():
    return Transcription(
        language="en",
        segments=(
            TranscriptSegment(
                start=0.0,
                end=1.0,
                text="Hello world.",
                words=(
                    Word(text="Hello", start=0.0, end=0.5),
                    Word(text="world.", start=0.5, end=1.0),
                ),
            ),
        ),
    )


class FakeBackend:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def transcribe(self, audio_path, language=None):
        self.calls.append((Path(audio_path), language))
        if self.error is not None:
            raise self.error
        return self.result


def _install_backend(monkeypatch, backend):
    import pipeline.stages.transcribe as transcribe_stage

    monkeypatch.setattr(transcribe_stage, "create_backend", lambda _config: backend)


def test_transcribe_writes_transcript_contract(tmp_path, monkeypatch):
    backend = FakeBackend(_fake_result())
    _install_backend(monkeypatch, backend)
    workdir = _prepare_workdir(tmp_path)

    assert run("transcribe", workdir, _write_config(tmp_path)) == 0

    data = json.loads((workdir / "transcript.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["language"] == "en"
    assert data["segments"][0]["text"] == "Hello world."
    assert [word["text"] for word in data["segments"][0]["words"]] == ["Hello", "world."]
    assert backend.calls == [(workdir / "media/audio.wav", None)]


def test_transcribe_forwards_configured_language(tmp_path, monkeypatch):
    backend = FakeBackend(_fake_result())
    _install_backend(monkeypatch, backend)
    workdir = _prepare_workdir(tmp_path)

    config = _write_config(tmp_path, "asr:\n  language: zh\n")
    assert run("transcribe", workdir, config) == 0
    assert backend.calls[0][1] == "zh"


def test_transcribe_backend_failure_names_stage_and_writes_nothing(tmp_path, monkeypatch, capsys):
    backend = FakeBackend(error=StageError("model weights are missing"))
    _install_backend(monkeypatch, backend)
    workdir = _prepare_workdir(tmp_path)

    assert run("transcribe", workdir, _write_config(tmp_path)) == 1
    error = capsys.readouterr().err
    assert "stage 'transcribe'" in error
    assert "model weights are missing" in error
    assert not (workdir / "transcript.json").exists()


def test_transcribe_rejects_empty_result(tmp_path, monkeypatch, capsys):
    backend = FakeBackend(Transcription(language="en", segments=()))
    _install_backend(monkeypatch, backend)
    workdir = _prepare_workdir(tmp_path)

    assert run("transcribe", workdir, _write_config(tmp_path)) == 1
    assert "segments" in capsys.readouterr().err.lower()
    assert not (workdir / "transcript.json").exists()


def test_transcribe_rejects_corrupt_audio(tmp_path, capsys):
    workdir = _prepare_workdir(tmp_path)
    (workdir / "media/audio.wav").write_bytes(b"not a wav file")

    assert run("transcribe", workdir, _write_config(tmp_path)) == 1
    error = capsys.readouterr().err
    assert "stage 'transcribe'" in error
    assert "audio" in error.lower()
    assert not (workdir / "transcript.json").exists()


def test_transcribe_rejects_unknown_backend(tmp_path, capsys):
    workdir = _prepare_workdir(tmp_path)
    config = _write_config(tmp_path, "asr:\n  backend: no_such_backend\n")

    assert run("transcribe", workdir, config) == 2
    error = capsys.readouterr().err
    assert "asr.backend" in error
    assert "no_such_backend" in error
    assert not (workdir / "transcript.json").exists()


def test_transcript_validator_rejects_end_past_audio(tmp_path, capsys):
    workdir = _prepare_workdir(tmp_path)
    payload = {
        "schema_version": 1,
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 31.0,
                "text": "Too long.",
                "words": [{"start": 0.0, "end": 31.0, "text": "Too long."}],
            }
        ],
    }
    (workdir / "transcript.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    assert run("transcribe", workdir, _write_config(tmp_path), validate_only=True) == 1
    assert "audio duration" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("你好，世界。", "zh"),
        ("Hello world.", "en"),
        ("こんにちは", "ja"),
        ("안녕하세요", "ko"),
        ("Привет", "ru"),
    ],
)
def test_detect_language_by_script(text, expected):
    assert detect_language(text) == expected


def test_resolve_language_explicit_and_auto():
    assert resolve_language("English", ["你好"]) == "en"
    assert resolve_language("auto", ["Hello world."]) == "en"
    with pytest.raises(ConfigError):
        resolve_language("klingon", ["Hello world."])