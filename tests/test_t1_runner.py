import hashlib
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

from pipeline.codec import JsonCodec
from pipeline.config import ReadOnlyConfig
from pipeline.contracts import ContractError, StageContext, StageSpec
from pipeline.registry import ARTIFACTS, STAGES, validate_registry
from pipeline.runner import run
from pipeline.validators import validate_artifact, validate_srt_set

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _write_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    return path


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_wav(path, seconds=2.0, rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))


def _context(tmp_path, config_path, workdir=None, **kwargs):
    workdir = Path(workdir) if workdir else tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        workdir=workdir,
        config=ReadOnlyConfig(config_path),
        source=kwargs.get("source"),
        llm_fixture=kwargs.get("llm_fixture"),
        validate_only=kwargs.get("validate_only", False),
        artifacts=ARTIFACTS,
        codec=JsonCodec(),
    )


def _noop_run(ctx):
    (ctx.workdir / "noop.txt").write_text("ok", encoding="utf-8")


NOOP = StageSpec("noop", "noop", (), (), (), _noop_run)


def _valid_transcript():
    return {
        "schema_version": 1,
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello world.",
                "words": [
                    {"start": 0.0, "end": 0.5, "text": "Hello"},
                    {"start": 0.5, "end": 1.0, "text": "world"},
                ],
            }
        ],
    }


def _valid_segments():
    return {
        "schema_version": 1,
        "language": "en",
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 1.0,
                "text": "Hello world.",
                "words": [
                    {"start": 0.0, "end": 0.5, "text": "Hello"},
                    {"start": 0.5, "end": 1.0, "text": "world"},
                ],
            },
            {
                "id": 2,
                "start": 1.0,
                "end": 2.0,
                "text": "Goodbye.",
                "words": [{"start": 1.0, "end": 2.0, "text": "Goodbye."}],
            },
        ],
    }


def _valid_translation():
    return {
        "schema_version": 1,
        "source_language": "en",
        "target_language": "zh-CN",
        "segments": [
            {"id": 1, "start": 0.0, "end": 1.0, "source": "Hello world.", "target": "你好，世界。"},
            {"id": 2, "start": 1.0, "end": 2.0, "source": "Goodbye.", "target": "再见。"},
        ],
    }


def _valid_srt():
    return (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Hello\n"
        "\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "World\n"
    )


def _write_srt_set(workdir, source=None, target=None, bilingual=None):
    source = source if source is not None else _valid_srt()
    target = target if target is not None else _valid_srt().replace("Hello", "你好").replace("World", "世界")
    bilingual = bilingual if bilingual is not None else _valid_srt().replace("Hello", "Hello\n你好").replace("World", "World\n世界")
    for key, text in (
        ("srt_source", source),
        ("srt_target", target),
        ("srt_bilingual", bilingual),
    ):
        path = workdir / ARTIFACTS[key].relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_noop_runs_without_wiping(tmp_path):
    config = _write_config(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    sentinel = workdir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    assert run("noop", workdir, config, stages=(NOOP,)) == 0
    assert (workdir / "noop.txt").read_text(encoding="utf-8") == "ok"
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_unknown_stage_returns_2(tmp_path):
    config = _write_config(tmp_path)
    assert run("nope", tmp_path / "work", config, stages=(NOOP,)) == 2


def test_empty_stages_returns_2(tmp_path):
    config = _write_config(tmp_path)
    assert run("", tmp_path / "work", config, stages=(NOOP,)) == 2
    assert run([], tmp_path / "work", config, stages=(NOOP,)) == 2


def test_duplicate_stages_returns_2(tmp_path):
    config = _write_config(tmp_path)
    assert run("noop,noop", tmp_path / "work", config, stages=(NOOP,)) == 2


def test_missing_config_returns_2(tmp_path):
    missing = tmp_path / "missing.yaml"
    assert run("noop", tmp_path / "work", missing, stages=(NOOP,)) == 2


def test_ingest_requires_source(tmp_path):
    config = _write_config(tmp_path)
    assert run("ingest", tmp_path / "work", config) == 2


def test_source_without_ingest_returns_2(tmp_path):
    config = _write_config(tmp_path)
    assert run(
        "noop",
        tmp_path / "work",
        config,
        source="http://example.com/a.mp4",
        stages=(NOOP,),
    ) == 2


def test_missing_prerequisite_does_not_auto_run(tmp_path):
    config = _write_config(tmp_path)
    workdir = tmp_path / "work"
    assert run("segment", workdir, config) == 2
    assert not (workdir / "segments.json").exists()
    assert not (workdir / "transcript.json").exists()


def test_corrupt_input_fails_before_stage(tmp_path, capsys):
    config = _write_config(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    _write_json(
        workdir / "transcript.json",
        {"schema_version": 1, "language": "en", "segments": []},
    )
    assert run("segment", workdir, config) == 1
    error = capsys.readouterr().err
    assert "segment" in error
    assert "transcript" in error
    assert "not implemented" not in error


def test_table_order_not_input_order(tmp_path):
    config = _write_config(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()

    def run_a(ctx):
        (ctx.workdir / "order.txt").write_text("a", encoding="utf-8")

    def run_b(ctx):
        path = ctx.workdir / "order.txt"
        path.write_text(path.read_text(encoding="utf-8") + "b", encoding="utf-8")

    stage_a = StageSpec("a", "a", (), (), (), run_a)
    stage_b = StageSpec("b", "b", (), (), (), run_b)
    assert run("b,a", workdir, config, stages=(stage_a, stage_b)) == 0
    assert (workdir / "order.txt").read_text(encoding="utf-8") == "ab"


def test_stage_failure_returns_1_and_names_stage(tmp_path, capsys):
    config = _write_config(tmp_path)

    def fail(_ctx):
        raise RuntimeError("boom")

    bad = StageSpec("bad", "bad", (), (), (), fail)
    assert run("bad", tmp_path / "work", config, stages=(bad,)) == 1
    assert "bad" in capsys.readouterr().err


def test_stage_pipeline_error_includes_stage_id(tmp_path, capsys):
    config = _write_config(tmp_path)

    def fail(_ctx):
        raise ContractError("bad input")

    stage = StageSpec("contract", "contract", (), (), (), fail)
    assert run("contract", tmp_path / "work", config, stages=(stage,)) == 1
    error = capsys.readouterr().err
    assert "stage 'contract'" in error
    assert "bad input" in error


def test_stub_not_implemented_returns_1(tmp_path, capsys):
    config = _write_config(tmp_path)
    assert run(
        "ingest",
        tmp_path / "work",
        config,
        source="http://example.com/a.mp4",
    ) == 1
    assert "ingest" in capsys.readouterr().err


def test_validate_only_does_not_execute(tmp_path):
    config = _write_config(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    _write_wav(workdir / "media" / "audio.wav")
    transcript = workdir / "transcript.json"
    _write_json(transcript, _valid_transcript())
    mtime = transcript.stat().st_mtime_ns

    def should_not_run(_ctx):
        raise AssertionError("stage ran during validate-only")

    emit = StageSpec("emit", "emit", (), ("transcript",), (), should_not_run)
    assert run("emit", workdir, config, validate_only=True, stages=(emit,)) == 0
    assert transcript.stat().st_mtime_ns == mtime


def test_validate_only_missing_output_returns_1(tmp_path):
    config = _write_config(tmp_path)

    def should_not_run(_ctx):
        raise AssertionError("stage ran during validate-only")

    emit = StageSpec("emit", "emit", (), ("transcript",), (), should_not_run)
    assert run("emit", tmp_path / "work", config, validate_only=True, stages=(emit,)) == 1


def test_validate_only_rejects_malformed_output(tmp_path):
    config = _write_config(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    _write_json(
        workdir / "transcript.json",
        {"schema_version": 1, "language": "en", "segments": []},
    )

    def should_not_run(_ctx):
        raise AssertionError("stage ran during validate-only")

    emit = StageSpec("emit", "emit", (), ("transcript",), (), should_not_run)
    assert run("emit", workdir, config, validate_only=True, stages=(emit,)) == 1


def test_validate_only_ingest_without_source(tmp_path):
    config = _write_config(tmp_path)
    workdir = tmp_path / "work"
    (workdir / "media").mkdir(parents=True)
    (workdir / "media" / "source.mp4").write_bytes(b"x")
    (workdir / "media" / "audio.wav").write_bytes(b"x")
    assert run("ingest", workdir, config, validate_only=True) == 0


def test_validate_only_missing_input_returns_2(tmp_path):
    config = _write_config(tmp_path)

    def should_not_run(_ctx):
        raise AssertionError("stage ran during validate-only")

    emit = StageSpec("emit", "emit", ("audio",), ("transcript",), (), should_not_run)
    assert run("emit", tmp_path / "work", config, validate_only=True, stages=(emit,)) == 2


def test_transcript_validator_accepts_valid(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    _write_wav(ctx.path("audio"))
    ctx.write("transcript", _valid_transcript())
    validate_artifact(ctx, "transcript")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["segments"][0].update(start=1.0, end=0.5),
        lambda data: data["segments"][0]["words"][0].update(start=1.5),
        lambda data: data["segments"][0].update(text=""),
        lambda data: data.pop("schema_version"),
        lambda data: data["segments"].append(
            {
                "start": 0.5,
                "end": 1.5,
                "text": "Overlap.",
                "words": [{"start": 0.5, "end": 1.5, "text": "Overlap."}],
            }
        ),
    ],
)
def test_transcript_validator_rejects_bad(tmp_path, mutate):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    _write_wav(ctx.path("audio"))
    data = _valid_transcript()
    mutate(data)
    ctx.write("transcript", data)
    with pytest.raises(ContractError):
        validate_artifact(ctx, "transcript")


def test_transcript_validator_rejects_audio_overrun(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    _write_wav(ctx.path("audio"), seconds=1.0)
    data = _valid_transcript()
    data["segments"][0]["end"] = 2.0
    data["segments"][0]["words"][-1]["end"] = 2.0
    ctx.write("transcript", data)
    with pytest.raises(ContractError):
        validate_artifact(ctx, "transcript")


def test_segments_validator_accepts_valid(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    ctx.write("segments", _valid_segments())
    validate_artifact(ctx, "segments")


def test_segments_without_language_is_valid(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    data = _valid_segments()
    data.pop("language")
    ctx.write("segments", data)
    validate_artifact(ctx, "segments")


def test_segments_validator_rejects_id_gap(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    data = _valid_segments()
    data["segments"][1]["id"] = 3
    ctx.write("segments", data)
    with pytest.raises(ContractError):
        validate_artifact(ctx, "segments")


def test_translation_validator_accepts_and_rejects_mismatch(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    ctx.write("segments", _valid_segments())
    ctx.write("translation", _valid_translation())
    validate_artifact(ctx, "translation")

    bad = _valid_translation()
    bad["segments"][1]["id"] = 3
    ctx.write("translation", bad)
    with pytest.raises(ContractError):
        validate_artifact(ctx, "translation")

    bad = _valid_translation()
    bad["segments"][0]["target"] = ""
    ctx.write("translation", bad)
    with pytest.raises(ContractError):
        validate_artifact(ctx, "translation")


def test_translation_validator_rejects_timing_drift(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    ctx.write("segments", _valid_segments())
    drifted = _valid_translation()
    drifted["segments"][0]["start"] = 0.5
    ctx.write("translation", drifted)
    with pytest.raises(ContractError):
        validate_artifact(ctx, "translation")


def test_srt_validators_accept_valid_set(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    _write_srt_set(ctx.workdir)
    validate_artifact(ctx, "srt_source")
    validate_artifact(ctx, "srt_target")
    validate_artifact(ctx, "srt_bilingual")
    validate_srt_set(ctx)


def test_srt_validator_rejects_numbering_gap(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    bad = _valid_srt().replace("2\n00:00:01", "3\n00:00:01")
    _write_srt_set(ctx.workdir, source=bad)
    with pytest.raises(ContractError):
        validate_artifact(ctx, "srt_source")


def test_srt_validator_rejects_bad_timestamp(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    bad = _valid_srt().replace("00:00:00,000", "00:00:00.000")
    _write_srt_set(ctx.workdir, source=bad)
    with pytest.raises(ContractError):
        validate_artifact(ctx, "srt_source")


def test_srt_validator_rejects_empty_text(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    bad = _valid_srt().replace("Hello\n", "\n")
    _write_srt_set(ctx.workdir, source=bad)
    with pytest.raises(ContractError):
        validate_artifact(ctx, "srt_source")


def test_bilingual_requires_two_text_lines(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    _write_srt_set(ctx.workdir, bilingual=_valid_srt())
    with pytest.raises(ContractError):
        validate_artifact(ctx, "srt_bilingual")


def test_srt_set_rejects_count_mismatch(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    short = "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
    _write_srt_set(ctx.workdir, target=short)
    with pytest.raises(ContractError):
        validate_srt_set(ctx)


def test_media_validator_rejects_missing_and_empty(tmp_path):
    config = _write_config(tmp_path)
    ctx = _context(tmp_path, config)
    with pytest.raises(ContractError):
        validate_artifact(ctx, "source_video")
    ctx.path("source_video").parent.mkdir(parents=True, exist_ok=True)
    ctx.path("source_video").write_bytes(b"")
    with pytest.raises(ContractError):
        validate_artifact(ctx, "source_video")
    ctx.path("source_video").write_bytes(b"x")
    validate_artifact(ctx, "source_video")


def test_run_does_not_change_config(tmp_path):
    config = _write_config(tmp_path)
    before = hashlib.sha256(config.read_bytes()).hexdigest()
    assert run("noop", tmp_path / "work", config, stages=(NOOP,)) == 0
    assert hashlib.sha256(config.read_bytes()).hexdigest() == before


def test_production_stage_table_is_consistent():
    validate_registry()
    assert [stage.id for stage in STAGES] == [
        "ingest",
        "transcribe",
        "segment",
        "translate",
        "subtitle",
        "burn",
    ]
    produced = {key for stage in STAGES for key in stage.produces}
    assert set(ARTIFACTS) == produced
    assert ARTIFACTS["transcript"].relpath == "transcript.json"
    assert ARTIFACTS["srt_bilingual"].relpath == "subs/bilingual.srt"
    assert ARTIFACTS["output_video"].relpath == "output.mp4"


def test_validate_registry_rejects_unknown_artifact():
    bad = StageSpec("bad", "bad", ("missing",), (), (), _noop_run)
    with pytest.raises(ValueError):
        validate_registry((bad,))


def test_validate_registry_rejects_consumer_before_producer():
    consumer = StageSpec("consumer", "consumer", ("transcript",), (), (), _noop_run)
    producer = StageSpec("producer", "producer", (), ("transcript",), (), _noop_run)
    with pytest.raises(ValueError):
        validate_registry((consumer, producer))


def test_json_io_microbenchmark(tmp_path):
    codec = JsonCodec()
    words = []
    segments = []
    for seg_index in range(2000):
        seg_start = seg_index * 1.8
        seg_words = []
        for word_index in range(10):
            start = seg_start + word_index * 0.18
            item = {"start": start, "end": start + 0.18, "text": f"w{seg_index}_{word_index}"}
            seg_words.append(item)
            words.append(item)
        segments.append(
            {
                "start": seg_start,
                "end": seg_start + 1.8,
                "text": " ".join(w["text"] for w in seg_words),
                "words": seg_words,
            }
        )
    payload = {"schema_version": 1, "language": "en", "segments": segments}
    path = tmp_path / "large_transcript.json"

    start = time.perf_counter()
    codec.write(path, payload)
    write_seconds = time.perf_counter() - start
    start = time.perf_counter()
    loaded = codec.read(path)
    read_seconds = time.perf_counter() - start

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"json_io: {len(words)} words, {size_mb:.2f} MiB, write={write_seconds:.3f}s, read={read_seconds:.3f}s")
    assert sum(len(s["words"]) for s in loaded["segments"]) == len(words)


def test_cli_help_and_missing_input(tmp_path):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    help_result = subprocess.run(
        [PYTHON, "-m", "pipeline.runner", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert help_result.returncode == 0

    config = _write_config(tmp_path)
    missing = subprocess.run(
        [
            PYTHON,
            "-m",
            "pipeline.runner",
            "--stages",
            "segment",
            "--workdir",
            str(tmp_path / "work"),
            "--config",
            str(config),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert missing.returncode == 2
    assert "segment" in missing.stderr

    unknown = subprocess.run(
        [
            PYTHON,
            "-m",
            "pipeline.runner",
            "--stages",
            "nope",
            "--workdir",
            str(tmp_path / "work"),
            "--config",
            str(config),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert unknown.returncode == 2