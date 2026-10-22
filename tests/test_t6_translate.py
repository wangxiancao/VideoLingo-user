import json
import shutil
import threading
from pathlib import Path

from pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = REPO_ROOT / "tests/fixtures/segments/sample_6s.json"
EXAM = REPO_ROOT / "tests/fixtures/llm/translate_exam.json"
MISSING = REPO_ROOT / "tests/fixtures/llm/translate_exam_missing.json"


def _config(tmp_path, extra=""):
    path = tmp_path / "config.yaml"
    path.write_text(extra or "{}\n", encoding="utf-8")
    return path


def _workdir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copyfile(SEGMENTS, workdir / "segments.json")
    return workdir


class CountingClient:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def complete(self, system, user):
        with self.lock:
            self.calls += 1
        request = json.loads(user)
        ids = [segment["id"] for segment in request["segments"]]
        suffix = "faithful" if "faithful" in system.lower() else "refined"
        return json.dumps(
            {
                "translations": [
                    {"id": segment_id, "target": f"{suffix}-{segment_id}"}
                    for segment_id in ids
                ]
            }
        )


def test_translate_two_steps_and_preserves_timing(tmp_path):
    workdir = _workdir(tmp_path)

    assert run(
        "translate",
        workdir,
        _config(tmp_path),
        llm_fixture=EXAM,
    ) == 0

    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["source_language"] == "en"
    assert data["target_language"] == "简体中文"
    assert len(data["segments"]) == 2
    assert [segment["id"] for segment in data["segments"]] == [1, 2]
    assert data["segments"][0]["source"].startswith("This is a deliberately long")
    assert data["segments"][0]["target"] == "这是经过意译和反思后的自然中文版本。"
    assert data["segments"][1]["target"] == "短句。"

    segments = json.loads((workdir / "segments.json").read_text(encoding="utf-8"))
    for translated, source in zip(data["segments"], segments["segments"]):
        assert translated["start"] == source["start"]
        assert translated["end"] == source["end"]


def test_translate_rejects_missing_ids(tmp_path, capsys):
    workdir = _workdir(tmp_path)

    assert run(
        "translate",
        workdir,
        _config(tmp_path),
        llm_fixture=MISSING,
    ) == 1
    error = capsys.readouterr().err
    assert "stage 'translate'" in error
    assert "id" in error.lower()
    assert not (workdir / "translation.json").exists()


def test_translate_cache_avoids_second_llm_call(tmp_path, monkeypatch):
    import pipeline.stages.translate as translate_stage

    client = CountingClient()
    monkeypatch.setattr(
        translate_stage, "create_llm_client", lambda _config, _fixture, _service=None: client
    )
    workdir = _workdir(tmp_path)
    config = _config(tmp_path)

    assert run("translate", workdir, config) == 0
    assert client.calls == 2
    first = (workdir / "translation.json").read_text(encoding="utf-8")

    assert run("translate", workdir, config) == 0
    assert client.calls == 2
    assert (workdir / "translation.json").read_text(encoding="utf-8") == first


def test_translate_batches_segments(tmp_path, monkeypatch):
    import pipeline.stages.translate as translate_stage

    client = CountingClient()
    monkeypatch.setattr(
        translate_stage, "create_llm_client", lambda _config, _fixture, _service=None: client
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    segments = []
    for index in range(1, 10):
        start = float(index - 1)
        segments.append(
            {
                "id": index,
                "start": start,
                "end": start + 1.0,
                "text": f"Sentence {index}.",
                "words": [{"start": start, "end": start + 1.0, "text": f"Sentence {index}."}],
            }
        )
    (workdir / "segments.json").write_text(
        json.dumps({"schema_version": 1, "segments": segments}), encoding="utf-8"
    )
    config = _config(tmp_path, "translation:\n  batch_size: 2\nmax_workers: 2\n")

    assert run("translate", workdir, config) == 0
    assert client.calls == 10
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert len(data["segments"]) == 9
    assert data["segments"][-1]["target"] == "refined-9"