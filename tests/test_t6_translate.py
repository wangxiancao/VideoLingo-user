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

class RecordingClient(CountingClient):
    def __init__(self):
        super().__init__()
        self.requests = []

    def complete(self, system, user):
        with self.lock:
            self.calls += 1
            self.requests.append(json.loads(user))
        ids = [segment["id"] for segment in self.requests[-1]["segments"]]
        suffix = "faithful" if "faithfully" in system else "refined"
        return json.dumps(
            {"translations": [
                {"id": segment_id, "target": f"{suffix}-{segment_id}"}
                for segment_id in ids
            ]}
        )


def test_translate_term_map_normalizes_request_only(tmp_path, monkeypatch):
    import pipeline.stages.translate as translate_stage

    client = RecordingClient()
    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: client,
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    segments = [{
        "id": 1, "start": 0.0, "end": 2.0, "text": "平均时间是1.96匹秒",
        "words": [{"start": 0.0, "end": 2.0, "text": "平均时间是1.96匹秒"}],
    }]
    (workdir / "segments.json").write_text(
        json.dumps({"schema_version": 1, "segments": segments}), encoding="utf-8"
    )
    config = _config(
        tmp_path,
        "translation:\n  term_map:\n    匹秒: 皮秒\n  cache: false\n",
    )

    assert run("translate", workdir, config) == 0
    faithful_texts = [segment["text"] for segment in client.requests[0]["segments"]]
    refine_sources = [segment["source"] for segment in client.requests[1]["segments"]]
    assert faithful_texts == ["平均时间是1.96皮秒"]
    assert refine_sources == ["平均时间是1.96皮秒"]
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["source"] == "平均时间是1.96匹秒"


class FlakyOnceClient(CountingClient):
    """Drops a trailing id on the first faithful call, then behaves."""

    def __init__(self):
        super().__init__()
        self.poisoned = True

    def complete(self, system, user):
        with self.lock:
            self.calls += 1
        request = json.loads(user)
        ids = [segment["id"] for segment in request["segments"]]
        suffix = "faithful" if "faithfully" in system else "refined"
        items = [{"id": segment_id, "target": f"{suffix}-{segment_id}"}
                 for segment_id in ids]
        if self.poisoned and suffix == "faithful":
            self.poisoned = False
            items = items[:-1]  # missing trailing id -> contract failure
        return json.dumps({"translations": items})


def test_translate_retries_once_on_missing_ids(tmp_path, monkeypatch):
    import pipeline.stages.translate as translate_stage

    client = FlakyOnceClient()
    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: client,
    )
    workdir = _workdir(tmp_path)
    config = _config(tmp_path, "translation:\n  cache: true\n")

    assert run("translate", workdir, config) == 0
    # faithful failed once (retried) + faithful ok + refine = 3 calls
    assert client.calls == 3
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["target"].startswith("refined-")
