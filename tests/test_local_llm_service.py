"""Lifecycle tests for the managed local LLM service (spec #32 V4-2).

All tests run against the stub server (`--stub`: no model, no GPU), so they
exercise supervision semantics only: start/health/stop, idempotent stop,
explicit failures, port conflicts and the runner's gating helper. Real-model
behaviour was measured in the V4-1 spike (docs/agents/v4-1-spike.md) and is
re-verified in the stage tickets (V4-3/V4-4), not here.
"""

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from pipeline.contracts import ConfigError, MissingInputError
from pipeline.llm.contracts import LlmError
from pipeline.llm.local_service import LocalLlmService

REPO = Path(__file__).resolve().parent.parent


def make_service(tmp_path, **kwargs):
    defaults = dict(model_dir=tmp_path / "model", stub=True, cwd=REPO)
    defaults.update(kwargs)
    return LocalLlmService(**defaults)


def test_stub_roundtrip(tmp_path):
    service = make_service(tmp_path)
    service.start()
    try:
        assert service.is_running()
        assert service.port != 0
        with urllib.request.urlopen(service.root_url + "/health", timeout=5) as r:
            assert json.loads(r.read())["status"] == "ok"
        client = service.client()
        content = client.complete("system", "user")
        payload = json.loads(content)
        assert payload["translations"][0]["id"] == 1
    finally:
        service.stop()
    assert not service.is_running()
    assert service.shutdown_seconds is not None


def test_stop_is_idempotent(tmp_path):
    service = make_service(tmp_path)
    service.start()
    service.stop()
    service.stop()  # must not raise
    assert service.shutdown_seconds is not None


def test_stop_terminates_process_group(tmp_path):
    service = make_service(tmp_path)
    service.start()
    pid = service._replicas[0].process.pid
    service.stop()
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with open(f"/proc/{pid}/stat") as handle:
                state = handle.read().split()[2]
            assert state == "Z", f"process {pid} alive in state {state}"
            return
        except FileNotFoundError:
            return
        time.sleep(0.1)
    pytest.fail(f"server process {pid} still alive after stop")


def test_missing_model_dir_fails_explicitly(tmp_path):
    service = make_service(tmp_path, stub=False,
                           model_dir=tmp_path / "does-not-exist")
    with pytest.raises(MissingInputError, match="model_dir not found"):
        service.start()


def test_incomplete_model_dir_fails_explicitly(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    service = make_service(tmp_path, stub=False, model_dir=model)
    with pytest.raises(ConfigError, match="incomplete"):
        service.start()


def test_occupied_port_fails_with_captured_output(tmp_path):
    blocker = make_service(tmp_path)
    blocker.start()
    try:
        service = make_service(tmp_path, port=blocker.port)
        with pytest.raises(LlmError, match="exited during startup"):
            service.start()
    finally:
        blocker.stop()


def test_context_manager_stops_on_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with make_service(tmp_path) as service:
            assert service.is_running()
            raise RuntimeError("stage error")
    assert not service.is_running()


def test_runner_helper_gating():
    from pipeline.runner import _maybe_start_local_llm

    class FakeStage:
        def __init__(self, stage_id):
            self.id = stage_id

    class FakeConfig:
        def __init__(self, enabled):
            self.values = {"llm.local.enabled": enabled}

        def get(self, key, default=None):
            return self.values.get(key, default)

    stages = (FakeStage("segment"), FakeStage("translate"))
    assert _maybe_start_local_llm(stages, FakeConfig(False), None, False) is None
    assert _maybe_start_local_llm(stages, FakeConfig(True), object(), False) is None
    assert _maybe_start_local_llm(stages, FakeConfig(True), None, True) is None
    assert _maybe_start_local_llm((FakeStage("ingest"),), FakeConfig(True),
                                  None, False) is None


def test_runner_helper_starts_service_for_llm_stage(tmp_path, monkeypatch):
    from pipeline.runner import _maybe_start_local_llm

    stub = make_service(tmp_path)
    monkeypatch.setattr("pipeline.llm.local_service.start_local_llm_service",
                        lambda config: stub.start())

    class FakeStage:
        id = "translate"

    class FakeConfig:
        def get(self, key, default=None):
            values = {"llm.local.enabled": True}
            return values.get(key, default)

    service = _maybe_start_local_llm((FakeStage(),), FakeConfig(), None, False)
    try:
        assert service is stub and service.is_running()
    finally:
        service.stop()


def test_round_robin_distributes_evenly():
    from pipeline.llm.round_robin import RoundRobinClient

    class Counting:
        def __init__(self, name):
            self.name = name
            self.calls = 0

        def complete(self, system, user):
            self.calls += 1
            return self.name

    clients = [Counting(f"c{i}") for i in range(3)]
    robin = RoundRobinClient(clients)
    results = [robin.complete("s", "u") for _ in range(9)]
    assert [client.calls for client in clients] == [3, 3, 3]
    assert results == ["c0", "c1", "c2"] * 3


def test_round_robin_thread_safe_under_concurrency():
    from concurrent.futures import ThreadPoolExecutor

    from pipeline.llm.round_robin import RoundRobinClient

    class Counting:
        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()

        def complete(self, system, user):
            with self.lock:
                self.calls += 1
            return "x"

    clients = [Counting() for _ in range(4)]
    robin = RoundRobinClient(clients)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: robin.complete("s", "u"), range(200)))
    total = sum(client.calls for client in clients)
    assert total == 200
    assert max(client.calls for client in clients) - \
        min(client.calls for client in clients) <= 1


def test_multi_replica_lifecycle_and_roundrobin_client(tmp_path):
    service = make_service(tmp_path, replicas=3)
    service.start()
    try:
        assert service.is_running()
        ports = [replica.port for replica in service._replicas]
        assert len(set(ports)) == 3 and all(p != 0 for p in ports)
        client = service.client()
        from pipeline.llm.round_robin import RoundRobinClient
        assert isinstance(client, RoundRobinClient)
        assert client.complete("s", "u")  # stub JSON round-trips
    finally:
        service.stop()
    assert not service.is_running()
    assert service.shutdown_seconds is not None


def test_replica_failure_stops_siblings(tmp_path):
    blocker = make_service(tmp_path)
    blocker.start()
    try:
        # first replica pinned to an occupied port -> startup failure must
        # tear down any already-started siblings and surface the error
        service = make_service(tmp_path, replicas=2, port=blocker.port)
        with pytest.raises(LlmError):
            service.start()
        assert service._replicas == []
    finally:
        blocker.stop()


def test_llamacpp_command_shape():
    from pipeline.llm.local_service import _llamacpp_command

    command = _llamacpp_command("bin/llama-server", "m.gguf", "127.0.0.1",
                                9940, 4096)
    assert command[:4] == ["bin/llama-server", "-m", "m.gguf", "-ngl"]
    assert "--jinja" in command
    assert command[command.index("-c") + 1] == "4096"
    assert command[command.index("-np") + 1] == "1"
    assert command[command.index("--repeat-penalty") + 1] == "1.0"
    assert command[command.index("--port") + 1] == "9940"


def test_llamacpp_missing_assets_fail_explicitly(tmp_path):
    from pipeline.llm.local_service import LocalLlmService

    service = LocalLlmService(runtime="llamacpp", replicas=1,
                              binary=tmp_path / "no-binary",
                              gguf=tmp_path / "no-model.gguf")
    with pytest.raises(MissingInputError, match="llama-server binary"):
        service.start()
    service = LocalLlmService(runtime="llamacpp", replicas=1,
                              binary="native/llamacpp/llama.cpp/build/bin/llama-server",
                              gguf=tmp_path / "no-model.gguf")
    with pytest.raises(MissingInputError, match="gguf"):
        service.start()


def test_invalid_runtime_and_replicas_rejected():
    from pipeline.llm.local_service import LocalLlmService

    with pytest.raises(ConfigError, match="runtime"):
        LocalLlmService(runtime="vllm")
    with pytest.raises(ConfigError, match="replicas"):
        LocalLlmService(replicas=0)
