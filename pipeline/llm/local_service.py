"""Managed lifecycle for the local HY-MT2 OpenAI-compatible service.

One service per pipeline run (spec #32 D3): the runner starts it before the
first LLM stage and stops it after the last one, on success, stage error,
SIGINT and SIGTERM alike. Shutdown wall-clock is recorded separately from
E2E RTFx. Children run in their own session so a group signal reaches the
server and its workers without touching the pipeline process.

V4-7 O2: the service fronts N replicas (round-robin across independent
ports) and two runtimes — `llamacpp` (Q4 GGUF via the STQ-built
llama-server, the default) and `transformers` (the BF16 FastAPI fallback).
Explicit-failure contract (spec #32 D8) is unchanged: missing model,
occupied port and startup timeout raise actionable errors; no cloud
fallback.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from pipeline.contracts import ConfigError, MissingInputError
from pipeline.llm.contracts import LlmError

POLL_INTERVAL_S = 0.5
TERM_WAIT_S = 15.0
KILL_WAIT_S = 5.0
DEFAULT_BINARY = "native/llamacpp/llama.cpp/build/bin/llama-server"
# The STQ build needs the CUDA 12.8 runtime and a libstdc++ new enough for
# it on this box; harmless elsewhere when the paths exist.
RUNTIME_LIBRARY_PATH = "/usr/local/lib64:/usr/local/cuda-12.8/lib64"


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _llamacpp_command(binary, gguf, host, port, ctx, gpu_layers=99):
    """Subtitle-serving shape: bounded context, one slot, greedy-safe."""
    return [
        str(binary), "-m", str(gguf),
        "-ngl", str(gpu_layers), "--jinja",
        "-c", str(ctx), "-np", "1",
        "--host", host, "--port", str(port),
        "--repeat-penalty", "1.0",
    ]


class _Replica:
    """One server subprocess with per-runtime spawn and identity checks."""

    def __init__(self, runtime, host, port, startup_timeout_s,
                 model_dir=None, gguf=None, binary=None, ctx=4096,
                 python_exe=None, cwd=None, stub=False):
        self.runtime = runtime
        self.host = host
        self.port = int(port)
        self.startup_timeout_s = float(startup_timeout_s)
        self.model_dir = Path(model_dir) if model_dir else None
        self.gguf = Path(gguf) if gguf else None
        self.binary = Path(binary) if binary else None
        self.ctx = int(ctx)
        self._python = python_exe or sys.executable
        self._cwd = str(cwd) if cwd else None
        self._stub = stub
        self.process: subprocess.Popen | None = None

    # -- validation -----------------------------------------------------

    def validate(self):
        if self._stub:
            return
        if self.runtime == "llamacpp":
            if self.binary is None or not self.binary.is_file():
                raise MissingInputError(
                    f"llama-server binary not found: {self.binary or DEFAULT_BINARY} "
                    "(build it once with native/llamacpp/build.sh)"
                )
            if self.gguf is None or not self.gguf.is_file():
                raise MissingInputError(
                    f"local llm gguf not found: {self.gguf} "
                    "(download it once from ModelScope; the pipeline never "
                    "downloads at runtime)"
                )
            return
        if self.model_dir is None or not self.model_dir.is_dir():
            raise MissingInputError(
                f"local llm model_dir not found: {self.model_dir} "
                "(download it once; the pipeline never downloads at runtime)"
            )
        required = ["config.json", "chat_template.jinja"]
        missing = [name for name in required
                   if not (self.model_dir / name).is_file()]
        weights = list(self.model_dir.glob("*.safetensors"))
        if missing or not weights:
            detail = ", ".join(missing + (["*.safetensors"] if not weights else []))
            raise ConfigError(
                f"local llm model_dir incomplete: {self.model_dir} "
                f"missing {detail}"
            )

    # -- lifecycle ------------------------------------------------------

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        """OpenAI SDK base URL; the SDK appends /chat/completions itself."""
        return f"http://{self.host}:{self.port}/v1"

    def _spawn_command(self):
        if self.runtime == "llamacpp":
            return _llamacpp_command(self.binary, self.gguf, self.host,
                                     self.port, self.ctx)
        command = [
            self._python, "-m", "pipeline.llm.local_server",
            "--host", self.host, "--port", str(self.port),
            "--die-with-parent",
        ]
        if self._stub:
            command.append("--stub")
        else:
            command += ["--model-dir", str(self.model_dir)]
        return command

    def _spawn_env(self):
        env = {**os.environ,
               "NO_PROXY": "127.0.0.1,localhost",
               "no_proxy": "127.0.0.1,localhost"}
        if self.runtime == "llamacpp":
            # build/bin carries the sibling .so files plus the CUDA runtime
            # and libstdc++ the STQ build was linked against
            lib_dirs = [str(self.binary.parent), RUNTIME_LIBRARY_PATH]
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing]
                                                          if existing else []))
        return env

    def _identity_ok(self) -> bool:
        """Prove the healthy port is OUR child, not a stale listener.

        FastAPI server: /health echoes its PID. llama-server: /v1/models
        echoes the -m path we passed. Both defeat the occupied-port race.
        """
        try:
            if self.runtime == "llamacpp":
                with urllib.request.urlopen(self.root_url + "/v1/models",
                                            timeout=2) as response:
                    payload = json.loads(response.read())
                models = payload.get("data") or []
                return any(str(item.get("id", "")) == str(self.gguf)
                           for item in models)
            with urllib.request.urlopen(self.root_url + "/health",
                                        timeout=2) as response:
                payload = json.loads(response.read())
            return payload.get("status") == "ok" and \
                payload.get("pid") == self.process.pid
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            return False

    def start(self) -> float:
        """Spawn and wait for a healthy, identity-verified server."""
        self.validate()
        if self.process is not None:
            raise LlmError("replica already started")
        started = time.perf_counter()
        self.process = subprocess.Popen(
            self._spawn_command(), cwd=self._cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True, env=self._spawn_env(),
        )
        deadline = started + self.startup_timeout_s
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                tail = (self.process.stdout.read() or "")[-800:]
                raise LlmError(
                    f"local llm replica exited during startup "
                    f"(code {self.process.returncode}): {tail.strip()}"
                )
            if self._identity_ok():
                return time.perf_counter() - started
            time.sleep(POLL_INTERVAL_S)
        self.stop()
        raise LlmError(
            f"local llm replica did not become healthy within "
            f"{self.startup_timeout_s:.0f}s on {self.base_url}"
        )

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        self.process = None
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=TERM_WAIT_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=KILL_WAIT_S)
                except subprocess.TimeoutExpired:
                    raise LlmError(
                        f"local llm replica refused SIGKILL (pid {process.pid})"
                    ) from None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class LocalLlmService:
    """Supervisor facade over N replicas behind one round-robin client."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_s: float = 180.0,
        python_exe: str | None = None,
        cwd: str | Path | None = None,
        stub: bool = False,
        runtime: str = "transformers",
        replicas: int = 1,
        gguf: str | Path | None = None,
        binary: str | Path | None = None,
        ctx: int = 4096,
    ):
        if runtime not in ("transformers", "llamacpp"):
            raise ConfigError(
                f"llm.local.runtime must be transformers or llamacpp, "
                f"got {runtime!r}"
            )
        if int(replicas) < 1 or int(replicas) > 8:
            raise ConfigError(
                f"llm.local.replicas must be 1..8, got {replicas}"
            )
        self.runtime = runtime
        self.replica_count = int(replicas)
        self.host = host
        self.startup_timeout_s = float(startup_timeout_s)
        self._replicas: list[_Replica] = []
        self._first_port = int(port)
        self._common = dict(model_dir=model_dir, gguf=gguf, binary=binary,
                            ctx=ctx, python_exe=python_exe, cwd=cwd,
                            stub=stub, startup_timeout_s=startup_timeout_s)
        self.startup_seconds: float | None = None
        self.shutdown_seconds: float | None = None

    @property
    def port(self) -> int:
        return self._replicas[0].port if self._replicas else self._first_port

    @property
    def root_url(self) -> str:
        return self._replicas[0].root_url if self._replicas else \
            f"http://{self.host}:{self._first_port}"

    @property
    def base_url(self) -> str:
        return self._replicas[0].base_url if self._replicas else \
            f"http://{self.host}:{self._first_port}/v1"

    @property
    def api_key(self) -> str:
        return "local-managed"

    @property
    def model(self) -> str:
        return "hy-mt2-1.8b"

    def start(self) -> "LocalLlmService":
        if self._replicas:
            raise LlmError("local llm service already started")
        started = time.perf_counter()
        ports = []
        for index in range(self.replica_count):
            port = self._first_port if index == 0 and self._first_port else \
                _pick_free_port(self.host)
            replica = _Replica(runtime=self.runtime, host=self.host, port=port,
                               **self._common)
            try:
                replica.start()
            except Exception:
                for started_replica in self._replicas:
                    started_replica.stop()
                self._replicas = []
                raise
            self._replicas.append(replica)
            ports.append(port)
        self.startup_seconds = time.perf_counter() - started
        return self

    def stop(self) -> None:
        """Stop every replica; idempotent. Records shutdown wall-clock."""
        replicas, self._replicas = self._replicas, []
        if not replicas:
            return
        started = time.perf_counter()
        for replica in replicas:
            replica.stop()
        self.shutdown_seconds = time.perf_counter() - started

    def is_running(self) -> bool:
        return bool(self._replicas) and all(r.is_running() for r in self._replicas)

    def client(self):
        """A round-robin client across all replica endpoints."""
        from pipeline.llm.openai_client import OpenAICompatibleClient
        from pipeline.llm.round_robin import RoundRobinClient

        clients = [
            OpenAICompatibleClient(api_key=self.api_key,
                                   base_url=replica.base_url,
                                   model=self.model)
            for replica in self._replicas
        ]
        return RoundRobinClient(clients) if len(clients) > 1 else clients[0]

    def __enter__(self) -> "LocalLlmService":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def start_local_llm_service(config) -> LocalLlmService:
    """Build and start the service from `llm.local.*` config keys."""
    enabled = config.get("llm.local.enabled", False)
    if not enabled:
        raise ConfigError("start_local_llm_service called with llm.local.enabled off")
    service = LocalLlmService(
        runtime=config.get("llm.local.runtime", "transformers"),
        replicas=int(config.get("llm.local.replicas", 1)),
        gguf=config.get("llm.local.gguf_path"),
        binary=config.get("llm.local.binary", DEFAULT_BINARY),
        ctx=int(config.get("llm.local.ctx", 4096)),
        model_dir=config.get("llm.local.model_dir", "_model_cache/HY-MT2-1.8B"),
        host=config.get("llm.local.host", "127.0.0.1"),
        port=config.get("llm.local.port", 0),
        startup_timeout_s=config.get("llm.local.startup_timeout_s", 180),
    )
    return service.start()
