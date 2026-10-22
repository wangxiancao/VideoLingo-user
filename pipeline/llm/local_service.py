"""Managed lifecycle for the local HY-MT2 OpenAI-compatible service (V4-2).

One service per pipeline run (spec #32 D3): the runner starts it before the
first LLM stage and stops it after the last one, on success, stage error,
SIGINT and SIGTERM alike. Shutdown wall-clock is recorded separately from
E2E RTFx. SIGKILL against the parent is covered by PR_SET_PDEATHSIG in the
child; the child also runs in its own session so a group signal reaches
uvicorn and its workers without touching the pipeline process.

Explicit-failure contract (spec #32 D8): missing model, occupied port and
startup timeout raise with actionable messages instead of silently
degrading; there is no cloud fallback.
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


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


class LocalLlmService:
    """Supervisor for one ``pipeline.llm.local_server`` subprocess."""

    def __init__(
        self,
        model_dir: str | Path,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_s: float = 180.0,
        python_exe: str | None = None,
        cwd: str | Path | None = None,
        stub: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.host = host
        self.port = int(port)
        self.startup_timeout_s = float(startup_timeout_s)
        self._python = python_exe or sys.executable
        self._cwd = str(cwd) if cwd else None
        self._stub = stub
        self._process: subprocess.Popen | None = None
        self.startup_seconds: float | None = None
        self.shutdown_seconds: float | None = None

    # -- validation -----------------------------------------------------

    def _validate_model_dir(self):
        if self._stub:
            return
        if not self.model_dir.is_dir():
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

    @property
    def api_key(self) -> str:
        return "local-managed"

    @property
    def model(self) -> str:
        return "hy-mt2-1.8b"

    def start(self) -> "LocalLlmService":
        self._validate_model_dir()
        if self._process is not None:
            raise LlmError("local llm service already started")
        if self.port == 0:
            self.port = _pick_free_port(self.host)
        command = [
            self._python, "-m", "pipeline.llm.local_server",
            "--host", self.host, "--port", str(self.port),
            "--die-with-parent",
        ]
        if self._stub:
            command.append("--stub")
        else:
            command += ["--model-dir", str(self.model_dir)]
        started = time.perf_counter()
        self._process = subprocess.Popen(
            command, cwd=self._cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
            env={**os.environ,
                 "NO_PROXY": "127.0.0.1,localhost",
                 "no_proxy": "127.0.0.1,localhost"},
        )
        deadline = started + self.startup_timeout_s
        while time.perf_counter() < deadline:
            if self._process.poll() is not None:
                tail = (self._process.stdout.read() or "")[-800:]
                raise LlmError(
                    f"local llm server exited during startup "
                    f"(code {self._process.returncode}): {tail.strip()}"
                )
            try:
                with urllib.request.urlopen(self.root_url + "/health",
                                            timeout=2) as response:
                    payload = json.loads(response.read())
                # The port may answer before our child binds it (a conflict
                # with an earlier server, or our child failing to bind while
                # another listener exists); only our own PID proves readiness.
                if payload.get("status") == "ok" and \
                        payload.get("pid") == self._process.pid:
                    self.startup_seconds = time.perf_counter() - started
                    return self
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(POLL_INTERVAL_S)
        self.stop()
        raise LlmError(
            f"local llm service did not become healthy within "
            f"{self.startup_timeout_s:.0f}s on {self.base_url}"
        )

    def stop(self) -> None:
        """Terminate the server; idempotent. Records shutdown wall-clock."""
        process = self._process
        if process is None:
            return
        self._process = None
        started = time.perf_counter()
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
                        f"local llm server refused SIGKILL (pid {process.pid})"
                    ) from None
        self.shutdown_seconds = time.perf_counter() - started

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def client(self):
        """An OpenAICompatibleClient aimed at this service."""
        from pipeline.llm.openai_client import OpenAICompatibleClient

        return OpenAICompatibleClient(
            api_key=self.api_key, base_url=self.base_url, model=self.model
        )

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
        model_dir=config.get("llm.local.model_dir", "_model_cache/HY-MT2-1.8B"),
        host=config.get("llm.local.host", "127.0.0.1"),
        port=config.get("llm.local.port", 0),
        startup_timeout_s=config.get("llm.local.startup_timeout_s", 180),
    )
    return service.start()
