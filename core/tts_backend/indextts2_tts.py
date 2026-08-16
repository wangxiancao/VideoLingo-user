import base64
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import requests

from core.utils import load_key, rprint


SERVER_PROCESS = None
SERVER_LOG_HANDLE = None
REFERENCE_CACHE = None
START_LOCK = threading.Lock()


def _project_root():
    return Path(__file__).resolve().parents[2]


def _resolve_path(path):
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (_project_root() / candidate).resolve()


def _load_optional(key, default):
    try:
        return load_key(key)
    except KeyError:
        return default


def _base_url():
    host = _load_optional("indextts2.host", "127.0.0.1")
    port = int(_load_optional("indextts2.port", 18721))
    return f"http://{host}:{port}"


def request_concurrency():
    return max(1, int(_load_optional("indextts2.max_concurrency", 4)))


def worker_count():
    """Backward-compatible alias for the former local worker pool size."""
    return request_concurrency()


def _health_ok():
    try:
        response = requests.get(f"{_base_url()}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _log_file():
    path = _project_root() / "output" / "log" / "vllm_omni_indextts2.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _tail_log(path, limit=6000):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def _runtime_python():
    # Do NOT resolve symlinks: venv bin/python symlinks to the base interpreter,
    # and executing the resolved path loses the venv site-packages.
    configured = _load_optional("indextts2.vllm_omni_python", None)
    if configured:
        candidate = Path(configured).expanduser()
    else:
        repo_dir = _resolve_path(_load_optional("indextts2.repo_dir", "external/index-tts"))
        candidate = repo_dir / ".venv" / "bin" / "python"
    return candidate.absolute()


def _model():
    configured = _load_optional("indextts2.model", None)
    if configured:
        candidate = _resolve_path(configured)
        return str(candidate) if candidate.exists() else str(configured)

    local_model = _resolve_path("external/index-tts/checkpoints")
    if local_model.exists():
        return str(local_model)
    return "IndexTeam/IndexTTS-2"


def _server_cmd():
    runtime_python = _runtime_python()
    launcher = _project_root() / "scripts" / "indextts2_vllm_omni_server.py"
    if not runtime_python.exists():
        raise FileNotFoundError(
            f"vLLM-Omni Python not found: {runtime_python}. "
            "Install vLLM and vLLM-Omni in the configured IndexTTS venv."
        )
    if not launcher.exists():
        raise FileNotFoundError(f"vLLM-Omni launcher not found: {launcher}")

    return [
        str(runtime_python),
        str(launcher),
        "--host",
        str(_load_optional("indextts2.host", "127.0.0.1")),
        "--port",
        str(int(_load_optional("indextts2.port", 18721))),
        "--model",
        _model(),
    ]


def _start_server():
    global SERVER_PROCESS, SERVER_LOG_HANDLE

    if _health_ok():
        return
    if SERVER_PROCESS is not None and SERVER_PROCESS.poll() is None:
        return

    cmd = _server_cmd()
    log_path = _log_file()
    SERVER_LOG_HANDLE = open(log_path, "a", encoding="utf-8")
    SERVER_LOG_HANDLE.write(
        f"\n\nStarting vLLM-Omni IndexTTS2 server: {' '.join(cmd)}\n"
    )
    SERVER_LOG_HANDLE.flush()
    SERVER_PROCESS = subprocess.Popen(
        cmd,
        cwd=_project_root(),
        stdout=SERVER_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    timeout = int(_load_optional("indextts2.vllm_omni_startup_timeout", 600))
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health_ok():
            rprint(f"[green]vLLM-Omni IndexTTS2 server is ready at {_base_url()}[/]")
            return
        if SERVER_PROCESS.poll() is not None:
            error = _tail_log(log_path)
            stop_indextts2_worker()
            raise RuntimeError(
                "vLLM-Omni IndexTTS2 server exited during startup.\n"
                f"Log tail:\n{error}"
            )
        time.sleep(2)

    error = _tail_log(log_path)
    stop_indextts2_worker()
    raise TimeoutError(
        f"vLLM-Omni IndexTTS2 server was not healthy within {timeout} seconds.\n"
        f"Log tail:\n{error}"
    )


def ensure_server():
    if _health_ok():
        return
    if not bool(_load_optional("indextts2.auto_start", True)):
        raise RuntimeError(
            f"vLLM-Omni IndexTTS2 server is unavailable at {_base_url()}. "
            "Start it with scripts/indextts2_vllm_omni_server.py or enable "
            "indextts2.auto_start."
        )
    with START_LOCK:
        if not _health_ok():
            _start_server()


def _resolve_reference_audio():
    global REFERENCE_CACHE
    if REFERENCE_CACHE is not None:
        return REFERENCE_CACHE

    preset_voice = load_key("indextts2.preset_voice")
    preset_voices = load_key("indextts2.preset_voices")
    if preset_voice not in preset_voices:
        raise ValueError(f"Unknown IndexTTS2 preset voice: {preset_voice}")
    reference = _resolve_path(preset_voices[preset_voice])
    if not reference.exists():
        raise FileNotFoundError(f"IndexTTS2 reference audio not found: {reference}")
    REFERENCE_CACHE = reference
    return reference


def _encode_audio_data_url(path):
    path = Path(path)
    mime_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
    }.get(path.suffix.lower(), "audio/wav")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _error_detail(response):
    try:
        return response.json()
    except ValueError:
        return response.text[:1000]


def indextts2_tts(text, save_path, number=None, task_df=None):
    ensure_server()
    reference = _resolve_reference_audio()
    output_path = _resolve_path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": _model(),
        "input": text,
        "response_format": "wav",
        "ref_audio": _encode_audio_data_url(reference),
    }
    api_key = str(_load_optional("indextts2.api_key", "sk-empty"))
    response = requests.post(
        f"{_base_url()}/v1/audio/speech",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=int(_load_optional("indextts2.request_timeout", 300)),
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"vLLM-Omni IndexTTS2 synthesis failed: {_error_detail(response)}"
        )
    if not response.content or not response.content.startswith(b"RIFF"):
        raise RuntimeError("vLLM-Omni returned an empty or invalid WAV response")

    partial_path = output_path.with_name(f"{output_path.name}.part")
    try:
        partial_path.write_bytes(response.content)
        partial_path.replace(output_path)
    finally:
        partial_path.unlink(missing_ok=True)
    print(f"Audio saved to {output_path} via vLLM-Omni")


def stop_indextts2_worker():
    """Stop only the vLLM-Omni server process started by this application."""
    global SERVER_PROCESS, SERVER_LOG_HANDLE, REFERENCE_CACHE

    if SERVER_PROCESS is not None and SERVER_PROCESS.poll() is None:
        rprint("[yellow]Stopping owned vLLM-Omni IndexTTS2 server...[/]")
        try:
            os.killpg(os.getpgid(SERVER_PROCESS.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            SERVER_PROCESS.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(SERVER_PROCESS.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            SERVER_PROCESS.wait(timeout=5)

    if SERVER_LOG_HANDLE is not None:
        SERVER_LOG_HANDLE.close()
    SERVER_PROCESS = None
    SERVER_LOG_HANDLE = None
    REFERENCE_CACHE = None
