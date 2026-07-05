import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import requests

from core.utils import load_key, rprint


# ponytail: module-level worker pool. Single GPU saturates around 3 workers,
# so a dict + round-robin counter is enough; no queue, no complex scheduler.
WORKER_PROCESSES = {}  # port -> subprocess.Popen
PORT_POOL = []         # cached list of ports we route to
RR_LOCK = threading.Lock()
RR_COUNTER = 0
REFERENCE_CACHE = None  # ponytail: cache resolved reference path; config doesn't change mid-run


def _project_root():
    return Path(__file__).resolve().parents[2]


def _resolve_path(path):
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (_project_root() / candidate).resolve()


def _resolve_ports():
    """Prefer indextts2.ports (list); fallback to [indextts2.port] for backward compat."""
    try:
        ports = load_key("indextts2.ports")
        if ports and isinstance(ports, list):
            return [int(p) for p in ports]
    except KeyError:
        pass
    return [int(load_key("indextts2.port"))]


def worker_count():
    """Number of TTS worker ports configured. Used by _10_gen_audio to size the pool."""
    return len(_resolve_ports())


def _worker_base_url(port):
    host = load_key("indextts2.host")
    return f"http://{host}:{port}"


def _health_ok(port):
    try:
        resp = requests.get(f"{_worker_base_url(port)}/health", timeout=2)
        return resp.status_code == 200 and resp.json().get("ok") is True
    except Exception:
        return False


def _log_file(port):
    path = _project_root() / "output" / "log" / f"indextts2_worker_{port}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _tail_log(path, limit=4000):
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-limit:]


def _worker_cmd(port):
    repo_dir = _resolve_path(load_key("indextts2.repo_dir"))
    worker_script = _project_root() / "scripts" / "indextts2_worker.py"
    if not repo_dir.exists():
        raise FileNotFoundError(f"IndexTTS2 repo directory not found: {repo_dir}")
    if not worker_script.exists():
        raise FileNotFoundError(f"IndexTTS2 worker script not found: {worker_script}")

    cmd = [
        "uv", "run", "python", str(worker_script),
        "--host", load_key("indextts2.host"),
        "--port", str(port),
        "--model-dir", "checkpoints",
    ]
    if load_key("indextts2.use_fp16"):
        cmd.append("--use-fp16")
    if load_key("indextts2.use_cuda_kernel"):
        cmd.append("--use-cuda-kernel")
    if load_key("indextts2.use_deepspeed"):
        cmd.append("--use-deepspeed")
    return cmd, repo_dir


def _start_one_worker(port):
    """Spawn one worker process on `port` and block until /health is ok."""
    global WORKER_PROCESSES

    if _health_ok(port):
        return  # external worker already up

    if port in WORKER_PROCESSES and WORKER_PROCESSES[port].poll() is None:
        return  # we already started it

    cmd, repo_dir = _worker_cmd(port)
    log_path = _log_file(port)
    log_handle = open(log_path, "a", encoding="utf-8")
    log_handle.write(f"\n\nStarting IndexTTS2 worker on :{port}: {' '.join(cmd)}\n")
    log_handle.flush()

    # ponytail: start_new_session so we can killpg the whole tree. `uv run`
    # forks a wrapper process and a child python — terminating just the
    # wrapper orphans the child and leaks VRAM.
    proc = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    WORKER_PROCESSES[port] = proc

    deadline = time.time() + int(load_key("indextts2.startup_timeout"))
    while time.time() < deadline:
        if _health_ok(port):
            rprint(f"[green]IndexTTS2 worker on :{port} is ready[/]")
            return
        if proc.poll() is not None:
            raise RuntimeError(
                f"IndexTTS2 worker on :{port} exited during startup.\n"
                f"Log tail:\n{_tail_log(log_path)}"
            )
        time.sleep(2)

    raise TimeoutError(
        f"IndexTTS2 worker on :{port} did not become healthy within "
        f"{load_key('indextts2.startup_timeout')} seconds.\n"
        f"Log tail:\n{_tail_log(log_path)}"
    )


_START_LOCK = threading.Lock()


def start_all_workers():
    """Sequentially start one worker per configured port.

    Sequential on purpose: concurrent PyTorch inits fight for VRAM and
    can trigger transient OOMs even when the steady-state fits.
    """
    global PORT_POOL
    with _START_LOCK:
        if PORT_POOL:
            return  # already initialized this process
        ports = _resolve_ports()
        for port in ports:
            _start_one_worker(port)
        PORT_POOL = list(ports)


def _next_port():
    global RR_COUNTER
    with RR_LOCK:
        port = PORT_POOL[RR_COUNTER % len(PORT_POOL)]
        RR_COUNTER += 1
    return port


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


def indextts2_tts(text, save_path, number=None, task_df=None):
    if not PORT_POOL:
        start_all_workers()

    reference = _resolve_reference_audio()
    port = _next_port()
    output_path = _resolve_path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": text,
        "spk_audio_prompt": str(reference),
        "output_path": str(output_path),
        "verbose": False,
    }
    resp = requests.post(
        f"{_worker_base_url(port)}/synthesize",
        json=payload,
        timeout=int(load_key("indextts2.request_timeout")),
    )
    try:
        data = resp.json()
    except Exception:
        data = {"ok": False, "error": resp.text}

    if resp.status_code != 200 or not data.get("ok"):
        raise RuntimeError(f"IndexTTS2 synthesis failed on :{port}: {data}")

    print(f"Audio saved to {output_path} (via :{port})")


def stop_indextts2_worker():
    global WORKER_PROCESSES, PORT_POOL, RR_COUNTER, REFERENCE_CACHE

    if not WORKER_PROCESSES:
        PORT_POOL = []
        RR_COUNTER = 0
        REFERENCE_CACHE = None
        return

    rprint(f"[yellow]Stopping {len(WORKER_PROCESSES)} IndexTTS2 worker(s)...[/]")
    for port, proc in list(WORKER_PROCESSES.items()):
        if proc.poll() is None:
            # Kill the whole process group (uv wrapper + python child).
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
    WORKER_PROCESSES = {}
    PORT_POOL = []
    RR_COUNTER = 0
    REFERENCE_CACHE = None
