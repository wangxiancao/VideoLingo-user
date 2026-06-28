import os
import subprocess
import time
from pathlib import Path

import requests

from core.utils import load_key, rprint
from core.utils.models import _8_1_AUDIO_TASK, _VOCAL_AUDIO_FILE


WORKER_PROCESS = None


def _project_root():
    return Path(__file__).resolve().parents[2]


def _resolve_path(path):
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (_project_root() / candidate).resolve()


def _worker_base_url():
    host = load_key("indextts2.host")
    port = int(load_key("indextts2.port"))
    return f"http://{host}:{port}"


def _health_ok():
    try:
        resp = requests.get(f"{_worker_base_url()}/health", timeout=2)
        return resp.status_code == 200 and resp.json().get("ok") is True
    except Exception:
        return False


def _log_file():
    path = _project_root() / "output" / "log" / "indextts2_worker.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _tail_log(path, limit=4000):
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-limit:]


def _start_worker():
    global WORKER_PROCESS

    if _health_ok():
        return

    repo_dir = _resolve_path(load_key("indextts2.repo_dir"))
    worker_script = _project_root() / "scripts" / "indextts2_worker.py"
    if not repo_dir.exists():
        raise FileNotFoundError(f"IndexTTS2 repo directory not found: {repo_dir}")
    if not worker_script.exists():
        raise FileNotFoundError(f"IndexTTS2 worker script not found: {worker_script}")

    cmd = [
        "uv",
        "run",
        "python",
        str(worker_script),
        "--host",
        load_key("indextts2.host"),
        "--port",
        str(load_key("indextts2.port")),
        "--model-dir",
        "checkpoints",
    ]
    if load_key("indextts2.use_fp16"):
        cmd.append("--use-fp16")
    if load_key("indextts2.use_cuda_kernel"):
        cmd.append("--use-cuda-kernel")
    if load_key("indextts2.use_deepspeed"):
        cmd.append("--use-deepspeed")

    log_path = _log_file()
    log_handle = open(log_path, "a", encoding="utf-8")
    log_handle.write(f"\n\nStarting IndexTTS2 worker: {' '.join(cmd)}\n")
    log_handle.flush()

    WORKER_PROCESS = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + int(load_key("indextts2.startup_timeout"))
    while time.time() < deadline:
        if _health_ok():
            rprint(f"[green]IndexTTS2 worker is ready at {_worker_base_url()}[/green]")
            return
        if WORKER_PROCESS.poll() is not None:
            raise RuntimeError(
                "IndexTTS2 worker exited during startup.\n"
                f"Log tail:\n{_tail_log(log_path)}"
            )
        time.sleep(2)

    raise TimeoutError(
        f"IndexTTS2 worker did not become healthy within {load_key('indextts2.startup_timeout')} seconds.\n"
        f"Log tail:\n{_tail_log(log_path)}"
    )


def _missing_upstream_files():
    expected = [
        "output/audio/trans_subs_for_audio.srt",
        "output/audio/src_subs_for_audio.srt",
        _8_1_AUDIO_TASK,
        _VOCAL_AUDIO_FILE,
    ]
    return [path for path in expected if not _resolve_path(path).exists()]


def _resolve_videolingo_reference():
    reference = _resolve_path(load_key("indextts2.reference_audio"))
    if reference.exists():
        return reference

    task_file = _resolve_path(_8_1_AUDIO_TASK)
    vocal_file = _resolve_path(_VOCAL_AUDIO_FILE)
    if task_file.exists() and vocal_file.exists():
        from core._9_refer_audio import extract_refer_audio_main

        extract_refer_audio_main()
        if reference.exists():
            return reference

    missing = _missing_upstream_files()
    raise FileNotFoundError(
        "IndexTTS2 VideoLingo reference audio is missing and cannot be generated. "
        f"Expected reference: {reference}. Missing upstream files: {missing}. "
        "Run subtitle/audio task processing first, or set indextts2.reference_mode to preset/custom."
    )


def _resolve_reference_audio():
    mode = load_key("indextts2.reference_mode")
    if mode == "videolingo":
        reference = _resolve_videolingo_reference()
    elif mode == "preset":
        preset_voice = load_key("indextts2.preset_voice")
        preset_voices = load_key("indextts2.preset_voices")
        if preset_voice not in preset_voices:
            raise ValueError(f"Unknown IndexTTS2 preset voice: {preset_voice}")
        reference = _resolve_path(preset_voices[preset_voice])
    elif mode == "custom":
        custom_reference = load_key("indextts2.custom_reference_audio")
        if not custom_reference:
            raise ValueError("indextts2.custom_reference_audio is empty")
        reference = _resolve_path(custom_reference)
    else:
        raise ValueError("indextts2.reference_mode must be one of: videolingo, preset, custom")

    if not reference.exists():
        raise FileNotFoundError(f"IndexTTS2 reference audio not found: {reference}")
    return reference


def indextts2_tts(text, save_path, number=None, task_df=None):
    reference = _resolve_reference_audio()
    _start_worker()
    output_path = _resolve_path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": text,
        "spk_audio_prompt": str(reference),
        "output_path": str(output_path),
        "verbose": False,
    }
    resp = requests.post(
        f"{_worker_base_url()}/synthesize",
        json=payload,
        timeout=int(load_key("indextts2.request_timeout")),
    )
    try:
        data = resp.json()
    except Exception:
        data = {"ok": False, "error": resp.text}

    if resp.status_code != 200 or not data.get("ok"):
        raise RuntimeError(f"IndexTTS2 synthesis failed: {data}")

    print(f"Audio saved to {output_path}")


def stop_indextts2_worker():
    global WORKER_PROCESS

    if WORKER_PROCESS is None:
        return
    if WORKER_PROCESS.poll() is not None:
        WORKER_PROCESS = None
        return

    rprint("[yellow]Stopping IndexTTS2 worker started by this process...[/yellow]")
    WORKER_PROCESS.terminate()
    try:
        WORKER_PROCESS.wait(timeout=15)
    except subprocess.TimeoutExpired:
        WORKER_PROCESS.kill()
        WORKER_PROCESS.wait(timeout=5)
    WORKER_PROCESS = None
