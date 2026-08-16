"""End-to-end smoke test for IndexTTS2 served by vLLM-Omni.

Run from the repository root with the existing IndexTTS virtual environment:

    external/index-tts/.venv/bin/python \
        scripts/test_indextts2_vllm_omni.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
import wave
from importlib.metadata import version
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEXTTS_ROOT = PROJECT_ROOT / "external" / "index-tts"
EXPECTED_VENV = INDEXTTS_ROOT / ".venv"
DEFAULT_MODEL = INDEXTTS_ROOT / "checkpoints"
DEFAULT_REFERENCE = INDEXTTS_ROOT / "examples" / "voice_09.wav"
SERVER_LAUNCHER = PROJECT_ROOT / "scripts" / "indextts2_vllm_omni_server.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start vLLM-Omni, synthesize one sentence, and validate its WAV."
    )
    parser.add_argument(
        "--text",
        default="这是使用 vLLM Omni 和 IndexTTS 二生成的语音测试。",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18721)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ref-audio", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/videolingo_indextts2_vllm_omni_test.wav"),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/tmp/videolingo_indextts2_vllm_omni_server.log"),
    )
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Keep a server started by this test running after validation.",
    )
    return parser.parse_args()


def require_existing_indextts_venv():
    if Path(sys.prefix).resolve() != EXPECTED_VENV.resolve():
        raise RuntimeError(
            "This test must run in the existing IndexTTS venv:\n"
            "external/index-tts/.venv/bin/python "
            "scripts/test_indextts2_vllm_omni.py"
        )


def require_file(path, label):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def health_ok(base_url):
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def tail(path, limit=8000):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def start_server(args, model, log_path):
    base_url = f"http://{args.host}:{args.port}"
    if health_ok(base_url):
        return None, None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(SERVER_LAUNCHER),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        str(model),
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if health_ok(base_url):
            return process, log_handle
        if process.poll() is not None:
            stop_server(process)
            log_handle.close()
            raise RuntimeError(
                "vLLM-Omni exited during startup.\n"
                f"Log tail:\n{tail(log_path)}"
            )
        time.sleep(2)

    stop_server(process)
    log_handle.close()
    raise TimeoutError(
        f"vLLM-Omni was not healthy within {args.startup_timeout} seconds.\n"
        f"Log tail:\n{tail(log_path)}"
    )


def stop_server(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=10)


def audio_data_url(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def synthesize(args, base_url, model, reference, output):
    started = time.monotonic()
    response = requests.post(
        f"{base_url}/v1/audio/speech",
        json={
            "model": str(model),
            "input": args.text,
            "response_format": "wav",
            "ref_audio": audio_data_url(reference),
        },
        headers={"Authorization": "Bearer sk-empty"},
        timeout=args.request_timeout,
    )
    elapsed = time.monotonic() - started
    if response.status_code != 200:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:2000]
        raise RuntimeError(f"Synthesis failed with HTTP {response.status_code}: {detail}")
    if not response.content.startswith(b"RIFF"):
        raise RuntimeError("Synthesis response is not a RIFF WAV file")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.part")
    try:
        partial.write_bytes(response.content)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return elapsed


def inspect_wav(path):
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
    if frames <= 0 or sample_rate <= 0:
        raise RuntimeError("Generated WAV contains no audio frames")
    return {
        "bytes": path.stat().st_size,
        "channels": channels,
        "duration_seconds": round(frames / sample_rate, 3),
        "frames": frames,
        "sample_rate": sample_rate,
        "sample_width_bytes": sample_width,
    }


def main():
    args = parse_args()
    require_existing_indextts_venv()
    model = require_file(args.model, "IndexTTS2 model")
    reference = require_file(args.ref_audio, "reference audio")
    output = args.output.expanduser().resolve()
    log_path = args.log.expanduser().resolve()
    base_url = f"http://{args.host}:{args.port}"

    process = None
    log_handle = None
    server_was_running = health_ok(base_url)
    total_started = time.monotonic()
    try:
        process, log_handle = start_server(args, model, log_path)
        synthesis_seconds = synthesize(
            args,
            base_url,
            model,
            reference,
            output,
        )
        result = {
            "status": "passed",
            "runtime": {
                "python": sys.version.split()[0],
                "torch": version("torch"),
                "vllm": version("vllm"),
                "vllm_omni": version("vllm-omni"),
                "venv": str(EXPECTED_VENV),
            },
            "server": {
                "base_url": base_url,
                "preexisting": server_was_running,
                "started_by_test": process is not None,
            },
            "inputs": {
                "model": str(model),
                "reference_audio": str(reference),
                "text": args.text,
            },
            "output": {
                "path": str(output),
                **inspect_wav(output),
            },
            "timing": {
                "synthesis_seconds": round(synthesis_seconds, 3),
                "total_seconds": round(time.monotonic() - total_started, 3),
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if process is not None and not args.keep_server:
            stop_server(process)
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
