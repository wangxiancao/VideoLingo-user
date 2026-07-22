"""
Single-video translation script.
Runs the full text/subtitle pipeline and saves results to history/.

Usage:
    # Subtitles only (default), no dubbing
    python translate_video.py /path/to/video.mp4
    python translate_video.py --mode no-tts /path/to/video.mp4

    # Subtitles + dubbing (cloud TTS via api.key, e.g. deepseek for translation
    # and any cloud TTS method set as tts_method in config.yaml)
    python translate_video.py --mode tts https://www.youtube.com/watch?v=xxx

    # Subtitles + dubbing with EdgeTTS (online, no GPU, preset voice)
    python translate_video.py --mode tts --voice male /path/to/video.mp4
    python translate_video.py --mode tts --voice female /path/to/video.mp4

    # Local Qwen36 LLM for this run only (subtitles only):
    python translate_video.py --llm qwen36 /path/to/video.mp4
    python translate_video.py --llm qwen36 --llm-model qwen3.6-35b-a3b /path/to/video.mp4

    # Local Qwen36 LLM + dubbing:
    python translate_video.py --llm qwen36 --mode tts /path/to/video.mp4
    python translate_video.py --llm qwen36 --mode tts --tts-workers 3 /path/to/video.mp4

    Parallel IndexTTS2 (multiple worker processes on different ports):
      --tts-workers N writes an IndexTTS2 port pool to config.yaml before the
      pipeline starts; the client spawns one worker per port and round-robins
      requests. Parallelism (max_workers in stage 10) auto-tracks the number of
      ports. With --llm qwen36, port 18722 is reserved for the Qwen36 bridge and
      skipped by the TTS pool.

    Recommended N by GPU VRAM (measured, FP16, single GPU):
      24 GB (3090/4090):   N=3   (N=4 OOMs at model load; ~6.5 GB/worker)
      16 GB:               N=2
      12 GB or less:       N=1   (default; same as omitting the flag)

    Measured throughput on RTX 3090, 33 short Chinese subtitle lines:
      N=1  → 0.199 lines/s  (1.00x baseline)
      N=2  → 0.295 lines/s  (1.48x)
      N=3  → 0.360 lines/s  (1.81x)

    EdgeTTS voice selection (tts_method=='edge_tts'):
      --voice selects the EdgeTTS voice for this run. Accepts 'male' /
      'female' shorthand (male=zh-CN-YunyangNeural, female=zh-CN-XiaoxiaoNeural)
      or any full edge-tts voice name (e.g. zh-CN-YunxiNeural). Writes
      edge_tts.voice to config.yaml before the pipeline starts; omit the flag
      to use the value already in config.yaml. No GPU and no worker pool —
      online only, fixed preset voice (no speaker cloning).

    Examples:
    python translate_video.py --mode tts --tts-workers 3 /path/to/video.mp4
    python translate_video.py --mode tts --tts-workers 2 /path/to/video.mp4
    python translate_video.py --mode tts --tts-workers 3 --llm qwen36 /path/to/video.mp4
    python translate_video.py --mode tts --voice male /path/to/video.mp4
    python translate_video.py --mode tts --voice female --llm qwen36 /path/to/video.mp4
    Notes:
      - --llm qwen36 temporarily routes api.* to http://127.0.0.1:18722/v1,
        starts the Qwen36 bridge after local ASR, stops it before local TTS,
        keeps reflect_translate enabled, and restores the original config on
        exit.
      - --tts-workers is a no-op unless tts_method=='indextts2' in config.yaml.
      - --voice is a no-op unless tts_method=='edge_tts' in config.yaml; it
        persists edge_tts.voice in config.yaml for subsequent runs.
      - Setting N persists in config.yaml; subsequent runs without the flag
        reuse the same pool. Pass --tts-workers 1 to revert to single-worker.
"""

import argparse
import os
import sys
import ctypes

# ------------
# LD_LIBRARY_PATH fix: conda env lib must be searched before system libs.
# Setting os.environ is too late — the dynamic linker only reads LD_LIBRARY_PATH
# at process startup. So we detect the problem and re-exec ourselves.
# ------------
_conda_lib = os.path.join(sys.prefix, "lib")

def _cxxabi_ok():
    """Check if libopenvino_c can be loaded with current libstdc++."""
    try:
        ctypes.CDLL("libopenvino_c.so.2430")
        return True
    except OSError:
        return False

if os.path.isdir(_conda_lib) and _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    if not _cxxabi_ok():
        os.environ["LD_LIBRARY_PATH"] = _conda_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        os.execv(sys.executable, [sys.executable] + sys.argv)

import shutil
import subprocess
import time
import signal
from functools import partial
from rich.panel import Panel
from rich.console import Console

import requests

console = Console()


def prepare_output_folder(output_folder="output"):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)


def process_input_file(file):
    from core._1_ytdlp import find_video_files, download_video_ytdlp
    from core.utils.config_utils import load_key

    if file.startswith("http"):
        download_video_ytdlp(file, resolution=load_key("ytb_resolution"))
        video_file = find_video_files()
    else:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Video file not found: {file}")
        filename = os.path.basename(file)
        output_file = os.path.join("output", filename)
        shutil.copy(file, output_file)
        video_file = output_file
    return {"video_file": video_file}


def split_sentences():
    from core._3_1_split_nlp import split_by_spacy
    from core._3_2_split_meaning import split_sentences_by_meaning

    split_by_spacy()
    split_sentences_by_meaning()


def summarize_and_translate():
    from core._4_1_summarize import get_summary
    from core._4_2_translate import translate_all

    get_summary()
    translate_all()


def process_and_align_subtitles():
    from core._5_split_sub import split_for_sub_main
    from core._6_gen_sub import align_timestamp_main

    split_for_sub_main()
    align_timestamp_main()


def parse_args():
    parser = argparse.ArgumentParser(description="Translate one video with optional dubbing.")
    parser.add_argument("video_path_or_url", help="Local video path or YouTube URL")
    parser.add_argument(
        "--mode",
        choices=["no-tts", "tts"],
        default="no-tts",
        help="no-tts translates subtitles only; tts translates first, then generates dubbing.",
    )
    parser.add_argument(
        "--tts-workers",
        type=int,
        default=None,
        help=(
            "Number of IndexTTS2 worker processes to spawn (only used with --mode tts and "
            "tts_method=indextts2). Writes indextts2.ports=[18721, 18722, ...] to config.yaml "
            "before the pipeline starts, skipping the Qwen36 bridge port when needed. "
            "Recommended 3 for a single 24GB GPU."
        ),
    )
    parser.add_argument(
        "--voice",
        default=None,
        help=(
            "EdgeTTS voice override (only used with tts_method=edge_tts). "
            "Accepts 'male' / 'female' shorthand or a full voice name "
            "(e.g. zh-CN-YunyangNeural). Writes edge_tts.voice to config.yaml."
        ),
    )
    parser.add_argument(
        "--llm",
        choices=["config", "qwen36"],
        default="config",
        help=(
            "LLM backend for this run. 'config' uses config.yaml as-is; "
            "'qwen36' temporarily routes api.* to the local Qwen36 bridge."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Model name to use with --llm qwen36. "
            "Defaults to qwen3.6-35b-a3b."
        ),
    )
    return parser.parse_args()


def _build_port_pool(base_port, count, reserved_ports=None):
    reserved_ports = set(reserved_ports or [])
    ports = []
    port = base_port
    while len(ports) < count:
        if port not in reserved_ports:
            ports.append(port)
        port += 1
    return ports


def apply_tts_workers_override(n, reserved_ports=None):
    """Translate --tts-workers N into an IndexTTS2 port pool.

    Writes to config.yaml so every downstream module reads the same pool.
    No-op when n is None, no existing port conflict is found, or tts_method
    isn't indextts2. Reserved ports are skipped so the TTS pool cannot collide
    with another local service.

    Writes the YAML directly via ruamel rather than update_key, because
    update_key refuses to add a key that doesn't yet exist — and existing
    config.yaml files only have the singular `port:` field.
    """
    from core.utils.config_utils import load_key, CONFIG_PATH, lock, yaml

    if load_key("tts_method") != "indextts2":
        console.print(
            "[yellow]--tts-workers ignored (tts_method is not 'indextts2')[/]"
        )
        return

    reserved_ports = set(reserved_ports or [])
    if n is None:
        if not reserved_ports:
            return
        try:
            existing_ports = load_key("indextts2.ports")
        except KeyError:
            return
        if not existing_ports or not isinstance(existing_ports, list):
            return
        existing_ports = [int(p) for p in existing_ports]
        if not (set(existing_ports) & reserved_ports):
            return
        n = len(existing_ports)
        console.print(Panel(
            f"[yellow]Configured IndexTTS2 ports collide with reserved local LLM port(s)[/]\n"
            f"  existing ports: {existing_ports}\n"
            f"  reserved ports: {sorted(reserved_ports)}\n"
            f"  reassigning a {n}-worker pool",
            border_style="yellow",
        ))

    if n < 1:
        raise ValueError(f"--tts-workers must be >= 1, got {n}")

    base_port = int(load_key("indextts2.port"))
    ports = _build_port_pool(base_port, n, reserved_ports=reserved_ports)

    # ponytail: direct YAML edit. update_key can't add a missing key, and we
    # don't want to require users to manually migrate config.yaml.
    with lock:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.load(f)
        data["indextts2"]["ports"] = ports
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

    console.print(Panel(
        f"[bold green]Configured IndexTTS2 worker pool[/]\n"
        f"  ports: {ports}\n"
        f"  max_workers: {n}\n"
        f"  reserved ports: {sorted(reserved_ports) if reserved_ports else 'none'}\n"
        f"  (single 24GB GPU maxes out at ~3 workers before OOM)",
        border_style="green",
    ))


# ponytail: 2-entry shorthand map. Pass a full voice name for anything else,
# or move to a config map if more presets are needed later.
EDGE_VOICE_ALIASES = {
    "male": "zh-CN-YunyangNeural",
    "female": "zh-CN-XiaoxiaoNeural",
}


def apply_edge_voice_override(voice):
    """Translate --voice into an edge_tts.voice config write.

    Accepts 'male'/'female' shorthand or any full edge-tts voice name.
    No-op when voice is None or tts_method isn't edge_tts.
    """
    from core.utils.config_utils import load_key, CONFIG_PATH, lock, yaml

    if voice is None:
        return
    if load_key("tts_method") != "edge_tts":
        console.print(
            "[yellow]--voice ignored (tts_method is not 'edge_tts')[/]"
        )
        return

    resolved = EDGE_VOICE_ALIASES.get(voice.lower(), voice)
    with lock:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.load(f)
        data["edge_tts"]["voice"] = resolved
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

    console.print(Panel(
        f"[bold green]EdgeTTS voice override[/]\n"
        f"  --voice {voice} -> {resolved}",
        border_style="green",
    ))


def run_steps(steps):
    for step_name, step_func in steps:
        for attempt in range(3):
            try:
                console.print(Panel(
                    f"[bold green]{step_name}[/]",
                    subtitle=f"Attempt {attempt + 1}/3" if attempt > 0 else None,
                    border_style="blue",
                ))
                step_func()
                break
            except Exception as e:
                if attempt == 2:
                    console.print(Panel(
                        f"[bold red]Failed after 3 attempts: {step_name}[/]\n{e}",
                        border_style="red",
                    ))
                    raise
                console.print(Panel(
                    f"[yellow]Attempt {attempt + 1} failed, retrying...[/]",
                    border_style="yellow",
                ))


def stop_optional_tts_workers():
    try:
        from core.tts_backend.indextts2_tts import stop_indextts2_worker

        stop_indextts2_worker()
    except Exception as e:
        console.print(Panel(
            f"[yellow]Warning: failed to stop optional TTS worker[/]\n{e}",
            border_style="yellow",
        ))


# ------------
# Local Qwen36 bridge lifecycle (auto-start if base_url points to localhost:18722)
# ------------

QWEN36_BRIDGE_PROCESS = None
QWEN36_BRIDGE_LOG = None
QWEN36_BRIDGE_PORT = 18722
QWEN36_BRIDGE_BASE_URL = f"http://127.0.0.1:{QWEN36_BRIDGE_PORT}/v1"
QWEN36_BRIDGE_MODEL = "qwen3.6-35b-a3b"
QWEN36_API_KEY = "not-needed"
QWEN36_MAX_WORKERS = 1
QWEN36_BRIDGE_SCRIPT = "/root/soft/llama.cpp/qwen36-openai-bridge.py"
QWEN36_BRIDGE_LD_PATH = (
    "/usr/local/lib64:"
    "/root/soft/llama.cpp/build/bin:"
    "/usr/local/cuda-12.8/lib64"
)


def _config_get(data, key):
    value = data
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Key '{key}' not found in configuration")
        value = value[part]
    return value


def _config_set(data, key, new_value):
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Key '{key}' not found in configuration")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise KeyError(f"Key '{key}' not found in configuration")
    current[parts[-1]] = new_value


def _update_config_values(updates):
    from core.utils.config_utils import CONFIG_PATH, lock, yaml

    with lock:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.load(f)
        snapshot = {key: _config_get(data, key) for key in updates}
        for key, value in updates.items():
            _config_set(data, key, value)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
    return snapshot


def _restore_config_values(snapshot):
    if not snapshot:
        return
    _update_config_values(snapshot)
    console.print("[green]Restored original LLM config[/]")


def apply_llm_override(args):
    """Apply per-run LLM config overrides and return the original values."""
    if args.llm == "config":
        if args.llm_model:
            console.print("[yellow]--llm-model ignored because --llm=config[/]")
        return None

    from core.utils.config_utils import load_key

    model = args.llm_model or QWEN36_BRIDGE_MODEL
    snapshot = _update_config_values({
        "api.key": QWEN36_API_KEY,
        "api.base_url": QWEN36_BRIDGE_BASE_URL,
        "api.model": model,
        "api.llm_support_json": False,
        "max_workers": QWEN36_MAX_WORKERS,
        "reflect_translate": True,
    })
    reflect_translate = load_key("reflect_translate")

    console.print(Panel(
        f"[bold green]Using local Qwen36 LLM for this run[/]\n"
        f"  base_url: {QWEN36_BRIDGE_BASE_URL}\n"
        f"  model: {model}\n"
        f"  max_workers: {QWEN36_MAX_WORKERS}\n"
        f"  reflect_translate: {reflect_translate}\n"
        f"  original config will be restored on exit",
        border_style="green",
    ))
    return snapshot


def _qwen36_bridge_health():
    try:
        resp = requests.get(f"http://127.0.0.1:{QWEN36_BRIDGE_PORT}/health", timeout=2)
        return resp.status_code == 200 and resp.json().get("ok") is True
    except Exception:
        return False


def _uses_qwen36_bridge():
    from core.utils.config_utils import load_key

    base_url = str(load_key("api.base_url"))
    return f":{QWEN36_BRIDGE_PORT}" in base_url or f"/{QWEN36_BRIDGE_PORT}" in base_url


def _start_qwen36_bridge(model_id=None):
    global QWEN36_BRIDGE_PROCESS, QWEN36_BRIDGE_LOG

    from core.utils.config_utils import load_key

    base_url = str(load_key("api.base_url"))
    if str(QWEN36_BRIDGE_PORT) not in base_url:
        return  # not using local qwen36 bridge
    bridge_model_id = model_id or load_key("api.model")

    if _qwen36_bridge_health():
        console.print("[green]Local Qwen36 bridge already running[/]")
        return

    if not os.path.isfile(QWEN36_BRIDGE_SCRIPT):
        console.print(
            f"[yellow]Qwen36 bridge script not found at {QWEN36_BRIDGE_SCRIPT}, "
            f"skipping auto-start[/]"
        )
        return

    console.print(
        "[bold blue]Starting local Qwen36 bridge (model loading, ~30-60s)...[/]"
    )

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = QWEN36_BRIDGE_LD_PATH
    os.makedirs(os.path.join("output", "log"), exist_ok=True)
    QWEN36_BRIDGE_LOG = open(
        os.path.join("output", "log", "qwen36_bridge.log"),
        "a",
        encoding="utf-8",
    )

    QWEN36_BRIDGE_PROCESS = subprocess.Popen(
        [
            sys.executable,
            QWEN36_BRIDGE_SCRIPT,
            "--port", str(QWEN36_BRIDGE_PORT),
            "--auto-start-supervisor",
            "--ctx-size", "16384",
            "--supervisor-startup-timeout", "900",
            "--model-id", str(bridge_model_id),
        ],
        env=env,
        stdout=QWEN36_BRIDGE_LOG,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.time() + 900
    while time.time() < deadline:
        if _qwen36_bridge_health():
            console.print("[green]Qwen36 bridge ready, model loaded in GPU[/]")
            return
        if QWEN36_BRIDGE_PROCESS.poll() is not None:
            raise RuntimeError(
                f"Qwen36 bridge exited with code {QWEN36_BRIDGE_PROCESS.returncode} "
                f"during startup"
            )
        time.sleep(2)

    raise TimeoutError("Qwen36 bridge did not become healthy within 900 seconds")


def _stop_qwen36_bridge():
    global QWEN36_BRIDGE_PROCESS, QWEN36_BRIDGE_LOG
    if QWEN36_BRIDGE_PROCESS is None:
        if QWEN36_BRIDGE_LOG:
            QWEN36_BRIDGE_LOG.close()
            QWEN36_BRIDGE_LOG = None
        return
    if QWEN36_BRIDGE_PROCESS.poll() is not None:
        QWEN36_BRIDGE_PROCESS = None
        if QWEN36_BRIDGE_LOG:
            QWEN36_BRIDGE_LOG.close()
            QWEN36_BRIDGE_LOG = None
        return

    console.print("[yellow]Stopping Qwen36 bridge and supervisor (unloading model from GPU)...[/]")
    try:
        pgid = os.getpgid(QWEN36_BRIDGE_PROCESS.pid)
    except ProcessLookupError:
        QWEN36_BRIDGE_PROCESS = None
        if QWEN36_BRIDGE_LOG:
            QWEN36_BRIDGE_LOG.close()
            QWEN36_BRIDGE_LOG = None
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        QWEN36_BRIDGE_PROCESS.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        QWEN36_BRIDGE_PROCESS.wait(timeout=5)
    QWEN36_BRIDGE_PROCESS = None
    if QWEN36_BRIDGE_LOG:
        QWEN36_BRIDGE_LOG.close()
        QWEN36_BRIDGE_LOG = None
    console.print("[green]Qwen36 bridge stopped[/]")


def check_api():
    from core.utils.config_utils import load_key
    from openai import OpenAI
    import httpx

    api_key = load_key("api.key")
    base_url = load_key("api.base_url")
    model = load_key("api.model")

    if not api_key:
        return "api.key is not set", "请在 config.yaml 中设置 api.key，例如:\n  api:\n    key: 'sk-your-api-key'"

    if not base_url:
        return "api.base_url is not set", "请在 config.yaml 中设置 api.base_url，例如:\n  api:\n    base_url: 'https://api.deepseek.com'"

    if not model:
        return "api.model is not set", "请在 config.yaml 中设置 api.model，例如:\n  api:\n    model: 'deepseek-chat'"

    resolved_url = base_url
    if 'ark' in resolved_url:
        resolved_url = "https://ark.cn-beijing.volces.com/api/v3"
    elif 'v1' not in resolved_url:
        resolved_url = resolved_url.strip('/') + '/v1'

    # Bypass SOCKS proxy (httpx doesn't support socks:// without socksio)
    http_client = httpx.Client(proxy=None, transport=httpx.HTTPTransport())
    kwargs = dict(api_key=api_key, base_url=resolved_url, http_client=http_client)

    try:
        client = OpenAI(**kwargs)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return None, None
    except Exception as e:
        error_msg = str(e)
        hint = f"API connection failed: {error_msg}\n\n请检查 config.yaml 中的 API 配置:\n"
        if "401" in error_msg or "authentication" in error_msg.lower():
            hint += "  - api.key 可能无效，请确认密钥是否正确"
        elif "404" in error_msg or "model" in error_msg.lower():
            hint += f"  - api.model '{model}' 可能不可用，请确认模型名称是否正确"
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            hint += f"  - api.base_url '{base_url}' 无法连接，请检查网络或 URL 是否正确"
        else:
            hint += f"  - api.base_url: {base_url}\n  - api.model: {model}\n  - api.key: {api_key[:8]}..."
        return error_msg, hint


def main():
    args = parse_args()
    file = args.video_path_or_url
    llm_snapshot = None

    try:
        # Override indextts2.ports before anything reads the config.
        reserved_tts_ports = None
        if args.llm == "qwen36" and (args.mode == "tts" or args.tts_workers is not None):
            reserved_tts_ports = {QWEN36_BRIDGE_PORT}
        apply_tts_workers_override(args.tts_workers, reserved_ports=reserved_tts_ports)
        apply_edge_voice_override(args.voice)

        # Route this run to a named local LLM when requested.
        llm_snapshot = apply_llm_override(args)
        qwen36_model = None
        if args.llm == "qwen36":
            qwen36_model = args.llm_model or QWEN36_BRIDGE_MODEL
        uses_qwen36_bridge = _uses_qwen36_bridge()

        # Remote/API LLMs can be checked before local media work. Qwen36 is
        # deferred until after local ASR so the two GPU-heavy models do not
        # compete for VRAM.
        if not uses_qwen36_bridge:
            console.print(Panel("[bold]Checking API connectivity...[/]", border_style="blue"))
            error, hint = check_api()
            if error:
                raise RuntimeError(f"API check failed\n{hint}")
            console.print("[green]API connection OK[/]")

        from core.utils.onekeycleanup import cleanup
        from core import _2_asr, _7_sub_into_vid

        pre_llm_steps = [
            ("Processing input file", partial(process_input_file, file)),
            ("Transcribing with Whisper", _2_asr.transcribe),
        ]
        llm_text_steps = [
            ("Splitting sentences", split_sentences),
            ("Summarizing and translating", summarize_and_translate),
            ("Processing and aligning subtitles", process_and_align_subtitles),
            ("Merging subtitles to video", _7_sub_into_vid.merge_subtitles_to_video),
        ]

        prepare_output_folder("output")
        if uses_qwen36_bridge:
            run_steps(pre_llm_steps)

            # Start local LLM bridge after ASR and validate before first LLM step.
            _start_qwen36_bridge(model_id=qwen36_model)
            console.print(Panel("[bold]Checking Qwen36 bridge connectivity...[/]", border_style="blue"))
            error, hint = check_api()
            if error:
                raise RuntimeError(f"Qwen36 bridge check failed\n{hint}")
            console.print("[green]Qwen36 bridge connection OK[/]")

            run_steps(llm_text_steps)
        else:
            run_steps(pre_llm_steps + llm_text_steps)

        if args.mode == "tts":
            from core import _8_1_audio_task, _8_2_dub_chunks, _9_refer_audio
            from core import _10_gen_audio, _11_merge_audio, _12_dub_to_vid

            if uses_qwen36_bridge:
                _stop_qwen36_bridge()

            tts_steps = [
                ("Generating audio tasks", _8_1_audio_task.gen_audio_task_main),
                ("Generating dubbing chunks", _8_2_dub_chunks.gen_dub_chunks),
                ("Extracting reference audio", _9_refer_audio.extract_refer_audio_main),
                ("Generating TTS audio", _10_gen_audio.gen_audio),
                ("Merging full audio", _11_merge_audio.merge_full_audio),
                ("Merging dubbing to video", _12_dub_to_vid.merge_video_audio),
            ]
            run_steps(tts_steps)

        console.print(Panel("[bold green]All steps completed! Saving to history...[/]", border_style="green"))
        cleanup("history")
        console.print("[green]Done. Results saved to history/[/]")
    except Exception as e:
        console.print(Panel(f"[bold red]Pipeline failed[/]\n{e}", border_style="red"))
        sys.exit(1)
    finally:
        stop_optional_tts_workers()
        _stop_qwen36_bridge()
        try:
            _restore_config_values(llm_snapshot)
        except Exception as e:
            console.print(Panel(
                f"[yellow]Warning: failed to restore original LLM config[/]\n{e}",
                border_style="yellow",
            ))


if __name__ == "__main__":
    main()
