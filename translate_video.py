"""
Single-video translation script.
Runs the full text/subtitle pipeline and saves results to history/.

Usage:
    # Subtitles only (default), no dubbing
    python translate_video.py /path/to/video.mp4
    python translate_video.py --mode no-tts /path/to/video.mp4

    # Subtitles + dubbing (TTS method set as tts_method in config.yaml)
    python translate_video.py --mode tts https://www.youtube.com/watch?v=xxx

    Proxy note:
    Whisper loading (huggingface_hub) and the API both go through httpx, which
    validates every proxy env var eagerly. Two common Clash-related failures:
    1. ALL_PROXY=socks://127.0.0.1:7897 uses the invalid bare 'socks://' scheme
       (httpx only accepts socks5:// / socks5h://, and those need the 'socksio'
       extra), so it aborts model loading with "Unknown scheme for proxy URL".
    2. The proxy port is configured but the proxy app is not running, so every
       request burns a connect timeout (or hard-fails where there is no cache
       fallback).
    This script auto-neutralizes unusable/dead proxy variables INSIDE ITS OWN
    PROCESS ONLY and falls back to a direct connection. Your shell or global
    environment is never modified, so other tools keep using the proxy vars as
    before. No extra flag is needed; it just works.
"""

import argparse
import os
import sys
import ctypes

# ------------
# LD_LIBRARY_PATH fix: the Python environment's lib dir (conda, uv venv, or
# standard venv) must be searched before system libs so libstdc++ ABI matches.
# Setting os.environ is too late — the dynamic linker only reads LD_LIBRARY_PATH
# at process startup. So we detect the problem and re-exec ourselves.
# ------------
_env_lib = os.path.join(sys.prefix, "lib")

def _cxxabi_ok():
    """Check if libopenvino_c can be loaded with current libstdc++."""
    try:
        ctypes.CDLL("libopenvino_c.so.2430")
        return True
    except OSError:
        return False

if os.path.isdir(_env_lib) and _env_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    if not _cxxabi_ok():
        os.environ["LD_LIBRARY_PATH"] = _env_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        os.execv(sys.executable, [sys.executable] + sys.argv)

import shutil
import subprocess
import time
from functools import partial
from rich.panel import Panel
from rich.console import Console

console = Console()


def _sanitize_proxy_env():
    """Make this run immune to unusable proxy variables in the environment.

    huggingface_hub>=1.x (Whisper download/load) uses httpx, which validates
    EVERY proxy env var eagerly when a Client is built. Two failure modes:

    1. Bad scheme: a common Clash setup exports ``ALL_PROXY=socks://127.0.0.1``
       — the bare ``socks://`` scheme is invalid (httpx only accepts
       ``socks5://`` / ``socks5h://``, and even those need the optional
       'socksio' extra). httpx raises ``ValueError: Unknown scheme for proxy
       URL`` before a single request is sent, which surfaces here as "Whisper
       failed to load" even when the model is already fully cached.
    2. Dead proxy: the proxy variables point at a local proxy app (e.g. Clash)
       that is not running. Requests then burn connect timeouts and fail
       outright in code paths that have no cache fallback.

    We deliberately do NOT touch the user's shell or global config. We only
    drop unusable/dead proxy variables inside THIS process; direct connections
    (and the explicit ``proxy=None`` used for the LLM API) keep working.
    """
    try:
        import socksio  # noqa: F401
        socks_usable = True
    except Exception:
        socks_usable = False

    for var in ("ALL_PROXY", "all_proxy"):
        val = os.environ.get(var)
        if not val or "://" not in val:
            continue
        scheme = val.split("://", 1)[0].lower()
        if scheme.startswith("socks") and not socks_usable:
            os.environ.pop(var, None)
            console.print(
                f"[yellow][proxy-fix] Ignoring {var}={val} for this run only: httpx "
                f"cannot use a SOCKS proxy without the 'socksio' package, so it would "
                f"abort Whisper/API loading. Falling back to http_proxy/https_proxy. "
                f"Your shell environment is unchanged.[/]"
            )

    # Drop proxy variables whose server does not accept TCP connections
    # (proxy app not running). One probe per unique address, short timeout.
    import socket
    from urllib.parse import urlparse

    proxy_vars = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                  "ALL_PROXY", "all_proxy")

    def _proxy_reachable(value):
        try:
            url = value if "://" in value else f"http://{value}"
            parsed = urlparse(url)
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((parsed.hostname, port), timeout=1.5):
                return True
        except Exception:
            return False

    liveness = {}
    for var in proxy_vars:
        val = os.environ.get(var)
        if not val:
            continue
        if val not in liveness:
            liveness[val] = _proxy_reachable(val)
        if not liveness[val]:
            os.environ.pop(var, None)
            console.print(
                f"[yellow][proxy-fix] Ignoring {var}={val} for this run only: the "
                f"proxy server is unreachable (is the proxy app running?). Requests "
                f"will use a direct connection. Your shell environment is unchanged.[/]"
            )


_sanitize_proxy_env()


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
    return parser.parse_args()


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


def main():
    args = parse_args()
    file = args.video_path_or_url

    try:
        from core.utils.onekeycleanup import cleanup
        from core import _2_asr, _7_sub_into_vid

        text_steps = [
            ("Processing input file", partial(process_input_file, file)),
            ("Transcribing with Whisper", _2_asr.transcribe),
            ("Splitting sentences", split_sentences),
            ("Summarizing and translating", summarize_and_translate),
            ("Processing and aligning subtitles", process_and_align_subtitles),
            ("Merging subtitles to video", _7_sub_into_vid.merge_subtitles_to_video),
        ]

        prepare_output_folder("output")
        run_steps(text_steps)

        if args.mode == "tts":
            from core import _8_1_audio_task, _8_2_dub_chunks, _9_refer_audio
            from core import _10_gen_audio, _11_merge_audio, _12_dub_to_vid

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


if __name__ == "__main__":
    main()
