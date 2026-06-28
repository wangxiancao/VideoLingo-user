"""
Single-video translation script.
Runs the full text/subtitle pipeline and saves results to history/.

Usage:
    python translate_video.py <video_path>
    python translate_video.py --mode no-tts /path/to/video.mp4
    python translate_video.py --mode tts https://www.youtube.com/watch?v=xxx
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
from functools import partial
from rich.panel import Panel
from rich.console import Console

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


def stop_optional_tts_workers():
    try:
        from core.tts_backend.indextts2_tts import stop_indextts2_worker

        stop_indextts2_worker()
    except Exception as e:
        console.print(Panel(
            f"[yellow]Warning: failed to stop optional TTS worker[/]\n{e}",
            border_style="yellow",
        ))


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

    console.print(Panel("[bold]Checking API connectivity...[/]", border_style="blue"))
    error, hint = check_api()
    if error:
        console.print(Panel(f"[bold red]API check failed[/]\n{hint}", border_style="red"))
        sys.exit(1)
    console.print("[green]API connection OK[/]")

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

    try:
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
    except Exception:
        sys.exit(1)
    finally:
        stop_optional_tts_workers()


if __name__ == "__main__":
    main()
