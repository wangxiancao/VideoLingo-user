import torch
import librosa
from core.utils import *
from core.utils.models import *
from rich import print as rprint

# ----------------------------------------
# Singleton pipeline cache
# ----------------------------------------
_PIPE = None


def get_model():
    """Load Whisper pipeline once and cache it globally."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    model_name = load_key("whisper_asr.model")
    device_cfg = load_key("whisper_asr.device")

    if device_cfg == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    rprint(f"[cyan]🔄 Loading Whisper: {model_name}, device={device}[/cyan]")

    from transformers import pipeline as hf_pipeline

    _PIPE = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
    )

    rprint(f"[green]✅ Whisper loaded[/green]")
    return _PIPE


@except_handler("Whisper processing error:")
def transcribe_audio_whisper(raw_audio_file, vocal_audio_file, start, end) -> dict:
    """
    Transcribe audio using Whisper with word-level timestamps via DTW.

    Args:
        raw_audio_file: Path to raw audio file (unused, kept for API compatibility)
        vocal_audio_file: Path to vocal audio file
        start: Start time in seconds (offset)
        end: End time in seconds

    Returns:
        Dict with 'language' and 'segments' matching the expected output contract.
    """
    # Load audio segment at 16kHz mono
    audio, sr = librosa.load(
        vocal_audio_file, sr=16000, offset=start, duration=end - start, mono=True
    )

    # Get or create the pipeline singleton
    pipe = get_model()

    # Transcribe with word-level timestamps (DTW cross-attention alignment)
    rprint(f"[green]▶️ Whisper transcribing segment {start:.2f}s to {end:.2f}s...[/green]")
    result = pipe(audio, return_timestamps="word", chunk_length_s=30)

    # Whisper auto-detects language; save to config
    detected_lang = result.get("language", "en") or "en"
    rprint(f"[cyan]🌐 Detected language: {detected_lang}[/cyan]")

    update_key("whisper.detected_language", detected_lang)
    whisper_lang = load_key("whisper.language")
    if whisper_lang == "auto":
        update_key("whisper.language", detected_lang)

    # Build word list from Whisper chunks
    # Each chunk: {"text": "Hello", "timestamp": (0.0, 0.5)}
    words = []
    for chunk in result.get("chunks", []):
        word_text = chunk.get("text", "").strip()
        ts_start, ts_end = chunk.get("timestamp", (None, None))

        if not word_text:
            continue

        # Whisper timestamps are relative to the chunk; offset by segment start
        word_start = (ts_start if ts_start is not None else 0.0) + start
        word_end = (ts_end if ts_end is not None else word_start) + start

        words.append({
            "word": word_text,
            "start": word_start,
            "end": word_end,
        })

    # Build one segment with all words (matching process_transcription contract)
    if words:
        segment = {
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": result.get("text", "").strip(),
            "words": words,
        }
    else:
        segment = {
            "start": start,
            "end": end,
            "text": result.get("text", "").strip(),
            "words": [],
        }

    return {
        "language": detected_lang,
        "segments": [segment],
    }
