import re
import torch
import librosa
from core.utils import *
from core.utils.models import *
from rich import print as rprint

# ----------------------------------------
# Singleton pipeline cache
# ----------------------------------------
_PIPE = None
TIMESTAMP_RESET_THRESHOLD = 1.0

# Process-level memoization for the broken return_language + word-timestamps
# combined pass. transformers 5.14.1 raises an IndexError on EVERY combined
# attempt (structural defect), so once observed we skip the doomed attempt for
# the rest of the process and go straight to word-only + probe. A future fixed
# transformers never sets the flag in a fresh process, and if the combined
# attempt ever succeeds the flag stays False (fallback still reachable).
_WORD_LANG_MODE_BROKEN = False

# transformers returns full language names from the ASR pipeline (e.g.
# "chinese"); map the common ones back to the ISO 639-1 codes the rest of the
# pipeline (joiner lists, spacy model map, prompts) expects.
_LANGUAGE_ALIASES = {
    "english": "en", "chinese": "zh", "mandarin": "zh", "cantonese": "zh",
    "yue": "zh", "german": "de", "spanish": "es", "russian": "ru",
    "korean": "ko", "french": "fr", "japanese": "ja", "portuguese": "pt",
    "turkish": "tr", "polish": "pl", "catalan": "ca", "dutch": "nl",
    "arabic": "ar", "swedish": "sv", "italian": "it", "indonesian": "id",
    "hindi": "hi", "finnish": "fi", "vietnamese": "vi", "hebrew": "he",
    "ukrainian": "uk", "greek": "el", "malay": "ms", "czech": "cs",
    "romanian": "ro", "danish": "da", "hungarian": "hu", "tamil": "ta",
    "norwegian": "no", "thai": "th", "urdu": "ur", "croatian": "hr",
    "bulgarian": "bg", "lithuanian": "lt", "latvian": "lv",
    "macedonian": "mk", "slovak": "sk", "estonian": "et",
    "persian": "fa", "bengali": "bn", "telugu": "te", "serbian": "sr",
    "slovenian": "sl", "marathi": "mr", "gujarati": "gu", "kannada": "kn",
    "punjabi": "pa", "malayalam": "ml", "burmese": "my", "nepali": "ne",
    "sinhala": "si", "khmer": "km", "lao": "lo", "mongolian": "mn",
    "amharic": "am", "tagalog": "tl", "azerbaijani": "az", "basque": "eu",
    "belarusian": "be", "bosnian": "bs", "galician": "gl", "georgian": "ka",
    "icelandic": "is", "kazakh": "kk", "kyrgyz": "ky", "maltese": "mt",
    "occitan": "oc", "pashto": "ps", "sundanese": "su", "swahili": "sw",
    "tatar": "tt", "tibetan": "bo", "turkmen": "tk", "uyghur": "ug",
    "uzbek": "uz", "welsh": "cy", "yiddish": "yi",
}


def _normalize_lang_token(token):
    """Normalize a Whisper language result into an ISO 639-1 code (or None)."""
    if token is None:
        return None
    tok = str(token).strip().lower().replace("_", "-")
    if tok in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[tok]
    base = tok.split("-")[0]
    if re.fullmatch(r"[a-z]{2}", base):
        return base
    if base in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[base]
    return None


def _probe_detected_language(pipe, audio, generate_kwargs, probe_seconds=30.0):
    """Best-effort Whisper language detection that avoids the broken path.

    transformers <=5.x crashes when return_language=True is combined with
    return_timestamps="word": pipeline postprocess prepends the language token
    to the token stream without the matching token timestamp, so the word
    branch of tokenizer._decode_asr indexes one past the timestamp list.
    Running chunk-level timestamps (return_timestamps=True) + return_language
    avoids that indexing entirely, so a short leading slice can be probed to
    recover the real detected language without a full second pass.

    Returns an ISO 639-1 code (or None) — never raises.
    """
    try:
        probe_audio = audio[: int(probe_seconds * 16000)]
        probe_result = pipe(
            probe_audio,
            return_timestamps=True,
            chunk_length_s=30,
            return_language=True,
            generate_kwargs=generate_kwargs,
        )
    except Exception:
        return None
    token = probe_result.get("language")
    if token is None:
        for chunk in probe_result.get("chunks", []):
            if chunk.get("language"):
                token = chunk["language"]
                break
    return _normalize_lang_token(token)


def _typical_word_span(words):
    """Robust estimate of the typical word span for a clip.

    Median span, clamped into [0.25, 2.0]s so a single bad word never
    inflates/deflates the anchor-detection budget for the whole clip.
    """
    spans = sorted(w["end"] - w["start"] for w in words)
    if not spans:
        return 0.0
    n = len(spans)
    median = spans[n // 2] if n % 2 == 1 else (spans[n // 2 - 1] + spans[n // 2]) / 2.0
    return max(min(median, 2.0), 0.25)


def clean_whisper_words(raw_words, clip_begin, clip_end):
    """Drop/clamp Whisper word-timestamp artifacts against LOCAL references.

    Reset detection runs in Whisper's ORIGINAL emission order, before any
    sorting, comparing each word against the previous KEPT word:

    - reset artifact: a word whose start (or end) regressed by more than
      TIMESTAMP_RESET_THRESHOLD vs the previous kept word is dropped. The
      reference is local, so one wide/anchor word earlier in the stream can
      never erase the rest of the clip through a running maximum end.
    - wide-span anchor: a word whose span is absurdly wide vs the clip's
      typical span while the next word starts (and ends) far before its end
      is clamped to the next word's start so it cannot poison later words.
    - identical-window duplicates: repeated words collapsed onto the exact
      same (start, end) window are reduced to a single word instead of
      surviving as a zero-width chain.

    After that the words are sorted and clamped into the clip window; small
    overlaps are clipped to keep strict monotonicity, but this post-sort pass
    NEVER drops a word based on a running maximum end.
    """
    typical_span = _typical_word_span(raw_words)
    kept = []
    dropped = 0
    prev = None
    for i, w in enumerate(raw_words):
        w_start, w_end = w["start"], w["end"]
        next_w = raw_words[i + 1] if i + 1 < len(raw_words) else None

        # Reset artifact vs the previous KEPT word (local reference).
        if (
            prev is not None
            and (
                w_start < prev["start"] - TIMESTAMP_RESET_THRESHOLD
                or w_end < prev["end"] - TIMESTAMP_RESET_THRESHOLD
            )
        ):
            dropped += 1
            continue

        # Repeated word collapsed onto the exact same window as the previous
        # kept word: drop the duplicate so no zero-width chain survives.
        if (
            prev is not None
            and w_start == prev["start"]
            and w_end == prev["end"]
        ):
            dropped += 1
            continue

        # Wide-span anchor vs local neighbors: the word's end sails past the
        # next word's whole window by more than a second while its own span
        # dwarfs the clip's typical span. Clamp its end down to the next
        # word's start instead of letting it poison later comparisons.
        if (
            w_end - w_start > typical_span + TIMESTAMP_RESET_THRESHOLD
            and next_w is not None
            and next_w["start"] < w_end - TIMESTAMP_RESET_THRESHOLD
            and next_w["end"] < w_end - TIMESTAMP_RESET_THRESHOLD
        ):
            w_end = max(w_start, next_w["start"])
            w["end"] = w_end

        w["start"] = w_start
        w["end"] = w_end
        kept.append(w)
        prev = w

    # Sort and clamp into the clip window; overlaps are clipped, never
    # dropped based on a running maximum end.
    kept.sort(key=lambda w: (w["start"], w["end"]))
    words = []
    prev_end = None
    for w in kept:
        w_start, w_end = w["start"], w["end"]

        # Word entirely outside the clip window is a timestamp anomaly; drop it
        # rather than emit a clamped zero-width artifact at the boundary.
        if w_start > clip_end or w_end < clip_begin:
            dropped += 1
            continue

        w_start = max(w_start, clip_begin)
        if clip_end > clip_begin:
            w_end = min(w_end, clip_end)
        if w_end < w_start:
            w_end = w_start

        if prev_end is not None and w_start < prev_end:
            w_start = prev_end
            w_end = max(w_end, w_start)

        w["start"] = w_start
        w["end"] = w_end
        words.append(w)
        prev_end = w_end

    return words, dropped


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

    # Pass the configured source language only when the user did not ask for
    # auto detection; Whisper otherwise auto-detects per segment.
    whisper_lang = (load_key("whisper.language") or "").strip().lower()
    generate_kwargs = {}
    if whisper_lang and whisper_lang != "auto":
        generate_kwargs["language"] = whisper_lang

    # Transcribe with word-level timestamps (DTW cross-attention alignment).
    # return_language=True is requested so Whisper's auto-detected language can
    # be written back to the config. transformers (<=5.x) has a defect in this
    # mode for long-form seq2seq Whisper: pipeline postprocess prepends the
    # detected <|lang|> token to the token stream WITHOUT a matching
    # token_timestamps entry, so tokenizer._decode_asr's word branch reads
    # token_timestamps[i] one index past the list and raises IndexError. The
    # word timestamps themselves are then unusable anyway (shifted by one
    # token), so the robust fix is to fall back to a word-only pass and derive
    # the language from a short probe / previous config afterwards. The
    # combined attempt always fails on installed transformers, so it is skipped
    # after the first observed failure for the rest of this process (see
    # _WORD_LANG_MODE_BROKEN); the try/except stays for forward compatibility
    # with a fixed transformers.
    rprint(f"[green]▶️ Whisper transcribing segment {start:.2f}s to {end:.2f}s...[/green]")
    global _WORD_LANG_MODE_BROKEN
    if not _WORD_LANG_MODE_BROKEN:
        try:
            result = pipe(
                audio,
                return_timestamps="word",
                chunk_length_s=30,
                return_language=True,
                generate_kwargs=generate_kwargs,
            )
        except Exception as e:
            _WORD_LANG_MODE_BROKEN = True
            rprint(
                f"[yellow]⚠️ transformers word+language pass failed ({e}); "
                f"disabling combined mode for this process and retrying "
                f"without return_language.[/yellow]"
            )

    if _WORD_LANG_MODE_BROKEN:
        result = pipe(
            audio,
            return_timestamps="word",
            chunk_length_s=30,
            return_language=False,
            generate_kwargs=generate_kwargs,
        )

    # Whisper auto-detects language (or reports the forced one); save the real
    # value, normalized to ISO 639-1, to the config. With word-level
    # timestamps the merged result has no top-level 'language' key: each word
    # chunk carries a 'language' field (full name) when return_language=True.
    lang_token = result.get("language")
    if lang_token is None:
        for chunk in result.get("chunks", []):
            if chunk.get("language"):
                lang_token = chunk["language"]
                break
    detected_lang = _normalize_lang_token(lang_token)
    if detected_lang is None:
        # Falling back from return_language=True → the word-only pass above
        # carries no language. Probe a short leading slice with chunk-level
        # timestamps (never the broken word branch) to recover real
        # auto-detection; if that also fails, keep the forced config value,
        # the previously detected one, or 'en' as last resort.
        detected_lang = _probe_detected_language(pipe, audio, generate_kwargs)
    if detected_lang is None:
        try:
            prev_detected = (load_key("whisper.detected_language") or "").strip().lower()
        except (KeyError, ValueError, TypeError):
            # Legacy config without the key (or a malformed value); treat as unset.
            prev_detected = ""
        if whisper_lang and whisper_lang != "auto":
            detected_lang = whisper_lang
        elif prev_detected and re.fullmatch(r"[a-z]{2}", prev_detected):
            detected_lang = prev_detected
        else:
            detected_lang = "en"
    rprint(f"[cyan]🌐 Detected language: {detected_lang}[/cyan]")

    update_key("whisper.detected_language", detected_lang)
    if load_key("whisper.language") == "auto":
        update_key("whisper.language", detected_lang)

    # Build word list from Whisper chunks
    # Each chunk: {"text": "Hello", "timestamp": (0.0, 0.5)}
    clip_begin, clip_end = float(start), float(end)
    raw_words = []
    for chunk in result.get("chunks", []):
        word_text = chunk.get("text", "").strip()
        ts_start, ts_end = chunk.get("timestamp", (None, None))

        if not word_text:
            continue

        # Whisper timestamps are relative to the chunk; offset by segment start
        w_start = (ts_start if ts_start is not None else 0.0) + clip_begin
        w_end = (ts_end if ts_end is not None else w_start) + clip_begin
        raw_words.append({"word": word_text, "start": w_start, "end": max(w_end, w_start)})

    # Clean Whisper word timestamps against LOCAL references (see
    # clean_whisper_words): reset artifacts and wide-span anchors are detected
    # in Whisper's ORIGINAL emission order before any sorting, using the
    # previous kept word as the reference, so a single hallucinated word can
    # never wipe out the rest of the clip through a running maximum end.
    words, dropped_words = clean_whisper_words(raw_words, clip_begin, clip_end)

    if dropped_words:
        rprint(f"[yellow]⚠️ Dropped {dropped_words} word(s) from Whisper timestamp resets.[/yellow]")

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


def _selftest():
    """Synthetic unit checks for clean_whisper_words (no model/GPU needed).

    Run with: .venv/bin/python -m core.asr_backend.whisper_asr
    """
    clip_begin, clip_end = 0.0, 60.0

    def run(words):
        cleaned, dropped = clean_whisper_words(
            [{"word": t, "start": s, "end": e} for t, s, e in words],
            clip_begin, clip_end,
        )
        return [(w["word"], w["start"], w["end"]) for w in cleaned], dropped

    # (a) A wide anchor word [5, 19.5] must NOT wipe out subsequent normal
    #     monotonic words; the anchor gets its end clamped, the rest survive.
    out, dropped = run([
        ("welcome", 5.0, 19.5),
        ("to", 12.0, 12.5),
        ("our", 12.8, 13.2),
        ("channel", 14.0, 14.4),
    ])
    texts = [t for t, _, _ in out]
    assert "to" in texts and "our" in texts and "channel" in texts, out
    assert all(e > s for _, s, e in out), out
    assert all(s >= prev_e for (_, s, _), (_, prev_e, _) in
               zip(out, [(None, -1.0, None)] + out[:-1])), out

    # (b) Identical-timestamp runs collapse to a single word: no zero-width
    #     chain survives.
    out, dropped = run([("thank", 10.0, 10.0)] * 5)
    assert len(out) == 1 and out[0][1] == out[0][2] == 10.0, out

    # (b2) A reset word that regressed far behind the previous kept word is
    #      dropped instead of re-sorting to the front.
    out, dropped = run([
        ("first", 10.0, 10.5),
        ("second", 11.0, 11.5),
        ("hallucination", 0.2, 0.6),
        ("third", 12.0, 12.5),
    ])
    texts = [t for t, _, _ in out]
    assert "hallucination" not in texts, out
    assert texts == ["first", "second", "third"], out

    # (c) Normal monotonic speech passes through (nearly) untouched.
    normal = [("w%d" % i, round(i * 0.6, 2), round(i * 0.6 + 0.4, 2)) for i in range(5)]
    out, dropped = run(normal)
    assert [t for t, _, _ in out] == [t for t, _, _ in normal], out
    assert all(
        abs(s - ns) < 1e-9 and abs(e - ne) < 1e-9
        for (_, s, e), (_, ns, ne) in zip(out, normal)
    ), out

    print("whisper_asr clean_whisper_words selftest: OK")


if __name__ == "__main__":
    _selftest()