import wave

from pipeline.asr.factory import create_backend
from pipeline.contracts import StageError

MAX_AUDIO_OVERRUN = 0.5
TIMESTAMP_TOLERANCE = 0.05


def _audio_duration(path):
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
    except (OSError, wave.Error) as exc:
        raise StageError(f"cannot read audio {path}: {exc}") from exc
    if channels != 1:
        raise StageError(f"audio must be mono, found {channels} channels")
    if sample_rate != 16000:
        raise StageError(f"audio must be 16 kHz, found {sample_rate} Hz")
    if frames <= 0:
        raise StageError("audio is empty")
    return frames / sample_rate


def _payload(result, audio_duration):
    language = str(result.language or "").strip()
    if not language:
        raise StageError("ASR backend returned no language")
    if not result.segments:
        raise StageError("ASR backend returned no segments")

    segments = []
    previous_end = None
    last_end = 0.0
    for index, segment in enumerate(result.segments):
        text = str(segment.text or "").strip()
        if not text:
            raise StageError(f"ASR segment {index} has empty text")
        if not segment.words:
            raise StageError(f"ASR segment {index} has no words")
        start = round(float(segment.start), 3)
        end = round(float(segment.end), 3)
        if start < 0 or end <= start:
            raise StageError(f"ASR segment {index} has invalid timing")
        if previous_end is not None and start < previous_end - TIMESTAMP_TOLERANCE:
            raise StageError(f"ASR segment {index} overlaps the previous segment")

        words = []
        previous_word_start = None
        for word_index, word in enumerate(segment.words):
            word_text = str(word.text or "").strip()
            if not word_text:
                raise StageError(f"ASR segment {index} word {word_index} has empty text")
            word_start = round(float(word.start), 3)
            word_end = round(float(word.end), 3)
            if word_end <= word_start:
                raise StageError(f"ASR segment {index} word {word_index} has invalid timing")
            if (
                word_start < start - TIMESTAMP_TOLERANCE
                or word_end > end + TIMESTAMP_TOLERANCE
            ):
                raise StageError(
                    f"ASR segment {index} word {word_index} is outside its segment"
                )
            if previous_word_start is not None and word_start < previous_word_start:
                raise StageError(
                    f"ASR segment {index} word {word_index} regresses"
                )
            previous_word_start = word_start
            words.append({"start": word_start, "end": word_end, "text": word_text})

        segments.append({"start": start, "end": end, "text": text, "words": words})
        previous_end = end
        last_end = end

    if last_end > audio_duration + MAX_AUDIO_OVERRUN:
        raise StageError(
            f"transcription ends at {last_end:.3f}s, beyond audio duration {audio_duration:.3f}s"
        )
    return {"schema_version": 1, "language": language, "segments": segments}


def run(ctx) -> None:
    audio_path = ctx.path("audio")
    audio_duration = _audio_duration(audio_path)

    backend = create_backend(ctx.config)
    configured = ctx.config.get("asr.language", "auto")
    language = None if str(configured).strip().lower() in ("", "auto") else str(configured)
    result = backend.transcribe(audio_path, language=language)
    ctx.write("transcript", _payload(result, audio_duration))