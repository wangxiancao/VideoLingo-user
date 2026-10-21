import sys
import time
from dataclasses import dataclass
from pathlib import Path

from pipeline.asr.contracts import TranscriptSegment, Transcription, Word
from pipeline.asr.language import aligner_language, normalize_language, resolve_language
from pipeline.contracts import StageError

SAMPLE_RATE = 16000
GLM_MODEL_ID = "ZhipuAI/GLM-ASR-Nano-2512"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
MAX_OVERRUN = 0.5


@dataclass
class Models:
    glm_processor: object
    glm_model: object
    aligner_processor: object
    aligner_model: object


def _resolve_models(cache_dir):
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise StageError(
            "modelscope is required for ASR weights; install it into the pipeline interpreter"
        ) from exc

    paths = {}
    for model_id in (GLM_MODEL_ID, ALIGNER_MODEL_ID):
        try:
            path = Path(
                snapshot_download(
                    model_id,
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                )
            )
        except Exception as exc:
            raise StageError(
                f"model weights for {model_id} are not available in {cache_dir}: {exc}; "
                f"download with: python -m bench.asr_rtfx download --cache-dir {cache_dir}"
            ) from exc
        weights = path / "model.safetensors"
        if not weights.is_file():
            raise StageError(
                f"model weights for {model_id} are incomplete in {path}: model.safetensors is missing"
            )
        paths[model_id] = str(path)
    return paths


def _load_models(paths):
    try:
        import torch
    except ImportError as exc:
        raise StageError(
            "torch is required for ASR; install it into the pipeline interpreter"
        ) from exc
    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoProcessor,
            GlmAsrForConditionalGeneration,
        )
    except ImportError as exc:
        raise StageError(
            "transformers with GLM-ASR and Qwen3-ASR support is required; "
            "install it into the pipeline interpreter"
        ) from exc

    if not torch.cuda.is_available():
        raise StageError("CUDA is not available; GLM-ASR transcription requires a GPU")

    try:
        glm_processor = AutoProcessor.from_pretrained(paths[GLM_MODEL_ID])
        glm_model = GlmAsrForConditionalGeneration.from_pretrained(
            paths[GLM_MODEL_ID],
            dtype="auto",
            device_map="cuda",
        )
        glm_model.eval()

        aligner_processor = AutoProcessor.from_pretrained(paths[ALIGNER_MODEL_ID])
        aligner_model = AutoModelForTokenClassification.from_pretrained(
            paths[ALIGNER_MODEL_ID],
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        aligner_model.eval()
        torch.cuda.synchronize()
    except Exception as exc:
        raise StageError(f"failed to load ASR models: {exc}") from exc

    return Models(glm_processor, glm_model, aligner_processor, aligner_model)


def _read_audio(path):
    try:
        import soundfile
    except ImportError as exc:
        raise StageError(
            "soundfile is required for ASR; install it into the pipeline interpreter"
        ) from exc

    try:
        samples, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise StageError(f"cannot read audio {path}: {exc}") from exc
    if sample_rate != SAMPLE_RATE:
        raise StageError(f"audio must be {SAMPLE_RATE} Hz, found {sample_rate} Hz")
    if getattr(samples, "ndim", 1) != 1:
        raise StageError("audio must be mono")
    if len(samples) == 0:
        raise StageError("audio is empty")
    return samples, sample_rate


def _chunks(samples, sample_rate, chunk_seconds):
    chunk_size = int(round(sample_rate * chunk_seconds))
    if chunk_size <= 0:
        raise StageError("asr.chunk_seconds must be positive")
    return [samples[index : index + chunk_size] for index in range(0, len(samples), chunk_size)]


def _batches(items, batch_size):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _transcribe_batch(models, audio_arrays, max_new_tokens):
    import torch

    inputs = models.glm_processor.apply_transcription_request(list(audio_arrays))
    inputs = inputs.to(models.glm_model.device, dtype=models.glm_model.dtype)
    with torch.inference_mode():
        output_ids = models.glm_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    transcripts = models.glm_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        strip_prefix=True,
    )
    return [text.strip() for text in transcripts]


def _align_batch(models, audio_arrays, transcripts, language):
    import torch

    aligner_inputs, word_lists = models.aligner_processor.prepare_forced_aligner_inputs(
        audio=list(audio_arrays),
        transcript=transcripts,
        language=language,
    )
    aligner_inputs = aligner_inputs.to(
        models.aligner_model.device,
        dtype=models.aligner_model.dtype,
    )
    with torch.inference_mode():
        outputs = models.aligner_model(**aligner_inputs)
    return models.aligner_processor.decode_forced_alignment(
        logits=outputs.logits,
        input_ids=aligner_inputs["input_ids"],
        word_lists=word_lists,
        timestamp_token_id=models.aligner_model.config.timestamp_token_id,
    )


def _clamp_words(timestamps, offset, chunk_duration):
    chunk_end = offset + chunk_duration
    words = []
    for item in timestamps:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = offset + float(item["start_time"])
        end = offset + float(item["end_time"])
        if start < offset - 0.05 or start > chunk_end:
            raise StageError(f"forced aligner word starts outside its chunk: {start:.3f}s")
        if end < start:
            raise StageError("forced aligner returned end before start")
        if end == start:
            end = min(start + 0.001, chunk_end)
            if end <= start:
                start = max(offset, chunk_end - 0.001)
                end = chunk_end
        end = min(end, chunk_end)
        if end <= start:
            start = min(start, chunk_end - 0.001)
            end = chunk_end
        start = round(start, 3)
        end = round(end, 3)
        if end <= start:
            end = round(min(chunk_end, start + 0.001), 3)
        if end <= start:
            start = round(max(offset, end - 0.001), 3)
        if end <= start:
            raise StageError("forced aligner produced a zero-length word")
        words.append(Word(text=text, start=start, end=end))
    if not words:
        raise StageError("forced aligner returned no words")
    previous_start = None
    for word in words:
        if previous_start is not None and word.start < previous_start:
            raise StageError("forced aligner word timestamps are not monotonic")
        previous_start = word.start
    return words


class GlmAsrBackend:
    def __init__(self, cache_dir, chunk_seconds=30.0, batch_size=1, max_new_tokens=512):
        self.cache_dir = Path(cache_dir)
        self.chunk_seconds = chunk_seconds
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def transcribe(self, audio_path, language=None):
        samples, sample_rate = _read_audio(audio_path)
        duration = len(samples) / sample_rate
        chunks = _chunks(samples, sample_rate, self.chunk_seconds)

        configured_language = None
        if language is not None and str(language).strip().lower() != "auto":
            configured_language = normalize_language(language)

        started = time.perf_counter()
        paths = _resolve_models(self.cache_dir)
        models = _load_models(paths)

        texts = []
        for batch in _batches(chunks, self.batch_size):
            texts.extend(_transcribe_batch(models, batch, self.max_new_tokens))
        if len(texts) != len(chunks):
            raise StageError("GLM-ASR returned a mismatched number of transcripts")

        spoken = [(index, text) for index, text in enumerate(texts) if text]
        if not spoken:
            raise StageError("GLM-ASR returned no speech for the audio")
        for index, text in enumerate(texts):
            if not text:
                print(
                    f"transcribe: no speech in chunk at {index * self.chunk_seconds:.1f}s",
                    file=sys.stderr,
                )

        language_code = configured_language or resolve_language(None, [text for _, text in spoken])
        aligner_language_name = aligner_language(language_code)

        segments = []
        for batch in _batches(spoken, self.batch_size):
            indices = [index for index, _ in batch]
            batch_texts = [text for _, text in batch]
            arrays = [chunks[index] for index in indices]
            aligned = _align_batch(models, arrays, batch_texts, aligner_language_name)
            for index, text, timestamps in zip(indices, batch_texts, aligned):
                offset = index * (int(round(sample_rate * self.chunk_seconds)) / sample_rate)
                chunk_duration = len(chunks[index]) / sample_rate
                words = _clamp_words(timestamps, offset, chunk_duration)
                segments.append(
                    TranscriptSegment(
                        start=words[0].start,
                        end=words[-1].end,
                        text=text,
                        words=tuple(words),
                    )
                )

        last_end = 0.0
        for segment in segments:
            if segment.start < last_end - 0.05:
                raise StageError("transcription segments overlap or regress")
            last_end = segment.end
        if last_end > duration + MAX_OVERRUN:
            raise StageError(
                f"transcription ends at {last_end:.3f}s, beyond audio duration {duration:.3f}s"
            )

        print(
            f"transcribe: {len(segments)} segments, "
            f"{sum(len(segment.words) for segment in segments)} words, "
            f"{time.perf_counter() - started:.2f}s",
            file=sys.stderr,
        )
        return Transcription(language=language_code, segments=tuple(segments))