"""Native ASR backend: the self-developed CUDA engine behind the pipeline seam.

Bridges the C++/CUDA `glm_asr_m0` binary into the `AsrBackend` protocol: the
audio file is decoded to 16 kHz mono chunks, the processor builds the shared
transcription prompt once, the engine transcribes all chunks in one batched
run (mel + encoder + batched prefill + lockstep decode), and results map back
to per-chunk segments identical to the transformers backend. Quality-gated
against the BF16 reference (spec #37 Q3: character match >= 99%); see
docs/agents/native-asr-m0.md and native-asr-m2.md.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pipeline.asr.contracts import Transcription, TranscriptSegment
from pipeline.asr.glm import (
    ALIGNER_MODEL_ID,
    _align_batch,
    _batched_results,
    _clamp_words,
)
from pipeline.asr.language import aligner_language, resolve_language
from pipeline.contracts import StageError

MODEL_ID = "ZhipuAI/GLM-ASR-Nano-2512"
EOS_IDS = (59246, 59253, 59255)
MAX_OVERRUN = 0.5


class _AlignerOnly:
    """The subset of glm.Models that _align_batch touches."""

    def __init__(self, aligner_processor, aligner_model):
        self.aligner_processor = aligner_processor
        self.aligner_model = aligner_model


def _load_aligner(cache_dir):
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise StageError("modelscope is required for aligner weights") from exc
    try:
        path = Path(
            snapshot_download(ALIGNER_MODEL_ID, cache_dir=str(cache_dir),
                              local_files_only=True)
        )
    except Exception as exc:
        raise StageError(
            f"aligner weights for {ALIGNER_MODEL_ID} are not available in "
            f"{cache_dir}: {exc}"
        ) from exc
    import torch
    from transformers import AutoModelForTokenClassification, AutoProcessor

    if not torch.cuda.is_available():
        raise StageError("CUDA is not available; forced alignment requires a GPU")
    processor = AutoProcessor.from_pretrained(str(path))
    model = AutoModelForTokenClassification.from_pretrained(
        str(path), dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    torch.cuda.synchronize()
    return _AlignerOnly(processor, model)


class NativeAsrBackend:
    def __init__(self, cache_dir, binary=None, model_dir=None, chunk_seconds=30.0):
        self.cache_dir = Path(cache_dir).expanduser()
        self.chunk_seconds = float(chunk_seconds)
        self.binary = Path(binary) if binary else Path("native/asr/build/glm_asr_m0")
        self.model_dir = Path(model_dir) if model_dir else self.cache_dir / MODEL_ID

    def transcribe(self, audio_path, language=None):
        if not self.binary.is_file():
            raise StageError(f"native ASR binary not found: {self.binary}")
        if not self.model_dir.is_dir():
            raise StageError(f"native ASR model dir not found: {self.model_dir}")

        sample_rate = 16000
        chunk_samples = int(round(sample_rate * self.chunk_seconds))
        started = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="native-asr-") as workdir:
            work = Path(workdir)
            pcm_path = work / "audio.p32"
            decode = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar",
                 str(sample_rate), "-f", "f32le", "-y", str(pcm_path)],
                capture_output=True,
            )
            if decode.returncode != 0:
                raise StageError(f"ffmpeg failed to decode audio: {decode.stderr.decode()[:200]}")
            samples = pcm_path.stat().st_size // 4
            if samples == 0:
                raise StageError("decoded audio is empty")
            chunks = (samples + chunk_samples - 1) // chunk_samples
            if chunks < 1 or chunks > 128:
                raise StageError(f"native ASR supports 1..128 chunks, got {chunks}")
            total_samples = chunks * chunk_samples
            if samples < total_samples:
                # Real audio is never an exact 30 s multiple and the engine
                # requires whole chunks; pad the tail with silence (at most
                # one extra chunk, ~0.4% compute; silence produces no words).
                with open(pcm_path, "ab") as handle:
                    handle.write(b"\x00" * ((total_samples - samples) * 4))

            # Build the shared transcription prompt once (tokenizer stays on
            # the tool side; the native runtime is tokenizer-free).
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(str(self.model_dir))
            import numpy as np

            first = np.fromfile(pcm_path, dtype=np.float32)[:chunk_samples]
            inputs = processor.apply_transcription_request([first])
            prompt_ids = inputs.input_ids[0].tolist()
            prompt_path = work / "prompt.txt"
            prompt_path.write_text("\n".join(map(str, prompt_ids)))

            run = subprocess.run(
                [str(self.binary), "transcribe", "--model-dir", str(self.model_dir),
                 "--pcm-long", str(pcm_path), "--prompt-ids", str(prompt_path)],
                capture_output=True, text=True, timeout=7200,
            )
            if run.returncode != 0:
                raise StageError(f"native ASR engine failed: {run.stderr.strip()[:300]}")
            result = json.loads(run.stdout)
            per_chunk = result["per_chunk_ids"]
            if len(per_chunk) != chunks:
                raise StageError(
                    f"native ASR returned {len(per_chunk)} transcripts for {chunks} chunks")

            texts = [
                processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
                for ids in per_chunk
            ]
            # keep the (padded) samples for the alignment tail below
            pcm_all = np.fromfile(pcm_path, dtype=np.float32)
            real_samples = samples

        spoken = [(index, text) for index, text in enumerate(texts) if text]
        if not spoken:
            raise StageError("native ASR returned no speech for the audio")
        for index, text in enumerate(texts):
            if not text:
                print(
                    f"transcribe: no speech in chunk at "
                    f"{index * self.chunk_seconds:.1f}s",
                    file=sys.stderr,
                )

        # Word timestamps come from the same Qwen3-ForcedAligner tail the
        # transformers backend uses (ADR-0002: the aligner stays on
        # transformers). The engine subprocess has exited by now, so its VRAM
        # is free before the aligner loads.
        language_code = resolve_language(None, [text for _, text in spoken])
        aligner_language_name = aligner_language(language_code)
        aligner = _load_aligner(self.cache_dir)
        duration = real_samples / sample_rate

        def align_batch(batch):
            indices = [index for index, _ in batch]
            batch_texts = [text for _, text in batch]
            arrays = [pcm_all[index * chunk_samples:(index + 1) * chunk_samples]
                      for index in indices]
            return _align_batch(aligner, arrays, batch_texts, aligner_language_name)

        aligned = _batched_results(spoken, 20, align_batch, "alignment")
        segments = []
        for (index, text), timestamps in zip(spoken, aligned):
            offset = index * self.chunk_seconds
            chunk_duration = self.chunk_seconds
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
                f"transcription ends at {last_end:.3f}s, beyond audio duration "
                f"{duration:.3f}s"
            )

        elapsed = time.perf_counter() - started
        print(
            f"transcribe[native]: chunks={chunks} chunk_seconds={self.chunk_seconds:g} "
            f"audio_seconds={samples / sample_rate:.1f} segments={len(segments)} "
            f"words={sum(len(segment.words) for segment in segments)} "
            f"elapsed={elapsed:.2f}s",
            file=sys.stderr,
        )
        return Transcription(language=language_code, segments=tuple(segments))
