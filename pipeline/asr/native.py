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
DEFAULT_ENGINE_VRAM_BUDGET_MIB = 16384


def _wait_for_vram_return(timeout_s=10.0):
    """Block until driver-level free VRAM stabilizes (release handoff)."""

    def free_mib():
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return int(probe.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001 - probe is best-effort
            return None

    deadline = time.time() + timeout_s
    previous = free_mib()
    while previous is not None and time.time() < deadline:
        time.sleep(0.5)
        current = free_mib()
        if current is not None and current <= previous + 256:
            return  # stable: the release has landed
        previous = current


def _engine_vram_budget_mib():
    """Cap the engine's VRAM reservation to what is actually free.

    The engine eagerly allocates arenas up to its budget; on a shared GPU
    (e.g. an infinity_emb tenant) the default 16 GiB cap OOMs. Query free
    memory, keep a 1 GiB margin for the engine's own runtime overhead
    beyond its estimate (the aligner and the LLM service only load after
    the engine subprocess has exited, so they need no margin here), and
    never go below a floor that still fits weights plus a small batch.
    On probe failure, keep the exclusive-GPU default.
    """
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        free_mib = int(probe.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - probe is best-effort
        return DEFAULT_ENGINE_VRAM_BUDGET_MIB
    return max(2048, min(DEFAULT_ENGINE_VRAM_BUDGET_MIB, free_mib - 1024))


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
                 "--pcm-long", str(pcm_path), "--prompt-ids", str(prompt_path),
                 "--max-vram-mib", str(_engine_vram_budget_mib())],
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

        # Aligner batch is bounded to fit the 16 GiB pipeline-peak budget:
        # b20 needs ~9.7 GiB reserved, which collides with the LLM replicas
        # spawning right after the release (V4-7 O4 measured 16.4 GiB peak).
        # b10 halves the footprint for ~2-3s more alignment time.
        aligner_batch = 4
        aligned = _batched_results(spoken, aligner_batch, align_batch, "alignment")
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
        # Hand the GPU back before the LLM replicas load (V4-7 O2): the
        # aligner's torch cache would otherwise pin ~11 GiB of reserved
        # memory through the translate stage. Wait until driver-level free
        # memory stabilizes, so replicas never spawn against a half-released
        # card (measured: ~16.6 GiB transient peak without the wait vs the
        # 16 GiB pipeline budget).
        del aligner
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - release is best-effort
            pass
        _wait_for_vram_return()
        return Transcription(language=language_code, segments=tuple(segments))
