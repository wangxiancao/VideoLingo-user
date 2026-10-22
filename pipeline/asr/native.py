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
from pipeline.contracts import StageError

MODEL_ID = "ZhipuAI/GLM-ASR-Nano-2512"
EOS_IDS = (59246, 59253, 59255)


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

        spoken = [(index, text) for index, text in enumerate(texts) if text]
        if not spoken:
            raise StageError("native ASR returned no speech for the audio")
        segments = []
        for index, text in spoken:
            offset = index * self.chunk_seconds
            segments.append(
                TranscriptSegment(
                    start=round(offset, 3),
                    end=round(offset + self.chunk_seconds, 3),
                    text=text,
                    words=(),
                )
            )
        elapsed = time.perf_counter() - started
        print(
            f"transcribe[native]: chunks={chunks} chunk_seconds={self.chunk_seconds:g} "
            f"audio_seconds={samples / sample_rate:.1f} elapsed={elapsed:.2f}s "
            f"(engine subprocess; engine-side timing lives in its own bench)",
            file=sys.stderr,
        )
        return Transcription(language=language or "auto", segments=tuple(segments))
