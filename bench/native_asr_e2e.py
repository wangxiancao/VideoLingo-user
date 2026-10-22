#!/usr/bin/env/env python3
"""End-to-end measurement harness for the native ASR engine.

Measures the full spoken-video path on a 60-minute video with the native
C++/CUDA engine as the ASR stage:

  ingest (ffmpeg video -> 16 kHz mono PCM)
  -> ASR (glm_asr_m0 transcribe --pcm-long: mel + encoder + decode)
  -> forced alignment (Qwen3-ForcedAligner via the pipeline's stack)
  -> per-chunk word timestamps

Outputs JSON with per-stage wall seconds and E2E RTFx. The E2E requirement
is throughput BELOW the ASR-only target (end-to-end may be slower); this
harness produces the acceptance evidence for that clause.

Usage:
  .venv/bin/python bench/native_asr_e2e.py --video /path/x.mp4 \
      --binary native/asr/build/glm_asr_m0 --minutes 60 --output out.json
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = Path("/root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512")
ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--binary", default=str(REPO / "native/asr/build/glm_asr_m0"))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="zh")
    args = parser.parse_args()

    stages = {}
    audio_seconds = args.minutes * 60.0
    workdir = Path("/tmp/m2-e2e")
    workdir.mkdir(exist_ok=True)
    pcm_path = workdir / "pcm.f32"

    # ---- stage 1: ingest (video -> 16 kHz mono PCM) ----
    t0 = time.perf_counter()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", args.video, "-ac", "1", "-ar", "16000",
         "-t", str(audio_seconds), "-y", "-f", "f32le", str(pcm_path)],
        check=True,
    )
    stages["ingest_s"] = time.perf_counter() - t0
    samples = pcm_path.stat().st_size // 4
    chunks = samples // 480000
    assert chunks == int(audio_seconds / 30), (samples, chunks)

    # ---- stage 2: ASR (native engine, includes mel + encoder + decode) ----
    import numpy as np
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_dir)
    pcm = np.fromfile(pcm_path, dtype=np.float32).reshape(chunks, 480000)
    t0 = time.perf_counter()
    inputs = processor.apply_transcription_request(list(pcm[:1]))
    prompt_ids = inputs.input_ids[0].tolist()
    prompt_file = workdir / "prompt.txt"
    prompt_file.write_text("\n".join(map(str, prompt_ids)))
    run = subprocess.run(
        [args.binary, "transcribe", "--model-dir", args.model_dir,
         "--pcm-long", str(pcm_path), "--prompt-ids", str(prompt_file)],
        check=True, capture_output=True, text=True, cwd=str(REPO),
    )
    stages["asr_s"] = time.perf_counter() - t0
    result = json.loads(run.stdout)
    transcripts = [
        processor.tokenizer.decode(ids, skip_special_tokens=True)
        for ids in result["per_chunk_ids"]
    ]
    tokens = sum(len(ids) for ids in result["per_chunk_ids"])

    # ---- stage 3: forced alignment (pipeline's aligner stack) ----
    import torch
    from transformers import AutoModelForTokenClassification, AutoProcessor as AP
    from pipeline.asr.glm import _align_batch  # reuse the pipeline's aligner helper

    t0 = time.perf_counter()
    aligner_processor = AP.from_pretrained(ALIGNER_ID)
    aligner_model = AutoModelForTokenClassification.from_pretrained(
        ALIGNER_ID, dtype=torch.bfloat16
    ).cuda().eval()

    class _M:  # minimal duck-type of pipeline.asr.glm.Models
        pass

    models = _M()
    models.aligner_processor = aligner_processor
    models.aligner_model = aligner_model

    word_rows = []
    batch = 16
    for base in range(0, chunks, batch):
        audios = list(pcm[base:base + batch])
        texts = transcripts[base:base + batch]
        t_batch = time.perf_counter()
        rows = _align_batch(models, audios, texts, args.language)
        stages["aligner_s"] = stages.get("aligner_s", 0.0) + (time.perf_counter() - t_batch)
        for row in rows:
            offset = base * 30.0  # chunk index within THIS batch loop is per-file
            for item in row:
                word_rows.append(
                    {"text": str(item.get("text", "")).strip(),
                     "start": float(item["start_time"]), "end": float(item["end_time"])}
                )
    total_e2e = stages["ingest_s"] + stages["asr_s"] + stages["aligner_s"]
    report = {
        "schema_version": 1,
        "tool": "native_asr_e2e",
        "video": str(args.video),
        "audio_minutes": args.minutes,
        "chunks": chunks,
        "asr_tokens": tokens,
        "stages_s": stages,
        "total_e2e_s": total_e2e,
        "e2e_rtfx": audio_seconds / total_e2e,
        "note": "E2E requirement is throughput BELOW the ASR-only target; "
                "native ASR-only official number is separate (bench --pcm-long)",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (workdir / "words.json").write_text(json.dumps(word_rows, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
