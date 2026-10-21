# Tests

T1 has one focused test file: `tests/test_t1_runner.py` (29 tests). It checks the stage runner and the artifact validators without running the full pipeline.

T2a adds `tests/test_t2a_rtfx.py` (6 tests). It checks audio chunking, RTFx summary math, and the preflight/run error paths without loading models.

T3 adds `tests/test_t3_ingest.py` (4 tests). It runs ingest through the stage runner for a local file and a localhost URL, plus missing-audio and unsupported-scheme failures.

T4 adds `tests/test_t4_transcribe.py` (13 tests). It uses a fake ASR backend for stage orchestration, language detection and failure paths, plus the transcript audio-duration validator; the real model is verified separately.

T2b adds `tests/test_t2b_batching.py` (2 tests). It checks the default batch size and the CUDA-OOM halving path with a fake batch function.

The tail-chunk fix adds `tests/test_t4_tail_chunks.py` (3 tests). It checks that a final chunk shorter than 1 s is merged into the previous chunk while exact 30/60 s chunking is unchanged.

T7 adds `tests/test_t7_subtitle.py` (3 tests). It checks three consistent SRTs, display-length splitting, bilingual two-line output and the validation failure path.

T8 adds `tests/test_t8_burn.py` (2 tests). It checks a burned MP4 with video+audio and duration drift ≤0.5 s, plus explicit failure on a corrupt source without output.

T5 adds `tests/test_t5_segment.py` (2 tests). It uses a fixed transcript and LLM exam answer to check sentence boundaries preserve every word/timestamp, and rejects uncovered words.

T6 adds `tests/test_t6_translate.py` (4 tests). It checks two-step exam translation, missing-id failure, cache reuse and batched/concurrent requests.

T2b adds `tests/test_t2b_chunk_bounds.py` (2 tests). It checks short-tail merging in the benchmark chunker and clamping of words slightly past a chunk boundary.

The full suite currently has **70 tests** (T1 29 + T2a 6 + T3 4 + T4 13 + T2b 2 + tail 3 + T7 3 + T8 2 + T5 2 + T6 4 + T2b bounds 2). Note: T2b batching has 2 tests and T2b chunk bounds adds 2; the earlier T2b count is part of the total.

Run all from the repo root with the existing dev venv:

```bash
/root/VideoLingo/.venv/bin/python -m pytest -q \
  tests/test_t1_runner.py tests/test_t2a_rtfx.py \
  tests/test_t3_ingest.py tests/test_t4_transcribe.py \
  tests/test_t2b_batching.py tests/test_t4_tail_chunks.py \
  tests/test_t7_subtitle.py tests/test_t8_burn.py \
  tests/test_t5_segment.py tests/test_t6_translate.py \
  tests/test_t2b_chunk_bounds.py \
  -p no:cacheprovider
```

`pytest` is a dev tool already present in that venv; it is not a runtime dependency. Do not add a broad test suite. Stage verification still follows `docs/agents/verification.md`; stage procedures live in `docs/agents/stages/`.

## RTFx benchmark

The model benchmark is not a pytest test. It requires the ModelScope weights, CUDA, and 16 kHz mono audio:

```bash
/root/VideoLingo/.venv/bin/python -m bench.asr_rtfx preflight --cache-dir /root/.cache/modelscope
```

See `docs/agents/rtfx-baseline.md` for the full 30 s / 60 s / 600 s cold/hot commands and recorded results.

## JSON I/O benchmark

This is not a pytest test. Run it as a one-off script:

```bash
/root/VideoLingo/.venv/bin/python tests/benchmark_json_io.py
```

Last recorded result: 20k words, 2.41 MiB, write 0.162 s, read 0.022 s.