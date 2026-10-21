# Tests

T1 has one focused test file: `tests/test_t1_runner.py` (29 tests). It checks the stage runner and the artifact validators without running the full pipeline.

T2a adds `tests/test_t2a_rtfx.py` (6 tests). It checks audio chunking, RTFx summary math, and the preflight/run error paths without loading models.

T3 adds `tests/test_t3_ingest.py` (4 tests). It runs ingest through the stage runner for a local file and a localhost URL, plus missing-audio and unsupported-scheme failures.

T4 adds `tests/test_t4_transcribe.py` (13 tests). It uses a fake ASR backend for stage orchestration, language detection and failure paths, plus the transcript audio-duration validator; the real model is verified separately.

T2b adds `tests/test_t2b_batching.py` (2 tests). It checks the default batch size and the CUDA-OOM halving path with a fake batch function.

Run all from the repo root with the existing dev venv:

```bash
/root/VideoLingo/.venv/bin/python -m pytest -q \
  tests/test_t1_runner.py tests/test_t2a_rtfx.py \
  tests/test_t3_ingest.py tests/test_t4_transcribe.py \
  tests/test_t2b_batching.py \
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