# Tests

T1 has one focused test file: `tests/test_t1_runner.py` (29 tests). It checks the stage runner and the artifact validators without running the full pipeline.

T2a adds `tests/test_t2a_rtfx.py` (6 tests). It checks audio chunking, RTFx summary math, and the preflight/run error paths without loading models.

Run both from the repo root with the existing dev venv:

```bash
/root/VideoLingo/.venv/bin/python -m pytest -q tests/test_t1_runner.py tests/test_t2a_rtfx.py -p no:cacheprovider
```

`pytest` is a dev tool already present in that venv; it is not a runtime dependency. Do not add a broad test suite. Stage verification still follows `docs/agents/verification.md`; the T2a benchmark procedure is in `docs/agents/rtfx-baseline.md`.

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