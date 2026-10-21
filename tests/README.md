# Tests

T1 has one focused test file: `tests/test_t1_runner.py` (29 tests). It checks the stage runner and the artifact validators without running the full pipeline.

Run it from the repo root with the existing dev venv:

```bash
/root/VideoLingo/.venv/bin/python -m pytest -q tests/test_t1_runner.py -p no:cacheprovider
```

`pytest` is a dev tool already present in that venv; it is not a runtime dependency. Do not add a broad test suite. Stage verification still follows `docs/agents/verification.md`.

## JSON I/O benchmark

This is not a pytest test. Run it as a one-off script:

```bash
/root/VideoLingo/.venv/bin/python tests/benchmark_json_io.py
```

Last recorded result: 20k words, 2.41 MiB, write 0.162 s, read 0.022 s.