# Tests

T1 has one focused test file: `tests/test_t1_runner.py`. It checks the stage runner and the artifact validators without running the full pipeline.

Run it from the repo root with the existing dev venv:

```bash
/root/VideoLingo/.venv/bin/python -m pytest -q tests/test_t1_runner.py -p no:cacheprovider
```

`pytest` is a dev tool already present in that venv; it is not a runtime dependency. Do not add a broad test suite. Stage verification still follows `docs/agents/verification.md`.