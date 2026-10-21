import subprocess
import sys
from pathlib import Path

import pytest

from bench.asr_rtfx import chunk_audio, summarize_runs

ROOT = Path(__file__).resolve().parents[1]


def _cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "bench.asr_rtfx", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_chunk_audio_covers_full_clip():
    chunks = chunk_audio([0] * 16000 * 35, sample_rate=16000, chunk_seconds=30)

    assert [len(chunk) for chunk in chunks] == [16000 * 30, 16000 * 5]


def test_chunk_audio_ten_minutes_uses_twenty_chunks():
    chunks = chunk_audio([0] * 16000 * 600, sample_rate=16000, chunk_seconds=30)

    assert len(chunks) == 20
    assert sum(len(chunk) for chunk in chunks) == 16000 * 600


def test_chunk_audio_rejects_bad_chunk_seconds():
    with pytest.raises(ValueError):
        chunk_audio([0] * 100, sample_rate=16000, chunk_seconds=0)


def test_summarize_runs_reports_median_and_bounds():
    summary = summarize_runs([1.0, 2.0, 3.0])

    assert summary["rtfx_median"] == 2.0
    assert summary["rtfx_min"] == 1.0
    assert summary["rtfx_max"] == 3.0


def test_preflight_without_weights_exits_2(tmp_path):
    empty_cache = tmp_path / "cache"
    empty_cache.mkdir()

    result = _cli("preflight", "--cache-dir", str(empty_cache))

    assert result.returncode == 2
    assert "ZhipuAI/GLM-ASR-Nano-2512" in result.stderr
    assert "not available" in result.stderr


def test_run_without_audio_exits_2():
    result = _cli("run", "--audio", "does-not-exist.wav", "--mode", "cold")

    assert result.returncode == 2
    assert "audio" in result.stderr.lower()