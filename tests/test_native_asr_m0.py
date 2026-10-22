"""Contract tests for the M0 native weight loader (`glm_asr_m0`).

The reference conversions are computed independently with numpy (raw
safetensors bytes -> float32 -> float16), so the C++ CPU and GPU paths are
checked against a third implementation rather than against each other only.

These tests need the locally cached official checkpoint and the built binary;
they are skipped (not failed) when either is absent.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BINARY = REPO / "native" / "asr" / "build" / "glm_asr_m0"
MODEL_DIR = Path("/root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512")

pytestmark = pytest.mark.skipif(
    not MODEL_DIR.is_dir() or not BINARY.exists(),
    reason="official GLM-ASR-Nano-2512 checkpoint or built glm_asr_m0 missing",
)


def run_loader(*args):
    return subprocess.run(
        [str(BINARY), *args], capture_output=True, text=True, timeout=1800
    )


def reference_hashes():
    """Independent numpy BF16->FP16 conversion and SHA-256 of every tensor."""
    import numpy as np

    path = MODEL_DIR / "model.safetensors"
    with open(path, "rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length))
    with open(path, "rb") as handle:
        handle.seek(8 + header_length)
        data = handle.read()

    hashes = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        begin, end = spec["data_offsets"]
        bits = np.frombuffer(data[begin:end], dtype="<u2").astype(np.uint32)
        values = (bits << 16).view(np.float32)
        fp16 = values.astype(np.float16)
        hashes[name] = hashlib.sha256(fp16.tobytes()).hexdigest()
    return hashes


def sample_names(reference, count=25):
    names = sorted(reference)
    step = max(1, len(names) // count)
    return names[::step]


def test_dump_plan_inventory_matches_reference():
    result = run_loader("--model-dir", str(MODEL_DIR), "--dump-plan")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    reference = reference_hashes()
    assert report["tensor_count"] == len(reference) == 745
    assert report["missing_tensors"] == []
    assert report["unexpected_tensors"] == []
    assert report["shape_mismatches"] == []
    config = (MODEL_DIR / "config.json").read_bytes()
    assert report["config_sha256"] == hashlib.sha256(config).hexdigest()
    assert report["fp16_bytes"] == report["total_params"] * 2


def test_cpu_conversion_hashes_match_numpy_reference():
    result = run_loader("--model-dir", str(MODEL_DIR), "--dump-plan", "--hash")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    reference = reference_hashes()
    produced = {t["name"]: t["sha256_fp16"] for t in report["tensors"]}
    assert set(produced) == set(reference)
    for name in sample_names(reference):
        assert produced[name] == reference[name], name
    assert report["bf16_to_fp16"]["overflow_count"] == 0
    assert report["bf16_to_fp16"]["nan_count"] == 0
    assert report["bf16_to_fp16"]["max_abs_value"] > 0


def test_device_load_matches_numpy_reference():
    result = run_loader("--model-dir", str(MODEL_DIR), "--load")
    if result.returncode == 1 and "driver" in result.stderr.lower():
        pytest.skip("no usable CUDA device")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    reference = reference_hashes()
    produced = {t["name"]: t["sha256_fp16"] for t in report["tensors"]}
    assert set(produced) == set(reference)
    for name in sample_names(reference):
        assert produced[name] == reference[name], name
    assert report["vram"]["budget_bytes"] == 16384 * 1024 * 1024
    assert report["vram"]["weights_device_bytes"] == report["fp16_bytes"]
    # weights + largest staging tensor only, far below the 16 GiB budget
    assert report["vram"]["peak_device_bytes"] < 8 * 1024**3
    assert report["bf16_to_fp16"]["overflow_count"] == 0


def test_vram_budget_is_enforced():
    result = run_loader(
        "--model-dir", str(MODEL_DIR), "--load", "--max-vram-mib", "100"
    )
    assert result.returncode == 1
    assert "exceeds VRAM budget" in result.stderr
    assert result.stdout == ""


def test_invalid_arguments_rejected():
    cases = [
        [],
        ["--model-dir", str(MODEL_DIR), "--hash"],  # --hash without --dump-plan
        ["--model-dir", str(MODEL_DIR), "--dump-plan", "--load"],
        ["--model-dir", str(MODEL_DIR), "--load", "--max-vram-mib", "0"],
        ["--model-dir", str(MODEL_DIR), "--load", "--max-vram-mib", "abc"],
        ["--model-dir", "/nonexistent-model-dir", "--dump-plan"],
        ["--model-dir", str(MODEL_DIR), "--wat"],
    ]
    for args in cases:
        result = run_loader(*args)
        assert result.returncode == 2, (args, result.returncode, result.stderr)
        assert result.stdout == ""
        assert result.stderr.strip() != ""


# ---------------------------------------------------------------------------
# M0 Task 2: mel frontend vs transformers WhisperFeatureExtractor
# ---------------------------------------------------------------------------


def deterministic_pcm(sample_count, seed=42):
    import numpy as np

    rng = np.random.default_rng(seed)
    t = np.arange(sample_count, dtype=np.float64)
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t / 16000)
        + 0.2 * np.sin(2 * np.pi * 1234 * t / 16000)
        + 0.1 * np.sin(2 * np.pi * 3000 * t / 16000)
        + 0.05 * rng.standard_normal(sample_count)
    )
    return signal.astype(np.float32)


def reference_mel(pcm32):
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        chunk_length=30,
        dither=0.0,
    )
    return extractor(
        pcm32, sampling_rate=16000, truncation=False, padding="max_length",
        return_tensors="np",
    ).input_features[0]


def assert_mel_matches_reference(pcm32, tmp_path, max_abs=1e-3):
    import numpy as np

    pcm_path = tmp_path / "pcm.f32"
    out_path = tmp_path / "mel.f32"
    pcm32.tofile(pcm_path)
    result = run_loader("mel", "--pcm", str(pcm_path), "--out", str(out_path))
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["measurement_kind"] == "correctness_only"
    assert summary["mel_bins"] == 128 and summary["frames"] == 3000
    assert summary["nonfinite"] == 0
    produced = np.fromfile(out_path, dtype=np.float32).reshape(128, 3000)
    reference = reference_mel(pcm32)
    diff = np.abs(produced.astype(np.float64) - reference.astype(np.float64))
    assert diff.max() <= max_abs, diff.max()
    assert diff.mean() <= 1e-5, diff.mean()


def test_mel_full_chunk_matches_transformers_reference(tmp_path):
    assert_mel_matches_reference(deterministic_pcm(480000, seed=42), tmp_path)


def test_mel_short_pcm_zero_padding_matches_reference(tmp_path):
    assert_mel_matches_reference(deterministic_pcm(30000, seed=7), tmp_path)


def test_mel_rejects_invalid_inputs(tmp_path):
    pcm_path = tmp_path / "pcm.f32"
    deterministic_pcm(1000, seed=1).tofile(pcm_path)
    cases = [
        ["mel"],  # missing --pcm/--out
        ["mel", "--pcm", str(tmp_path / "missing.f32"), "--out", str(tmp_path / "o.f32")],
        ["mel", "--pcm", str(pcm_path), "--out", str(tmp_path / "o.f32"), "--samples", "0"],
        ["mel", "--pcm", str(pcm_path), "--out", str(tmp_path / "o.f32"), "--wat"],
    ]
    for args in cases:
        result = run_loader(*args)
        assert result.returncode == 2, (args, result.returncode, result.stderr)
    import numpy as np

    oversize = tmp_path / "oversize.f32"
    np.zeros(480001, dtype=np.float32).tofile(oversize)
    result = run_loader("mel", "--pcm", str(oversize), "--out", str(tmp_path / "o.f32"))
    assert result.returncode == 2
