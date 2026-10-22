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
