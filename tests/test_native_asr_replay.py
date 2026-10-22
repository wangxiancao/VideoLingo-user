from __future__ import annotations

import json
import hashlib
import math
import subprocess
import sys
from pathlib import Path

import pytest

from bench.native_asr_replay import build_cases, summarize, write_manifest


ROOT = Path(__file__).resolve().parents[1]


def glm_config():
    return {
        "model_type": "glmasr",
        "audio_config": {
            "hidden_size": 1280,
            "intermediate_size": 5120,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "max_position_embeddings": 1500,
        },
        "text_config": {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 59264,
        },
    }


def valid_native_result(cases):
    return {
        "schema_version": 1,
        "benchmark": "native_gemm_replay",
        "mode": "fp16_acc32",
        "device": {
            "name": "test",
            "compute_capability": "8.6",
            "total_memory_bytes": 24 * 1024**3,
        },
        "cases": [
            dict(
                case,
                times_ms=[1.0, 2.0, 3.0],
                median_ms=2.0,
                finite=True,
                max_abs_error=0.0,
                relative_l2_error=0.0,
                buffer_bytes=1024,
            )
            for case in cases
        ],
    }


def _case(cases, case_id):
    return next(case for case in cases if case["id"] == case_id)


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "bench.native_asr_replay", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_glm_planner_uses_gqa_and_all_generation_work():
    cases = build_cases(glm_config())

    assert _case(cases, "prefill_qkv_b32") == {
        "id": "prefill_qkv_b32",
        "m": 12640,
        "n": 3072,
        "k": 2048,
        "repeats": 84,
    }
    assert _case(cases, "prefill_lm_head_b32")["m"] == 32
    assert _case(cases, "decode_qkv_b120")["repeats"] == 28 * 119
    assert _case(cases, "decode_lm_head_b120")["repeats"] == 119
    assert _case(cases, "projector_up_b8") == {
        "id": "projector_up_b8",
        "m": 3000,
        "n": 4096,
        "k": 5120,
        "repeats": 15,
    }
    assert _case(cases, "projector_down_b8")["k"] == 4096
    assert _case(cases, "projector_down_b8")["n"] == 2048


def test_partial_batches_have_exact_shapes_and_aggregated_repeats():
    cases = build_cases(
        glm_config(),
        chunks=10,
        encoder_batch=4,
        prefill_batch=6,
        decode_batch=6,
        output_tokens=2,
    )

    assert _case(cases, "encoder_qkv_b4") == {
        "id": "encoder_qkv_b4",
        "m": 6000,
        "n": 3840,
        "k": 1280,
        "repeats": 64,
    }
    assert _case(cases, "encoder_qkv_b2")["repeats"] == 32
    assert _case(cases, "projector_up_b2")["m"] == 750
    assert _case(cases, "prefill_qkv_b4")["m"] == 4 * 395
    assert _case(cases, "decode_qkv_b4")["repeats"] == 28


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("chunks", 0),
        ("chunks", True),
        ("encoder_batch", 121),
        ("prefill_batch", False),
        ("decode_batch", 121),
        ("output_tokens", 0),
        ("prompt_tokens", 2**31),
    ],
)
def test_planner_rejects_invalid_scenario_parameters(name, value):
    kwargs = {name: value}

    with pytest.raises(ValueError, match=name):
        build_cases(glm_config(), **kwargs)


def test_planner_rejects_an_unrelated_checkpoint_shape():
    config = glm_config()
    config["text_config"]["num_key_value_heads"] = 16

    with pytest.raises(ValueError, match="text_config.num_key_value_heads"):
        build_cases(config)


def test_manifest_is_stable_native_tsv(tmp_path):
    cases = build_cases(
        glm_config(), chunks=1, encoder_batch=1, prefill_batch=1, decode_batch=1
    )
    path = tmp_path / "shapes.tsv"

    write_manifest(cases[:2], path)

    assert path.read_text(encoding="utf-8") == (
        "id\tm\tn\tk\trepeats\n"
        "encoder_qkv_b1\t1500\t3840\t1280\t32\n"
        "encoder_o_b1\t1500\t1280\t1280\t32\n"
    )


def test_summary_is_an_honest_isolated_gemm_estimate():
    cases = build_cases(
        glm_config(), chunks=1, encoder_batch=1, prefill_batch=1, decode_batch=1
    )
    result = valid_native_result(cases)
    result["cases"][0]["times_ms"] = [4.0, 2.0]
    result["cases"][0]["median_ms"] = 3.0

    summary = summarize(cases, result)

    assert summary["asr_acceptance"] is False
    assert summary["metric_kind"] == "isolated_gemm_estimate"
    assert "attention" in summary["omitted_operations"]
    assert "model_loading" in summary["omitted_operations"]
    assert summary["weighted_estimated_linear_ms"] == sum(
        native_case["median_ms"] * native_case["repeats"]
        for native_case in result["cases"]
    )
    assert not any("rtfx" in key.lower() for key in summary)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_summary_rejects_non_exact_case_sets(mutation):
    cases = build_cases(
        glm_config(), chunks=1, encoder_batch=1, prefill_batch=1, decode_batch=1
    )
    result = valid_native_result(cases)
    if mutation == "missing":
        result["cases"].pop()
    elif mutation == "extra":
        result["cases"].append(
            dict(result["cases"][0], id="unexpected", m=1, n=1, k=1, repeats=1)
        )
    else:
        result["cases"].append(dict(result["cases"][0]))

    with pytest.raises(ValueError, match="case"):
        summarize(cases, result)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("m", 999),
        ("repeats", 999),
        ("times_ms", [1.0, math.inf]),
        ("times_ms", [1.0, -0.1]),
        ("median_ms", 99.0),
        ("finite", False),
        ("finite", 1),
        ("max_abs_error", -0.1),
        ("relative_l2_error", math.nan),
    ],
)
def test_summary_rejects_invalid_native_measurements(field, value):
    cases = build_cases(
        glm_config(), chunks=1, encoder_batch=1, prefill_batch=1, decode_batch=1
    )
    result = valid_native_result(cases)
    result["cases"][0][field] = value

    with pytest.raises(ValueError):
        summarize(cases, result)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2), ("benchmark", "other"), ("mode", "bf16")],
)
def test_summary_rejects_unsupported_native_contract(field, value):
    cases = build_cases(
        glm_config(), chunks=1, encoder_batch=1, prefill_batch=1, decode_batch=1
    )
    result = valid_native_result(cases)
    result[field] = value

    with pytest.raises(ValueError, match=field):
        summarize(cases, result)


def test_summary_accepts_additional_native_fields():
    cases = build_cases(
        glm_config(), chunks=1, encoder_batch=1, prefill_batch=1, decode_batch=1
    )
    result = valid_native_result(cases)
    result["native_version"] = "future"
    result["device"]["ordinal"] = 0
    result["cases"][0]["sample_count"] = 3

    summary = summarize(cases, result)

    assert summary["metric_kind"] == "isolated_gemm_estimate"


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("top", "schema_version"),
        ("device", "total_memory_bytes"),
        ("case", "buffer_bytes"),
    ],
)
def test_summary_rejects_boolean_numeric_fields(location, field):
    cases = build_cases(
        glm_config(), chunks=1, encoder_batch=1, prefill_batch=1, decode_batch=1
    )
    result = valid_native_result(cases)
    target = {
        "top": result,
        "device": result["device"],
        "case": result["cases"][0],
    }[location]
    target[field] = True

    with pytest.raises(ValueError):
        summarize(cases, result)


def test_cli_missing_input_paths_exit_2(tmp_path):
    plan = _cli(
        "plan",
        "--model-config",
        str(tmp_path / "missing-config.json"),
        "--manifest",
        str(tmp_path / "manifest.tsv"),
        "--metadata",
        str(tmp_path / "metadata.json"),
    )
    summary = _cli(
        "summarize",
        "--metadata",
        str(tmp_path / "missing-metadata.json"),
        "--result",
        str(tmp_path / "missing-result.json"),
        "--output",
        str(tmp_path / "summary.json"),
    )

    assert plan.returncode == 2
    assert "not found" in plan.stderr.lower()
    assert summary.returncode == 2
    assert "not found" in summary.stderr.lower()


def test_plan_cli_records_assumptions_without_claiming_observations(tmp_path):
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.tsv"
    metadata_path = tmp_path / "metadata.json"
    config_path.write_text(json.dumps(glm_config()), encoding="utf-8")

    result = _cli(
        "plan",
        "--model-config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--metadata",
        str(metadata_path),
        "--chunks",
        "1",
        "--encoder-batch",
        "1",
        "--prefill-batch",
        "1",
        "--decode-batch",
        "1",
    )

    assert result.returncode == 0, result.stderr
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["scenario_assumptions"]["output_tokens"] == 120
    assert metadata["scenario_assumptions"]["prompt_tokens"] == 20
    assert metadata["scenario_assumptions"]["decode_active_length"] == "constant"
    assert metadata["cases"] == build_cases(
        glm_config(), chunks=1, encoder_batch=1, prefill_batch=1, decode_batch=1
    )
    assert metadata["model_config_sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    canonical_cases = json.dumps(
        metadata["cases"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert metadata["shape_sha256"] == hashlib.sha256(canonical_cases).hexdigest()


def test_summarize_cli_validates_metadata_hash_and_writes_stable_summary(tmp_path):
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.tsv"
    metadata_path = tmp_path / "metadata.json"
    result_path = tmp_path / "result.json"
    output_path = tmp_path / "summary.json"
    config_path.write_text(json.dumps(glm_config()), encoding="utf-8")
    planned = _cli(
        "plan",
        "--model-config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--metadata",
        str(metadata_path),
        "--chunks",
        "1",
        "--encoder-batch",
        "1",
        "--prefill-batch",
        "1",
        "--decode-batch",
        "1",
        "--output-tokens",
        "2",
    )
    assert planned.returncode == 0, planned.stderr
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    result_path.write_text(
        json.dumps(valid_native_result(metadata["cases"])), encoding="utf-8"
    )

    completed = _cli(
        "summarize",
        "--metadata",
        str(metadata_path),
        "--result",
        str(result_path),
        "--output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["shape_sha256"] == metadata["shape_sha256"]
    assert summary["metric_kind"] == "isolated_gemm_estimate"
    assert summary["asr_acceptance"] is False
    assert "rtfx" not in output_path.read_text(encoding="utf-8").lower()

    metadata["shape_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    rejected = _cli(
        "summarize",
        "--metadata",
        str(metadata_path),
        "--result",
        str(result_path),
        "--output",
        str(output_path),
    )
    assert rejected.returncode == 2
    assert "shape_sha256" in rejected.stderr
