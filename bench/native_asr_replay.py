"""Plan and summarize native GLM-ASR GEMM shape replays.

This module deliberately estimates only isolated linear operations.  It does
not run a model and its summaries cannot be used as ASR acceptance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_MODEL_TYPE = "glmasr"
_RESULT_MODES = frozenset({"fp16_acc32", "fp16_acc16", "int8"})
_CASE_KEYS = frozenset({"id", "m", "n", "k", "repeats"})
_RESULT_CASE_KEYS = _CASE_KEYS | frozenset(
    {
        "times_ms",
        "median_ms",
        "finite",
        "max_abs_error",
        "relative_l2_error",
        "buffer_bytes",
    }
)
_DEVICE_KEYS = frozenset(
    {"name", "compute_capability", "total_memory_bytes"}
)
_RESULT_KEYS = frozenset(
    {"schema_version", "benchmark", "mode", "device", "cases"}
)
_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "planner",
        "model_config_sha256",
        "shape_sha256",
        "scenario_assumptions",
        "cases",
    }
)
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")
_MAX_SCENARIO_INTEGER = 1_000_000
_ENCODER_TOKENS = 1500
_MERGED_AUDIO_TOKENS = 375
_PROJECTOR_INPUT = 5120
_PROJECTOR_HIDDEN = 4096
_EXPECTED_CONFIG = {
    "model_type": _MODEL_TYPE,
    "audio_config.hidden_size": 1280,
    "audio_config.intermediate_size": 5120,
    "audio_config.num_hidden_layers": 32,
    "audio_config.num_attention_heads": 20,
    "audio_config.num_key_value_heads": 20,
    "audio_config.max_position_embeddings": 1500,
    "text_config.hidden_size": 2048,
    "text_config.intermediate_size": 6144,
    "text_config.num_hidden_layers": 28,
    "text_config.num_attention_heads": 16,
    "text_config.num_key_value_heads": 4,
    "text_config.head_dim": 128,
    "text_config.vocab_size": 59264,
}
_OMITTED_OPERATIONS = [
    "mel",
    "convolutions",
    "attention",
    "norms",
    "activations",
    "residuals",
    "KV",
    "routing",
    "tokenization",
    "scheduling",
    "model_loading",
]


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _nested_value(config: Mapping[str, Any], dotted_name: str) -> Any:
    value: Any = config
    traversed: list[str] = []
    for part in dotted_name.split("."):
        traversed.append(part)
        mapping = _require_mapping(value, ".".join(traversed[:-1]) or "config")
        if part not in mapping:
            raise ValueError(f"missing required config value: {dotted_name}")
        value = mapping[part]
    return value


def _validate_model_config(config: Mapping[str, Any]) -> None:
    _require_mapping(config, "config")
    for name, expected in _EXPECTED_CONFIG.items():
        actual = _nested_value(config, name)
        if isinstance(expected, int):
            if isinstance(actual, bool) or not isinstance(actual, int):
                raise ValueError(f"{name} must be the integer {expected}")
        if actual != expected:
            raise ValueError(f"{name} must be {expected!r}, got {actual!r}")

    audio_hidden = _nested_value(config, "audio_config.hidden_size")
    audio_heads = _nested_value(config, "audio_config.num_attention_heads")
    audio_head_dim = _nested_value(config, "audio_config.head_dim") if (
        "head_dim" in _nested_value(config, "audio_config")
    ) else audio_hidden // audio_heads
    if audio_heads * audio_head_dim != audio_hidden:
        raise ValueError("audio_config heads and head dimension do not match hidden_size")

    text_hidden = _nested_value(config, "text_config.hidden_size")
    text_heads = _nested_value(config, "text_config.num_attention_heads")
    text_head_dim = _nested_value(config, "text_config.head_dim")
    if text_heads * text_head_dim != text_hidden:
        raise ValueError("text_config heads and head_dim do not match hidden_size")

    # The official checkpoint does not expose projector dimensions separately.
    # Validate the plan matrices against the two adjoining configured widths.
    if _PROJECTOR_INPUT != 4 * audio_hidden:
        raise ValueError("projector input must equal four merged audio hidden states")
    if _PROJECTOR_HIDDEN != 2 * text_hidden:
        raise ValueError("projector hidden width must equal twice text hidden_size")


def _scenario_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0 or value > _MAX_SCENARIO_INTEGER:
        raise ValueError(
            f"{name} must be between 1 and {_MAX_SCENARIO_INTEGER}"
        )
    return value


def _batch_counts(chunks: int, batch_size: int) -> list[tuple[int, int]]:
    full_batches, remainder = divmod(chunks, batch_size)
    groups: list[tuple[int, int]] = []
    if full_batches:
        groups.append((batch_size, full_batches))
    if remainder:
        groups.append((remainder, 1))
    return groups


def _add_case(
    cases: list[dict[str, int | str]],
    operation: str,
    batch: int,
    m: int,
    n: int,
    k: int,
    repeats: int,
) -> None:
    cases.append(
        {
            "id": f"{operation}_b{batch}",
            "m": m,
            "n": n,
            "k": k,
            "repeats": repeats,
        }
    )


def build_cases(
    config: dict,
    chunks: int = 120,
    encoder_batch: int = 8,
    prefill_batch: int = 32,
    decode_batch: int = 120,
    output_tokens: int = 120,
    prompt_tokens: int = 20,
) -> list[dict]:
    """Build aggregated real-shape GEMM cases for the approved GLM scenario."""

    config = dict(_require_mapping(config, "config"))
    _validate_model_config(config)
    chunks = _scenario_integer("chunks", chunks)
    encoder_batch = _scenario_integer("encoder_batch", encoder_batch)
    prefill_batch = _scenario_integer("prefill_batch", prefill_batch)
    decode_batch = _scenario_integer("decode_batch", decode_batch)
    output_tokens = _scenario_integer("output_tokens", output_tokens)
    prompt_tokens = _scenario_integer("prompt_tokens", prompt_tokens)
    for name, value in (
        ("encoder_batch", encoder_batch),
        ("prefill_batch", prefill_batch),
        ("decode_batch", decode_batch),
    ):
        if value > chunks:
            raise ValueError(f"{name} cannot exceed chunks")

    text_config = _require_mapping(config["text_config"], "text_config")
    max_text_positions = text_config.get("max_position_embeddings")
    if max_text_positions is not None:
        if isinstance(max_text_positions, bool) or not isinstance(
            max_text_positions, int
        ):
            raise ValueError("text_config.max_position_embeddings must be an integer")
        required_positions = _MERGED_AUDIO_TOKENS + prompt_tokens + output_tokens
        if required_positions > max_text_positions:
            raise ValueError(
                "prompt_tokens and output_tokens exceed "
                "text_config.max_position_embeddings"
            )

    audio_hidden = config["audio_config"]["hidden_size"]
    audio_intermediate = config["audio_config"]["intermediate_size"]
    audio_layers = config["audio_config"]["num_hidden_layers"]
    text_hidden = config["text_config"]["hidden_size"]
    text_intermediate = config["text_config"]["intermediate_size"]
    text_layers = config["text_config"]["num_hidden_layers"]
    q_heads = config["text_config"]["num_attention_heads"]
    kv_heads = config["text_config"]["num_key_value_heads"]
    head_dim = config["text_config"]["head_dim"]
    vocab_size = config["text_config"]["vocab_size"]
    decoder_qkv = (q_heads + 2 * kv_heads) * head_dim
    prefill_tokens = _MERGED_AUDIO_TOKENS + prompt_tokens
    generated_steps = output_tokens - 1
    cases: list[dict[str, int | str]] = []

    for batch, count in _batch_counts(chunks, encoder_batch):
        rows = batch * _ENCODER_TOKENS
        layer_repeats = audio_layers * count
        _add_case(cases, "encoder_qkv", batch, rows, 3 * audio_hidden, audio_hidden, layer_repeats)
        _add_case(cases, "encoder_o", batch, rows, audio_hidden, audio_hidden, layer_repeats)
        _add_case(cases, "encoder_mlp_up", batch, rows, audio_intermediate, audio_hidden, layer_repeats)
        _add_case(cases, "encoder_mlp_down", batch, rows, audio_hidden, audio_intermediate, layer_repeats)
        projector_rows = batch * _MERGED_AUDIO_TOKENS
        _add_case(cases, "projector_up", batch, projector_rows, _PROJECTOR_HIDDEN, _PROJECTOR_INPUT, count)
        _add_case(cases, "projector_down", batch, projector_rows, text_hidden, _PROJECTOR_HIDDEN, count)

    for batch, count in _batch_counts(chunks, prefill_batch):
        rows = batch * prefill_tokens
        layer_repeats = text_layers * count
        _add_case(cases, "prefill_qkv", batch, rows, decoder_qkv, text_hidden, layer_repeats)
        _add_case(cases, "prefill_o", batch, rows, text_hidden, text_hidden, layer_repeats)
        _add_case(cases, "prefill_mlp_gate_up", batch, rows, 2 * text_intermediate, text_hidden, layer_repeats)
        _add_case(cases, "prefill_mlp_down", batch, rows, text_hidden, text_intermediate, layer_repeats)
        _add_case(cases, "prefill_lm_head", batch, batch, vocab_size, text_hidden, count)

    if generated_steps:
        for batch, count in _batch_counts(chunks, decode_batch):
            layer_repeats = text_layers * generated_steps * count
            _add_case(cases, "decode_qkv", batch, batch, decoder_qkv, text_hidden, layer_repeats)
            _add_case(cases, "decode_o", batch, batch, text_hidden, text_hidden, layer_repeats)
            _add_case(cases, "decode_mlp_gate_up", batch, batch, 2 * text_intermediate, text_hidden, layer_repeats)
            _add_case(cases, "decode_mlp_down", batch, batch, text_hidden, text_intermediate, layer_repeats)
            _add_case(cases, "decode_lm_head", batch, batch, vocab_size, text_hidden, generated_steps * count)

    _validate_cases(cases)
    return cases


def _validate_cases(cases: Any) -> list[Mapping[str, Any]]:
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise ValueError("cases must be a list")
    validated: list[Mapping[str, Any]] = []
    ids: set[str] = set()
    for index, raw_case in enumerate(cases):
        case = _require_mapping(raw_case, f"cases[{index}]")
        if frozenset(case) != _CASE_KEYS:
            raise ValueError(f"cases[{index}] has an invalid case schema")
        case_id = case["id"]
        if not isinstance(case_id, str) or not _SAFE_ID.fullmatch(case_id):
            raise ValueError(f"cases[{index}].id is invalid")
        if case_id in ids:
            raise ValueError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        for field in ("m", "n", "k", "repeats"):
            value = case[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"cases[{index}].{field} must be a positive integer")
        validated.append(case)
    if not validated:
        raise ValueError("cases cannot be empty")
    return validated


def _manifest_text(cases: list[dict]) -> str:
    validated = _validate_cases(cases)
    lines = ["id\tm\tn\tk\trepeats"]
    lines.extend(
        f"{case['id']}\t{case['m']}\t{case['n']}\t{case['k']}\t{case['repeats']}"
        for case in validated
    )
    return "\n".join(lines) + "\n"


def write_manifest(cases: list[dict], path: str | Path) -> None:
    Path(path).write_text(_manifest_text(cases), encoding="utf-8", newline="")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def summarize(cases: list[dict], result: dict) -> dict:
    """Validate native replay JSON and summarize measured linear work only."""

    expected_cases = _validate_cases(cases)
    result = _require_mapping(result, "result")
    if not _RESULT_KEYS <= frozenset(result):
        raise ValueError("result is missing required top-level fields")
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if result["benchmark"] != "native_gemm_replay":
        raise ValueError("benchmark must be 'native_gemm_replay'")
    if not isinstance(result["mode"], str) or result["mode"] not in _RESULT_MODES:
        raise ValueError("mode is unsupported")

    device = _require_mapping(result["device"], "device")
    if not _DEVICE_KEYS <= frozenset(device):
        raise ValueError("device is missing required fields")
    if not isinstance(device["name"], str) or not device["name"]:
        raise ValueError("device.name must be a nonempty string")
    if not isinstance(device["compute_capability"], str) or not re.fullmatch(
        r"\d+\.\d+", device["compute_capability"]
    ):
        raise ValueError("device.compute_capability must have major.minor form")
    memory = device["total_memory_bytes"]
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise ValueError("device.total_memory_bytes must be a positive integer")

    native_cases = result["cases"]
    if isinstance(native_cases, (str, bytes)) or not isinstance(native_cases, list):
        raise ValueError("result cases must be a list")
    expected_by_id = {case["id"]: case for case in expected_cases}
    native_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_case in enumerate(native_cases):
        native_case = _require_mapping(raw_case, f"result.cases[{index}]")
        if not _RESULT_CASE_KEYS <= frozenset(native_case):
            raise ValueError(f"result case {index} is missing required fields")
        case_id = native_case["id"]
        if not isinstance(case_id, str):
            raise ValueError(f"result case {index} id must be a string")
        if case_id in native_by_id:
            raise ValueError(f"duplicate result case id: {case_id}")
        native_by_id[case_id] = native_case
    if set(native_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(native_by_id))
        extra = sorted(set(native_by_id) - set(expected_by_id))
        raise ValueError(f"result case set mismatch: missing={missing}, extra={extra}")

    summaries: list[dict[str, Any]] = []
    weighted_total = 0.0
    for expected_case in expected_cases:
        case_id = expected_case["id"]
        native_case = native_by_id[case_id]
        for field in ("m", "n", "k", "repeats"):
            if native_case[field] != expected_case[field] or isinstance(
                native_case[field], bool
            ):
                raise ValueError(f"result case {case_id} has mismatched {field}")
        times = native_case["times_ms"]
        if not isinstance(times, list) or not times:
            raise ValueError(f"result case {case_id}.times_ms must be a nonempty list")
        numeric_times = [
            _number(value, f"result case {case_id}.times_ms") for value in times
        ]
        actual_median = float(statistics.median(numeric_times))
        reported_median = _number(
            native_case["median_ms"], f"result case {case_id}.median_ms"
        )
        if reported_median != actual_median:
            raise ValueError(f"result case {case_id}.median_ms is not the median")
        if native_case["finite"] is not True:
            raise ValueError(f"result case {case_id}.finite must be true")
        _number(native_case["max_abs_error"], f"result case {case_id}.max_abs_error")
        _number(
            native_case["relative_l2_error"],
            f"result case {case_id}.relative_l2_error",
        )
        buffer_bytes = native_case["buffer_bytes"]
        if (
            isinstance(buffer_bytes, bool)
            or not isinstance(buffer_bytes, int)
            or buffer_bytes < 0
        ):
            raise ValueError(f"result case {case_id}.buffer_bytes must be nonnegative")
        weighted_ms = reported_median * expected_case["repeats"]
        weighted_total += weighted_ms
        summaries.append(
            {
                "id": case_id,
                "median_ms": reported_median,
                "repeats": expected_case["repeats"],
                "weighted_estimated_linear_ms": weighted_ms,
            }
        )

    return {
        "schema_version": 1,
        "metric_kind": "isolated_gemm_estimate",
        "asr_acceptance": False,
        "mode": result["mode"],
        "device": dict(device),
        "weighted_estimated_linear_ms": weighted_total,
        "omitted_operations": list(_OMITTED_OPERATIONS),
        "cases": summaries,
    }


def _read_json(path: Path, description: str) -> Any:
    if not path.is_file():
        raise ValueError(f"{description} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {description}: {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )


def _validate_metadata(metadata: Any) -> Mapping[str, Any]:
    metadata = _require_mapping(metadata, "metadata")
    if frozenset(metadata) != _METADATA_KEYS:
        raise ValueError("metadata has an invalid schema")
    if (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != 1
    ):
        raise ValueError("metadata.schema_version must be 1")
    if metadata["planner"] != "native_asr_replay":
        raise ValueError("metadata.planner must be 'native_asr_replay'")
    for field in ("model_config_sha256", "shape_sha256"):
        if not isinstance(metadata[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", metadata[field]
        ):
            raise ValueError(f"metadata.{field} must be a SHA256 hex digest")
    cases = _validate_cases(metadata["cases"])
    if metadata["shape_sha256"] != _sha256(_canonical_json(cases)):
        raise ValueError("metadata.shape_sha256 does not match cases")
    assumptions = _require_mapping(
        metadata["scenario_assumptions"], "metadata.scenario_assumptions"
    )
    required_assumptions = {
        "chunks",
        "encoder_batch",
        "prefill_batch",
        "decode_batch",
        "output_tokens",
        "prompt_tokens",
        "decode_active_length",
    }
    if set(assumptions) != required_assumptions:
        raise ValueError("metadata.scenario_assumptions has an invalid schema")
    if assumptions["decode_active_length"] != "constant":
        raise ValueError("metadata decode_active_length must be 'constant'")
    for field in required_assumptions - {"decode_active_length"}:
        _scenario_integer(field, assumptions[field])
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and summarize isolated native GLM-ASR GEMM replays"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--model-config", type=Path, required=True)
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--metadata", type=Path, required=True)
    plan.add_argument("--chunks", type=int, default=120)
    plan.add_argument("--encoder-batch", type=int, default=8)
    plan.add_argument("--prefill-batch", type=int, default=32)
    plan.add_argument("--decode-batch", type=int, default=120)
    plan.add_argument("--output-tokens", type=int, default=120)
    plan.add_argument("--prompt-tokens", type=int, default=20)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--metadata", type=Path, required=True)
    summarize_parser.add_argument("--result", type=Path, required=True)
    summarize_parser.add_argument("--output", type=Path, required=True)
    return parser


def _run_plan(args: argparse.Namespace) -> None:
    if not args.model_config.is_file():
        raise ValueError(f"model config not found: {args.model_config}")
    config_bytes = args.model_config.read_bytes()
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model config JSON: {args.model_config}: {exc}") from exc
    cases = build_cases(
        config,
        chunks=args.chunks,
        encoder_batch=args.encoder_batch,
        prefill_batch=args.prefill_batch,
        decode_batch=args.decode_batch,
        output_tokens=args.output_tokens,
        prompt_tokens=args.prompt_tokens,
    )
    write_manifest(cases, args.manifest)
    metadata = {
        "schema_version": 1,
        "planner": "native_asr_replay",
        "model_config_sha256": _sha256(config_bytes),
        "shape_sha256": _sha256(_canonical_json(cases)),
        "scenario_assumptions": {
            "chunks": args.chunks,
            "encoder_batch": args.encoder_batch,
            "prefill_batch": args.prefill_batch,
            "decode_batch": args.decode_batch,
            "output_tokens": args.output_tokens,
            "prompt_tokens": args.prompt_tokens,
            "decode_active_length": "constant",
        },
        "cases": cases,
    }
    _write_json(args.metadata, metadata)


def _run_summarize(args: argparse.Namespace) -> None:
    metadata = _validate_metadata(_read_json(args.metadata, "metadata"))
    result = _read_json(args.result, "result")
    summary = summarize(list(metadata["cases"]), result)
    summary["model_config_sha256"] = metadata["model_config_sha256"]
    summary["shape_sha256"] = metadata["shape_sha256"]
    summary["scenario_assumptions"] = dict(metadata["scenario_assumptions"])
    _write_json(args.output, summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            _run_plan(args)
        else:
            _run_summarize(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
