from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GLM_MODEL_ID = "ZhipuAI/GLM-ASR-Nano-2512"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
DEFAULT_CACHE = Path(os.environ.get("MODELSCOPE_CACHE", "~/.cache/modelscope")).expanduser()
SAMPLE_RATE = 16000


class PreflightError(RuntimeError):
    pass


class BenchmarkError(RuntimeError):
    pass


def chunk_audio(samples, sample_rate, chunk_seconds):
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    chunk_size = int(round(sample_rate * chunk_seconds))
    return [samples[index : index + chunk_size] for index in range(0, len(samples), chunk_size)]


def summarize_runs(rtfx_values):
    if not rtfx_values:
        raise ValueError("at least one run is required")
    return {
        "count": len(rtfx_values),
        "rtfx_median": statistics.median(rtfx_values),
        "rtfx_min": min(rtfx_values),
        "rtfx_max": max(rtfx_values),
    }


def _check_dependencies():
    required = ("numpy", "torch", "transformers", "soundfile", "modelscope")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise PreflightError(
            "missing Python dependencies: "
            + ", ".join(missing)
            + "; install them into the benchmark interpreter"
        )

    import transformers

    missing_classes = [
        name
        for name in ("AutoProcessor", "GlmAsrForConditionalGeneration", "AutoModelForTokenClassification")
        if not hasattr(transformers, name)
    ]
    if missing_classes:
        raise PreflightError(
            "installed transformers lacks: "
            + ", ".join(missing_classes)
            + f" (found {transformers.__version__})"
        )


def _resolve_models(cache_dir):
    from modelscope import snapshot_download

    resolved = {}
    for model_id in (GLM_MODEL_ID, ALIGNER_MODEL_ID):
        try:
            path = Path(
                snapshot_download(
                    model_id,
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                )
            )
        except Exception as exc:
            raise PreflightError(
                f"model weights for {model_id} are not available in {cache_dir}: {exc}"
            ) from exc

        weights = path / "model.safetensors"
        if not weights.is_file():
            raise PreflightError(
                f"model weights for {model_id} are incomplete in {path}: model.safetensors is missing"
            )
        resolved[model_id] = {
            "path": str(path),
            "safetensors_bytes": weights.stat().st_size,
        }
    return resolved


def _audio_info(path):
    import soundfile

    audio_path = Path(path)
    if not audio_path.is_file():
        raise PreflightError(f"audio file not found: {audio_path}")

    try:
        info = soundfile.info(str(audio_path))
    except Exception as exc:
        raise PreflightError(f"cannot read audio file {audio_path}: {exc}") from exc

    if info.samplerate != SAMPLE_RATE:
        raise PreflightError(
            f"audio file {audio_path} has sample rate {info.samplerate}; expected {SAMPLE_RATE}"
        )
    if info.channels != 1:
        raise PreflightError(
            f"audio file {audio_path} has {info.channels} channels; expected mono"
        )
    if info.frames <= 0:
        raise PreflightError(f"audio file {audio_path} is empty")
    return {
        "path": str(audio_path),
        "duration_s": info.frames / info.samplerate,
        "sample_rate": info.samplerate,
        "channels": info.channels,
    }


def preflight(cache_dir=DEFAULT_CACHE, audio_paths=()):
    _check_dependencies()

    import torch
    import transformers
    import modelscope

    if not torch.cuda.is_available():
        raise PreflightError("CUDA is not available; the RTFx baseline requires a GPU")

    properties = torch.cuda.get_device_properties(0)
    models = _resolve_models(Path(cache_dir))
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "modelscope": modelscope.__version__,
        "cuda": torch.version.cuda,
        "gpu": {
            "name": properties.name,
            "total_memory_mb": round(properties.total_memory / 2**20, 1),
        },
        "models": models,
        "audio": [_audio_info(path) for path in audio_paths],
    }


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _synchronize():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class Models:
    glm_processor: object
    glm_model: object
    aligner_processor: object
    aligner_model: object


def load_models(model_paths):
    import torch
    from transformers import AutoModelForTokenClassification, AutoProcessor, GlmAsrForConditionalGeneration

    started = time.perf_counter()
    glm_path = model_paths[GLM_MODEL_ID]["path"]
    aligner_path = model_paths[ALIGNER_MODEL_ID]["path"]

    try:
        glm_processor = AutoProcessor.from_pretrained(glm_path)
        glm_model = GlmAsrForConditionalGeneration.from_pretrained(
            glm_path,
            dtype="auto",
            device_map="cuda",
        )
        glm_model.eval()

        aligner_processor = AutoProcessor.from_pretrained(aligner_path)
        aligner_model = AutoModelForTokenClassification.from_pretrained(
            aligner_path,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        aligner_model.eval()
    except Exception as exc:
        raise BenchmarkError(f"failed to load models: {exc}") from exc
    _synchronize()

    models = Models(glm_processor, glm_model, aligner_processor, aligner_model)
    return models, time.perf_counter() - started


def _transcribe_batch(models, audio_arrays, max_new_tokens):
    import torch

    inputs = models.glm_processor.apply_transcription_request(list(audio_arrays))
    inputs = inputs.to(models.glm_model.device, dtype=models.glm_model.dtype)
    with torch.inference_mode():
        output_ids = models.glm_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    transcripts = models.glm_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        strip_prefix=True,
    )
    return [text.strip() for text in transcripts]


def _align_batch(models, audio_arrays, transcripts, language):
    import torch

    aligner_inputs, word_lists = models.aligner_processor.prepare_forced_aligner_inputs(
        audio=list(audio_arrays),
        transcript=transcripts,
        language=language,
    )
    aligner_inputs = aligner_inputs.to(
        models.aligner_model.device,
        dtype=models.aligner_model.dtype,
    )
    with torch.inference_mode():
        outputs = models.aligner_model(**aligner_inputs)
    return models.aligner_processor.decode_forced_alignment(
        logits=outputs.logits,
        input_ids=aligner_inputs["input_ids"],
        word_lists=word_lists,
        timestamp_token_id=models.aligner_model.config.timestamp_token_id,
    )


def _validate_timestamps(timestamps, chunk_duration):
    if not timestamps:
        raise BenchmarkError("forced aligner returned no words")

    previous_start = -1.0
    for item in timestamps:
        start = float(item["start_time"])
        end = float(item["end_time"])
        if start < previous_start - 1e-3:
            raise BenchmarkError("forced aligner timestamps are not monotonic")
        if start < -1e-3 or end > chunk_duration + 0.5:
            raise BenchmarkError(
                f"forced aligner timestamp {start:.3f}-{end:.3f}s outside chunk 0-{chunk_duration:.3f}s"
            )
        if end < start:
            raise BenchmarkError("forced aligner returned end before start")
        previous_start = start


def _run_pass(models, samples, sample_rate, chunk_seconds, batch_size, language, max_new_tokens, chunk_limit=None):
    chunks = chunk_audio(samples, sample_rate, chunk_seconds)
    if chunk_limit is not None:
        chunks = chunks[:chunk_limit]

    words = 0
    asr_s = 0.0
    align_s = 0.0
    started = time.perf_counter()

    for index in range(0, len(chunks), batch_size):
        batch = chunks[index : index + batch_size]

        _synchronize()
        batch_started = time.perf_counter()
        transcripts = _transcribe_batch(models, batch, max_new_tokens)
        _synchronize()
        asr_s += time.perf_counter() - batch_started

        if any(not transcript for transcript in transcripts):
            raise BenchmarkError("GLM-ASR returned an empty transcript")

        _synchronize()
        batch_started = time.perf_counter()
        aligned = _align_batch(models, batch, transcripts, language)
        _synchronize()
        align_s += time.perf_counter() - batch_started

        for timestamps, chunk in zip(aligned, batch):
            _validate_timestamps(timestamps, len(chunk) / sample_rate)
            words += len(timestamps)

    _synchronize()
    return {
        "chunks": len(chunks),
        "words": words,
        "asr_s": asr_s,
        "align_s": align_s,
        "inference_s": time.perf_counter() - started,
    }


def _vram_snapshot():
    import torch

    return {
        "vram_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "vram_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }


def _measure_cold(paths, audio, config):
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    models, load_s = load_models(paths)
    result = _run_pass(
        models,
        audio["samples"],
        audio["info"]["sample_rate"],
        config["chunk_seconds"],
        config["batch_size"],
        config["language"],
        config["max_new_tokens"],
    )
    wall_s = time.perf_counter() - started
    duration = audio["info"]["duration_s"]
    return {
        **result,
        "load_s": load_s,
        "wall_s": wall_s,
        "rtfx": duration / wall_s,
        "compute_rtfx": duration / result["inference_s"],
        **_vram_snapshot(),
    }


def _measure_hot(paths, audio, config):
    import torch

    models, load_s = load_models(paths)
    if config["warmups"] > 0:
        _run_pass(
            models,
            audio["samples"],
            audio["info"]["sample_rate"],
            config["chunk_seconds"],
            config["batch_size"],
            config["language"],
            config["max_new_tokens"],
            chunk_limit=config["batch_size"],
        )

    runs = []
    duration = audio["info"]["duration_s"]
    for run_index in range(config["runs"]):
        torch.cuda.reset_peak_memory_stats()
        _synchronize()
        started = time.perf_counter()
        result = _run_pass(
            models,
            audio["samples"],
            audio["info"]["sample_rate"],
            config["chunk_seconds"],
            config["batch_size"],
            config["language"],
            config["max_new_tokens"],
        )
        wall_s = time.perf_counter() - started
        runs.append(
            {
                "run": run_index + 1,
                **result,
                "wall_s": wall_s,
                "rtfx": duration / wall_s,
                "compute_rtfx": duration / result["inference_s"],
                **_vram_snapshot(),
            }
        )
    return load_s, runs


def _load_samples(path):
    import soundfile

    samples, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    return samples, sample_rate


def _cmd_preflight(args):
    report = preflight(args.cache_dir, args.audio)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_run(args):
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must not be negative")

    audio_info = _audio_info(args.audio)
    environment = preflight(args.cache_dir, (args.audio,))
    samples, sample_rate = _load_samples(args.audio)
    audio = {"samples": samples, "info": audio_info}
    config = {
        "chunk_seconds": args.chunk_seconds,
        "batch_size": args.batch_size,
        "language": args.language,
        "max_new_tokens": args.max_new_tokens,
        "warmups": args.warmups,
        "runs": args.runs,
    }

    if args.mode == "cold":
        runs = [_measure_cold(environment["models"], audio, config)]
        load_s = runs[0]["load_s"]
    else:
        load_s, runs = _measure_hot(environment["models"], audio, config)

    report = {
        "schema_version": 1,
        "mode": args.mode,
        "audio": audio_info,
        "config": config,
        "environment": {
            "python": environment["python"],
            "torch": environment["torch"],
            "transformers": environment["transformers"],
            "modelscope": environment["modelscope"],
            "cuda": environment["cuda"],
            "gpu": environment["gpu"],
        },
        "models": environment["models"],
        "load_s": load_s,
        "runs": runs,
        "summary": {
            **summarize_runs([run["rtfx"] for run in runs]),
            "vram_peak_allocated_mb": max(run["vram_peak_allocated_mb"] for run in runs),
            "vram_peak_reserved_mb": max(run["vram_peak_reserved_mb"] for run in runs),
            "words": runs[-1]["words"],
        },
    }

    if args.output:
        _write_json(args.output, report)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "audio_s": audio_info["duration_s"],
                "batch_size": config["batch_size"],
                "summary": report["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_download(args):
    _check_dependencies()
    for model_id in (GLM_MODEL_ID, ALIGNER_MODEL_ID):
        from modelscope import snapshot_download

        path = snapshot_download(model_id, cache_dir=str(Path(args.cache_dir).expanduser()))
        print(f"{model_id}: {path}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m bench.asr_rtfx",
        description="GLM-ASR + Qwen3-ForcedAligner RTFx benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight", help="check dependencies, weights and audio")
    preflight_parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    preflight_parser.add_argument("--audio", action="append", default=[])
    preflight_parser.set_defaults(func=_cmd_preflight)

    run_parser = subparsers.add_parser("run", help="measure one cold or hot run")
    run_parser.add_argument("--audio", required=True)
    run_parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    run_parser.add_argument("--mode", choices=("cold", "hot"), required=True)
    run_parser.add_argument("--chunk-seconds", type=float, default=30.0)
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument("--language", default="Chinese")
    run_parser.add_argument("--max-new-tokens", type=int, default=512)
    run_parser.add_argument("--warmups", type=int, default=1)
    run_parser.add_argument("--runs", type=int, default=3)
    run_parser.add_argument("--output")
    run_parser.set_defaults(func=_cmd_run)

    download_parser = subparsers.add_parser("download", help="download weights from ModelScope")
    download_parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    download_parser.set_defaults(func=_cmd_download)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PreflightError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2
    except (BenchmarkError, ValueError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())