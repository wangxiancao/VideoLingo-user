import argparse
import sys
from pathlib import Path
from typing import Iterable

from pipeline.codec import DEFAULT_CODEC
from pipeline.config import ReadOnlyConfig
from pipeline.contracts import (
    CodecError,
    ConfigError,
    ContractError,
    MissingInputError,
    PipelineError,
    StageContext,
    StageError,
    StageSpec,
    UsageError,
)
from pipeline.registry import ARTIFACTS, STAGES, producer_of, validate_registry
from pipeline.validators import validate_artifact

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def _normalize_stage_ids(stage_ids: str | Iterable[str]) -> list[str]:
    if isinstance(stage_ids, str):
        raw = [part.strip() for part in stage_ids.split(",")]
    else:
        raw = [str(part).strip() for part in stage_ids]
    raw = [part for part in raw if part]
    if not raw:
        raise UsageError("--stages must name at least one stage")
    if len(set(raw)) != len(raw):
        raise UsageError("stage ids must be unique")
    return raw


def _select_stages(
    stage_ids: list[str],
    stages: tuple[StageSpec, ...],
) -> tuple[StageSpec, ...]:
    try:
        validate_registry(stages)
    except ValueError as exc:
        raise UsageError(f"invalid stage table: {exc}") from exc
    by_id = {stage.id: stage for stage in stages}
    unknown = [stage_id for stage_id in stage_ids if stage_id not in by_id]
    if unknown:
        raise UsageError(f"unknown stage id(s): {', '.join(unknown)}")
    selected = set(stage_ids)
    return tuple(stage for stage in stages if stage.id in selected)


def _check_source(selected: tuple[StageSpec, ...], source: str | None) -> None:
    needs_source = any(stage.external_input == "source" for stage in selected)
    if needs_source and not source:
        raise MissingInputError("--source is required when ingest is selected")
    if source and not needs_source:
        raise UsageError("--source is only valid with ingest")
    if source and "://" not in source and not Path(source).exists():
        raise MissingInputError(f"source not found: {source}")


def _check_llm_fixture(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    fixture = Path(path)
    if not fixture.is_file():
        raise MissingInputError(f"llm fixture not found: {fixture}")
    return fixture


def _check_requires(
    stage: StageSpec,
    ctx: StageContext,
    stages: tuple[StageSpec, ...],
) -> None:
    for key in stage.requires:
        path = ctx.path(key)
        try:
            present = path.is_file() and path.stat().st_size > 0
        except OSError:
            present = False
        if not present:
            producer = producer_of(key, stages) or "unknown"
            raise MissingInputError(
                f"stage '{stage.id}' requires '{key}' at {path} "
                f"(produced by '{producer}')"
            )


def _run_stage(
    stage: StageSpec,
    ctx: StageContext,
    stages: tuple[StageSpec, ...],
    validate_only: bool,
) -> None:
    _check_requires(stage, ctx, stages)
    if validate_only:
        for key in stage.produces:
            validate_artifact(ctx, key)
        return

    for key in stage.produces:
        ctx.path(key).parent.mkdir(parents=True, exist_ok=True)
    try:
        stage.run(ctx)
    except NotImplementedError as exc:
        raise StageError(f"stage '{stage.id}' is not implemented: {exc}") from exc
    except PipelineError:
        raise
    except Exception as exc:
        raise StageError(f"stage '{stage.id}' failed: {exc}") from exc

    for key in stage.produces:
        validate_artifact(ctx, key)


def run(
    stage_ids: str | Iterable[str],
    workdir: str | Path,
    config_path: str | Path,
    source: str | None = None,
    llm_fixture: str | Path | None = None,
    validate_only: bool = False,
    stages: tuple[StageSpec, ...] = STAGES,
    codec=DEFAULT_CODEC,
) -> int:
    try:
        ids = _normalize_stage_ids(stage_ids)
        selected = _select_stages(ids, stages)
        config = ReadOnlyConfig(config_path)
        fixture = _check_llm_fixture(llm_fixture)
        _check_source(selected, source)

        work_path = Path(workdir)
        if work_path.exists() and not work_path.is_dir():
            raise UsageError(f"workdir is not a directory: {work_path}")
        if not validate_only:
            work_path.mkdir(parents=True, exist_ok=True)

        ctx = StageContext(
            workdir=work_path,
            config=config,
            source=source,
            llm_fixture=fixture,
            validate_only=validate_only,
            artifacts=ARTIFACTS,
            codec=codec,
        )
        for stage in selected:
            _run_stage(stage, ctx, stages, validate_only)
        return EXIT_OK
    except (UsageError, ConfigError, MissingInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (StageError, ContractError, CodecError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as exc:
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        return EXIT_FAILURE


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="pipeline.runner",
        description="Run selected VideoLingo pipeline stages against one workdir.",
    )
    parser.add_argument("--stages", required=True, help="comma-separated stage ids")
    parser.add_argument("--workdir", required=True, help="artifact work directory")
    parser.add_argument("--config", required=True, help="read-only config path")
    parser.add_argument("--source", help="local path or URL for ingest")
    parser.add_argument("--llm-fixture", help="fixture for LLM stages")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate existing artifacts without running stages",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return run(
        args.stages,
        args.workdir,
        args.config,
        source=args.source,
        llm_fixture=args.llm_fixture,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    sys.exit(main())