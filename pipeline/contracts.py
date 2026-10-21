from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


class PipelineError(Exception):
    """Base class for expected pipeline errors."""


class UsageError(PipelineError):
    """Bad stage selection or CLI usage."""


class ConfigError(PipelineError):
    """Config file is missing or invalid."""


class MissingInputError(PipelineError):
    """A required artifact or external input is missing."""


class StageError(PipelineError):
    """A stage raised an error."""


class ContractError(PipelineError):
    """A produced artifact violates its contract."""


class CodecError(PipelineError):
    """An artifact could not be read or written."""


@dataclass(frozen=True)
class Artifact:
    key: str
    relpath: str
    kind: str


@dataclass(frozen=True)
class StageSpec:
    id: str
    title: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    legacy: tuple[str, ...]
    run: Callable[["StageContext"], None]
    external_input: str | None = None


@dataclass
class StageContext:
    workdir: Path
    config: Any
    source: str | None
    llm_fixture: Path | None
    validate_only: bool
    artifacts: Mapping[str, Artifact]
    codec: Any

    def path(self, key: str) -> Path:
        artifact = self.artifacts.get(key)
        if artifact is None:
            raise UsageError(f"unknown artifact key: {key}")
        return self.workdir / artifact.relpath

    def read(self, key: str) -> Any:
        artifact = self.artifacts.get(key)
        if artifact is None:
            raise UsageError(f"unknown artifact key: {key}")
        if artifact.kind != "json":
            raise UsageError(f"artifact '{key}' is not JSON")
        return self.codec.read(self.path(key))

    def write(self, key: str, payload: Any) -> None:
        artifact = self.artifacts.get(key)
        if artifact is None:
            raise UsageError(f"unknown artifact key: {key}")
        if artifact.kind != "json":
            raise UsageError(f"artifact '{key}' is not JSON")
        self.codec.write(self.path(key), payload)