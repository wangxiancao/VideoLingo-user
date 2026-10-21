from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from pipeline.contracts import ConfigError


class ReadOnlyConfig:
    """Read a config file once. This class has no write path."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise ConfigError(f"config file not found: {self.path}")
        yaml = YAML(typ="safe")
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = yaml.load(handle)
        except Exception as exc:
            raise ConfigError(f"cannot read config {self.path}: {exc}") from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(f"config {self.path} must be a mapping")
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        current: Any = self._data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def require(self, key: str) -> Any:
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise ConfigError(f"config key not found: {key}")
        return value