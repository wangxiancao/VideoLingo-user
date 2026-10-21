import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pipeline.contracts import CodecError


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


class JsonCodec:
    """Default artifact codec. The runner and stages only see decoded objects."""

    def read(self, path: Path) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle, parse_constant=_reject_constant)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise CodecError(f"cannot read JSON {path}: {exc}") from exc

    def write(self, path: Path, payload: Any) -> None:
        path = Path(path)
        tmp_name = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{path.name}.",
                suffix=".tmp",
                dir=str(path.parent),
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                handle.write("\n")
            os.replace(tmp_name, path)
        except (OSError, TypeError, ValueError) as exc:
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
            raise CodecError(f"cannot write JSON {path}: {exc}") from exc


DEFAULT_CODEC = JsonCodec()