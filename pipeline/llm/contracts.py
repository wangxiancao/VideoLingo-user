from pathlib import Path
from typing import Protocol


class LlmError(RuntimeError):
    pass


class LlmClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...