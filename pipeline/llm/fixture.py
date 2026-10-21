import json
from pathlib import Path

from pipeline.llm.contracts import LlmError


class FixtureLlmClient:
    def __init__(self, path):
        self.path = Path(path)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LlmError(f"cannot read LLM fixture {self.path}: {exc}") from exc
        responses = data.get("responses")
        if not isinstance(responses, list) or not responses:
            raise LlmError(f"LLM fixture {self.path} has no responses")
        self.responses = responses
        self.index = 0

    def complete(self, system: str, user: str) -> str:
        if self.index >= len(self.responses):
            raise LlmError(f"LLM fixture {self.path} has no response left")
        response = self.responses[self.index]
        self.index += 1
        return json.dumps(response, ensure_ascii=False)