from pipeline.contracts import ConfigError
from pipeline.llm.contracts import LlmError


class OpenAICompatibleClient:
    def __init__(self, api_key, base_url, model):
        if not isinstance(api_key, str) or not api_key.strip() or api_key == "your-api-key":
            raise ConfigError("api.key must be set for LLM calls")
        if not isinstance(model, str) or not model.strip():
            raise ConfigError("api.model must be set for LLM calls")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigError("openai package is required for LLM calls") from exc
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=120,
        )

    def complete(self, system: str, user: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
            )
        except Exception as exc:
            raise LlmError(f"LLM request failed: {exc}") from exc
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise LlmError("LLM returned an empty response")
        return content