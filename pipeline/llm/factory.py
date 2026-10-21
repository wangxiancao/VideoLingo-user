from pathlib import Path

from pipeline.llm.fixture import FixtureLlmClient
from pipeline.llm.openai_client import OpenAICompatibleClient


def create_llm_client(config, fixture_path=None):
    if fixture_path is not None:
        return FixtureLlmClient(Path(fixture_path))
    return OpenAICompatibleClient(
        api_key=config.get("api.key"),
        base_url=config.get("api.base_url"),
        model=config.get("api.model"),
    )