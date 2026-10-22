from pathlib import Path

from pipeline.asr.glm import GlmAsrBackend
from pipeline.contracts import ConfigError


def _positive_number(value, name, cast):
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a positive number")
    try:
        number = cast(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a positive number") from None
    if number <= 0:
        raise ConfigError(f"{name} must be a positive number")
    return number


def create_backend(config):
    cache = config.get("asr.modelscope_cache", "~/.cache/modelscope")
    chunk_seconds = _positive_number(
        config.get("asr.chunk_seconds", 30), "asr.chunk_seconds", float
    )

    name = config.get("asr.backend", "glm_asr")
    if name == "glm_asr_native":
        from pipeline.asr.native import NativeAsrBackend

        binary = config.get("asr.native_binary", "native/asr/build/glm_asr_m0")
        if not isinstance(binary, str) or not binary.strip():
            raise ConfigError("asr.native_binary must be a non-empty path")
        return NativeAsrBackend(
            cache_dir=Path(cache).expanduser(),
            binary=binary,
            chunk_seconds=chunk_seconds,
        )
    if name != "glm_asr":
        raise ConfigError(f"unsupported asr.backend: {name}")
    if not isinstance(cache, str) or not cache.strip():
        raise ConfigError("asr.modelscope_cache must be a non-empty path")

    batch_size = _positive_number(
        config.get("asr.batch_size", 20), "asr.batch_size", int
    )
    max_new_tokens = _positive_number(
        config.get("asr.max_new_tokens", 512), "asr.max_new_tokens", int
    )
    return GlmAsrBackend(
        cache_dir=Path(cache).expanduser(),
        chunk_seconds=chunk_seconds,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )