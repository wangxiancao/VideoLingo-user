import json
from concurrent.futures import ThreadPoolExecutor

from pipeline.asr.language import detect_language
from pipeline.contracts import ConfigError, StageError
from pipeline.llm.cache import ResponseCache, cache_key
from pipeline.llm.contracts import LlmError
from pipeline.llm.factory import create_llm_client

SYSTEM_FAITHFUL = (
    "You translate subtitle segments faithfully into the target language. "
    "Return JSON only: {\"translations\":[{\"id\":1,\"target\":\"...\"}]}. "
    "Preserve names, numbers and meaning; do not add explanations."
)
SYSTEM_REFINE = (
    "You refine subtitle translations for natural, fluent target-language speech. "
    "Do not add facts or explanations. Return JSON only in the same shape, with every id."
)


def _positive_int(value, name):
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be positive")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be positive") from None
    if number <= 0:
        raise ConfigError(f"{name} must be positive")
    return number


def _parse_json(text):
    text = str(text).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise StageError(f"translate: LLM returned invalid JSON: {exc}") from exc


def _parse_translations(raw, expected_ids):
    data = _parse_json(raw)
    translations = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(translations, list):
        raise StageError("translate: LLM response must contain a translations list")
    result = {}
    for item in translations:
        if not isinstance(item, dict):
            raise StageError("translate: each translation must be an object")
        segment_id = item.get("id")
        target = item.get("target")
        if isinstance(segment_id, bool) or not isinstance(segment_id, int):
            raise StageError("translate: translation id must be an integer")
        if not isinstance(target, str) or not target.strip():
            raise StageError(f"translate: translation {segment_id} has empty target")
        if segment_id not in expected_ids:
            raise StageError(f"translate: unexpected translation id {segment_id}")
        if segment_id in result:
            raise StageError(f"translate: duplicate translation id {segment_id}")
        result[segment_id] = target.strip()
    if set(result) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(result))
        raise StageError(f"translate: missing translation id(s): {missing}")
    return result


def _source_language(segments, config):
    text = " ".join(str(segment["text"]) for segment in segments)
    try:
        return detect_language(text)
    except StageError:
        value = config.get("translation.source_language", "auto")
        return str(value) if value else "auto"


def _complete(client, cache, system, user):
    if cache is None:
        return client.complete(system, user)
    key = cache_key(system, user)
    cached = cache.get(key)
    if cached is not None:
        return cached
    response = client.complete(system, user)
    cache.set(key, response)
    return response


def _translate_batch(client, cache, segments, source_language, target_language):
    ids = [segment["id"] for segment in segments]
    faithful_request = {
        "source_language": source_language,
        "target_language": target_language,
        "segments": [
            {"id": segment["id"], "text": segment["text"]} for segment in segments
        ],
    }
    try:
        faithful_raw = _complete(
            client,
            cache,
            SYSTEM_FAITHFUL,
            json.dumps(faithful_request, ensure_ascii=False),
        )
        drafts = _parse_translations(faithful_raw, ids)

        refine_request = {
            "source_language": source_language,
            "target_language": target_language,
            "segments": [
                {
                    "id": segment["id"],
                    "source": segment["text"],
                    "draft": drafts[segment["id"]],
                }
                for segment in segments
            ],
        }
        refined_raw = _complete(
            client,
            cache,
            SYSTEM_REFINE,
            json.dumps(refine_request, ensure_ascii=False),
        )
        refined = _parse_translations(refined_raw, ids)
    except LlmError as exc:
        raise StageError(f"translate: LLM call failed: {exc}") from exc
    return [(segment["id"], refined[segment["id"]]) for segment in segments]


def _batches(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def run(ctx) -> None:
    segments = ctx.read("segments")["segments"]
    if not segments:
        raise StageError("translate: segments must not be empty")
    target_language = ctx.config.get("target_language", "简体中文")
    if not isinstance(target_language, str) or not target_language.strip():
        raise ConfigError("target_language must be a non-empty string")
    batch_size = _positive_int(
        ctx.config.get("translation.batch_size", 8), "translation.batch_size"
    )
    max_workers = _positive_int(ctx.config.get("max_workers", 4), "max_workers")
    source_language = _source_language(segments, ctx.config)

    client = create_llm_client(ctx.config, ctx.llm_fixture, ctx.llm_service)
    cache = None
    if ctx.config.get("translation.cache", True):
        cache = ResponseCache(ctx.workdir / ".llm-cache")

    batches = list(_batches(segments, batch_size))

    def translate_batch(batch):
        return _translate_batch(
            client,
            cache,
            batch,
            source_language,
            target_language.strip(),
        )

    if len(batches) == 1:
        results = [translate_batch(batches[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as pool:
            results = list(pool.map(translate_batch, batches))

    targets = {segment_id: target for batch in results for segment_id, target in batch}
    translated_segments = []
    for segment in segments:
        target = targets.get(segment["id"])
        if not target:
            raise StageError(f"translate: missing refined target for id {segment['id']}")
        translated_segments.append(
            {
                "id": segment["id"],
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "source": str(segment["text"]).strip(),
                "target": target,
            }
        )

    ctx.write(
        "translation",
        {
            "schema_version": 1,
            "source_language": source_language,
            "target_language": target_language.strip(),
            "segments": translated_segments,
        },
    )