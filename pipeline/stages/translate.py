import json
import sys
from concurrent.futures import ThreadPoolExecutor

from pipeline.asr.language import detect_language
from pipeline.contracts import ConfigError, StageError
from pipeline.llm.cache import ResponseCache, cache_key
from pipeline.llm.contracts import LlmError
from pipeline.llm.factory import create_llm_client

# Id-free ordered-array prompts (V4-7 O4). The earlier id-carrying JSON
# protocol failed three ways on small models: echoed the placeholder schema
# verbatim, renumbered ids after the example, and dropped trailing entries.
# Sending an ordered array of plain strings and asking for an ordered array
# of translated strings back removes the id dimension entirely — order is
# the only contract, and the stage maps results back to ids by index.
SYSTEM_FAITHFUL = (
    "You translate subtitle segments faithfully into the target language. "
    "The request lists segments as an ordered array of strings. Output ONLY "
    "a JSON object shaped exactly like "
    '{"translations":["Hello world.", "Second translation."]}'
    " with exactly one translated string per input segment, in the same "
    "order, never empty. Preserve names, numbers and meaning."
)
SYSTEM_REFINE = (
    "You refine draft subtitle translations for natural, fluent "
    "target-language speech. The request lists sources and drafts as ordered "
    "arrays of strings. Output ONLY a JSON object shaped exactly like "
    '{"translations":["Refined first sentence.", "Refined second."]}'
    " with exactly one refined string per draft, in the same order, never "
    "empty. Do not add facts or explanations."
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
        # Small models sometimes append a second JSON object or trailing
        # prose after a complete response ("Extra data"); the contract only
        # needs the first complete value, so take it and drop the tail.
        try:
            value, _ = json.JSONDecoder().raw_decode(text.lstrip())
        except json.JSONDecodeError:
            raise StageError(
                f"translate: LLM returned invalid JSON: {exc}"
            ) from exc
        return value


def _parse_translation_strings(raw, expected_count):
    """Validate an ordered-array response: one non-empty string per input."""
    data = _parse_json(raw)
    translations = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(translations, list):
        raise StageError("translate: LLM response must contain a translations list")
    if len(translations) != expected_count:
        raise StageError(
            f"translate: expected {expected_count} translations, "
            f"got {len(translations)}"
        )
    result = []
    for index, item in enumerate(translations):
        if not isinstance(item, str) or not item.strip():
            raise StageError(f"translate: translation {index + 1} is empty")
        result.append(item.strip())
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


def _call_and_parse(client, cache, system, user, expected_count):
    """One LLM round with a single retry on contract failure.

    Small models miss the ordered-array contract on garbled batches; a
    fresh greedy call usually heals it. The failed response is cached
    before parsing, so evict it before retrying.
    """
    raw = _complete(client, cache, system, user)
    try:
        return _parse_translation_strings(raw, expected_count)
    except StageError:
        if cache is not None:
            cache.delete(cache_key(system, user))
        raw = _complete(client, cache, system, user)
        return _parse_translation_strings(raw, expected_count)


def _apply_term_map(text, term_map):
    """Normalize known ASR garbles before the text reaches the LLM.

    Only the request payload is normalized; the stored source text keeps the
    raw transcript.
    """
    if not term_map:
        return text
    for wrong, right in sorted(term_map.items(), key=lambda item: -len(item[0])):
        text = text.replace(str(wrong), str(right))
    return text


def _translate_batch(client, cache, segments, source_language, target_language,
                     term_map):
    """Translate one batch, escalating on contract failures.

    A same-batch retry heals stochastic misses; deterministic id drops
    (seen on the real 60-min run: id 32 dropped twice at batch 8) heal when
    the batch is bisected, so split and recurse before giving up.
    """
    try:
        return _translate_batch_once(
            client, cache, segments, source_language, target_language, term_map
        )
    except StageError:
        if len(segments) == 1:
            # Terminal fallback at the bisection floor (retry already ran):
            # for untranslatable ASR fragments — empty strings, count drift
            # on a single line — echo the raw source rather than failing the
            # run or hallucinating. One segment at a time, always logged.
            # The subtitle stage caps cue length, so cap the echo too.
            source = _apply_term_map(segments[0]["text"], term_map).strip()
            if len(source) > 87:
                source = source[:87] + "…"
            print(
                f"translate: segment {segments[0]['id']} untranslatable "
                f"({segments[0]['text'][:30]!r}); keeping the source text",
                file=sys.stderr,
            )
            return [(segments[0]["id"], source)]
        middle = len(segments) // 2
        return (
            _translate_batch(client, cache, segments[:middle],
                             source_language, target_language, term_map)
            + _translate_batch(client, cache, segments[middle:],
                               source_language, target_language, term_map)
        )


def _translate_batch_once(client, cache, segments, source_language,
                          target_language, term_map):
    texts = [_apply_term_map(segment["text"], term_map) for segment in segments]
    faithful_request = {
        "source_language": source_language,
        "target_language": target_language,
        "segments": texts,
    }
    try:
        drafts = _call_and_parse(
            client, cache, SYSTEM_FAITHFUL,
            json.dumps(faithful_request, ensure_ascii=False), len(texts),
        )

        refine_request = {
            "source_language": source_language,
            "target_language": target_language,
            "sources": texts,
            "drafts": drafts,
        }
        refined = _call_and_parse(
            client, cache, SYSTEM_REFINE,
            json.dumps(refine_request, ensure_ascii=False), len(texts),
        )
    except LlmError as exc:
        raise StageError(f"translate: LLM call failed: {exc}") from exc
    return [(segment["id"], refined[index]) for index, segment in enumerate(segments)]


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
    if ctx.llm_service is not None:
        # The managed local service knows its runtime's reliability curve
        # (Q4 wants small batches, BF16 handles 12); defer to it.
        batch_size = min(batch_size, int(ctx.llm_service.batch_segments))
    max_workers = _positive_int(ctx.config.get("max_workers", 4), "max_workers")
    source_language = _source_language(segments, ctx.config)

    client = create_llm_client(ctx.config, ctx.llm_fixture, ctx.llm_service)
    cache = None
    if ctx.config.get("translation.cache", True):
        cache = ResponseCache(ctx.workdir / ".llm-cache")
    term_map = ctx.config.get("translation.term_map", {})
    if term_map is None:
        term_map = {}
    if not isinstance(term_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in term_map.items()
    ):
        raise ConfigError("translation.term_map must be a string-to-string mapping")

    batches = list(_batches(segments, batch_size))

    def translate_batch(batch):
        return _translate_batch(
            client,
            cache,
            batch,
            source_language,
            target_language.strip(),
            term_map,
        )

    if len(batches) == 1:
        results = [translate_batch(batches[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as pool:
            results = list(pool.map(translate_batch, batches))

    targets = {segment_id: target for batch in results for segment_id, target in batch}
    translated_segments = []
    # The subtitle stage caps cue length (75 chars x 1.2 multiplier); very
    # verbose translations of garbled run-ons would otherwise fail there
    # when the segment is too short to split into more cues.
    target_limit = 87
    for segment in segments:
        target = targets.get(segment["id"])
        if not target:
            raise StageError(f"translate: missing refined target for id {segment['id']}")
        if len(target) > target_limit + 1:
            target = target[:target_limit] + "…"
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