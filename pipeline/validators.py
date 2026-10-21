import math
import re
from typing import Any

from pipeline.contracts import CodecError, ContractError, StageContext

SCHEMA_VERSION = 1
TIMESTAMP_TOLERANCE = 0.05
TIME_EQUAL_TOLERANCE = 0.001

SRT_TIME_RE = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})$"
)


def _fail(ctx: StageContext, key: str, message: str) -> None:
    raise ContractError(f"{key} ({ctx.path(key)}): {message}")


def _read_json(ctx: StageContext, key: str) -> Any:
    try:
        return ctx.read(key)
    except CodecError as exc:
        raise ContractError(f"{key} ({ctx.path(key)}): {exc}") from exc


def _mapping(value: Any, ctx: StageContext, key: str, field: str) -> dict:
    if not isinstance(value, dict):
        _fail(ctx, key, f"{field} must be an object")
    return value


def _list(value: Any, ctx: StageContext, key: str, field: str) -> list:
    if not isinstance(value, list):
        _fail(ctx, key, f"{field} must be a list")
    return value


def _text(value: Any, ctx: StageContext, key: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(ctx, key, f"{field} must be a non-empty string")
    return value


def _number(value: Any, ctx: StageContext, key: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(ctx, key, f"{field} must be a finite number")
    if not math.isfinite(value):
        _fail(ctx, key, f"{field} must be a finite number")
    return float(value)


def _check_schema(data: Any, ctx: StageContext, key: str) -> dict:
    document = _mapping(data, ctx, key, "document")
    if document.get("schema_version") != SCHEMA_VERSION:
        _fail(ctx, key, f"schema_version must be {SCHEMA_VERSION}")
    return document


def _validate_words(
    words: list,
    seg_start: float,
    seg_end: float,
    segment_index: int,
    ctx: StageContext,
    key: str,
) -> None:
    previous_start = None
    for word_index, word in enumerate(words):
        field = f"segments[{segment_index}].words[{word_index}]"
        item = _mapping(word, ctx, key, field)
        start = _number(item.get("start"), ctx, key, f"{field}.start")
        end = _number(item.get("end"), ctx, key, f"{field}.end")
        if end <= start:
            _fail(ctx, key, f"{field}.end must be > start")
        if start < seg_start - TIMESTAMP_TOLERANCE or end > seg_end + TIMESTAMP_TOLERANCE:
            _fail(ctx, key, f"{field} is outside its segment")
        if previous_start is not None and start < previous_start:
            _fail(ctx, key, f"{field}.start regressed")
        previous_start = start
        _text(item.get("text"), ctx, key, f"{field}.text")


def _validate_timing(
    segments: list,
    ctx: StageContext,
    key: str,
    require_ids: bool,
) -> None:
    if not segments:
        _fail(ctx, key, "segments must not be empty")
    previous_start = None
    previous_end = None
    for index, raw_segment in enumerate(segments):
        field = f"segments[{index}]"
        segment = _mapping(raw_segment, ctx, key, field)
        if require_ids and segment.get("id") != index + 1:
            _fail(ctx, key, f"{field}.id must be {index + 1}")
        start = _number(segment.get("start"), ctx, key, f"{field}.start")
        end = _number(segment.get("end"), ctx, key, f"{field}.end")
        if start < 0:
            _fail(ctx, key, f"{field}.start must be >= 0")
        if end <= start:
            _fail(ctx, key, f"{field}.end must be > start")
        if previous_start is not None and start < previous_start:
            _fail(ctx, key, f"{field}.start regressed")
        if previous_end is not None and start < previous_end - TIMESTAMP_TOLERANCE:
            _fail(ctx, key, f"{field} overlaps the previous segment")
        previous_start = start
        previous_end = end


def _validate_segments(
    segments: list,
    ctx: StageContext,
    key: str,
    require_ids: bool,
) -> None:
    _validate_timing(segments, ctx, key, require_ids)
    for index, raw_segment in enumerate(segments):
        field = f"segments[{index}]"
        segment = _mapping(raw_segment, ctx, key, field)
        _text(segment.get("text"), ctx, key, f"{field}.text")
        words = _list(segment.get("words"), ctx, key, f"{field}.words")
        if not words:
            _fail(ctx, key, f"{field}.words must not be empty")
        _validate_words(
            words,
            float(segment["start"]),
            float(segment["end"]),
            index,
            ctx,
            key,
        )


def validate_transcript(ctx: StageContext) -> None:
    data = _check_schema(_read_json(ctx, "transcript"), ctx, "transcript")
    _text(data.get("language"), ctx, "transcript", "language")
    segments = _list(data.get("segments"), ctx, "transcript", "segments")
    _validate_segments(segments, ctx, "transcript", require_ids=False)


def validate_segments(ctx: StageContext) -> None:
    data = _check_schema(_read_json(ctx, "segments"), ctx, "segments")
    segments = _list(data.get("segments"), ctx, "segments", "segments")
    _validate_segments(segments, ctx, "segments", require_ids=True)


def validate_translation(ctx: StageContext) -> None:
    data = _check_schema(_read_json(ctx, "translation"), ctx, "translation")
    _text(data.get("source_language"), ctx, "translation", "source_language")
    _text(data.get("target_language"), ctx, "translation", "target_language")
    segments = _list(data.get("segments"), ctx, "translation", "segments")
    _validate_timing(segments, ctx, "translation", require_ids=True)
    for index, raw_segment in enumerate(segments):
        field = f"segments[{index}]"
        segment = _mapping(raw_segment, ctx, "translation", field)
        _text(segment.get("source"), ctx, "translation", f"{field}.source")
        _text(segment.get("target"), ctx, "translation", f"{field}.target")

    reference = _read_json(ctx, "segments")
    reference_segments = reference.get("segments") if isinstance(reference, dict) else None
    if not isinstance(reference_segments, list) or len(reference_segments) != len(segments):
        _fail(ctx, "translation", "segment count does not match segments.json")
    for index, (translated, source) in enumerate(zip(segments, reference_segments)):
        if not isinstance(source, dict) or translated.get("id") != source.get("id"):
            _fail(ctx, "translation", f"segments[{index}].id does not match segments.json")
        try:
            start_delta = abs(float(translated["start"]) - float(source["start"]))
            end_delta = abs(float(translated["end"]) - float(source["end"]))
        except (KeyError, TypeError, ValueError) as exc:
            _fail(ctx, "translation", f"segments[{index}] has invalid reference timing: {exc}")
        if start_delta > TIME_EQUAL_TOLERANCE or end_delta > TIME_EQUAL_TOLERANCE:
            _fail(ctx, "translation", f"segments[{index}] timing drifted from segments.json")


def _time_to_seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def _parse_srt(ctx: StageContext, key: str) -> list[dict]:
    path = ctx.path(key)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"{key} ({path}): cannot read SRT: {exc}") from exc
    if not text.strip():
        _fail(ctx, key, "SRT file is empty")
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n") if block.strip()]
    cues = []
    for block_index, block in enumerate(blocks):
        lines = block.split("\n")
        if len(lines) < 3:
            _fail(ctx, key, f"SRT cue {block_index + 1} needs index, time and text")
        raw_index = lines[0].strip()
        if not raw_index.isdigit():
            _fail(ctx, key, f"SRT cue {block_index + 1} has a bad index")
        index = int(raw_index)
        if index != block_index + 1:
            _fail(ctx, key, f"SRT cue index must be {block_index + 1}, got {index}")
        match = SRT_TIME_RE.match(lines[1].strip())
        if not match:
            _fail(ctx, key, f"SRT cue {index} has a bad timestamp line")
        start = _time_to_seconds(*match.groups()[:4])
        end = _time_to_seconds(*match.groups()[4:])
        if end <= start:
            _fail(ctx, key, f"SRT cue {index} end must be > start")
        text_lines = [line for line in lines[2:] if line.strip()]
        if not text_lines:
            _fail(ctx, key, f"SRT cue {index} has empty text")
        cues.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "text": "\n".join(text_lines),
            }
        )
    return cues


def _validate_cues(cues: list[dict], ctx: StageContext, key: str) -> None:
    previous_start = None
    previous_end = None
    for cue in cues:
        if previous_start is not None and cue["start"] < previous_start:
            _fail(ctx, key, f"SRT cue {cue['index']} start regressed")
        if previous_end is not None and cue["start"] < previous_end - TIME_EQUAL_TOLERANCE:
            _fail(ctx, key, f"SRT cue {cue['index']} overlaps the previous cue")
        previous_start = cue["start"]
        previous_end = cue["end"]


def validate_srt(ctx: StageContext, key: str) -> None:
    _validate_cues(_parse_srt(ctx, key), ctx, key)


def validate_srt_set(ctx: StageContext) -> None:
    cues_by_key = {}
    for key in ("srt_source", "srt_target", "srt_bilingual"):
        cues = _parse_srt(ctx, key)
        _validate_cues(cues, ctx, key)
        cues_by_key[key] = cues
    counts = {key: len(cues) for key, cues in cues_by_key.items()}
    if len(set(counts.values())) != 1:
        _fail(ctx, "srt_bilingual", "SRT files have different cue counts")
    reference = cues_by_key["srt_source"]
    for key in ("srt_target", "srt_bilingual"):
        for index, (left, right) in enumerate(zip(reference, cues_by_key[key])):
            if (
                abs(left["start"] - right["start"]) > TIME_EQUAL_TOLERANCE
                or abs(left["end"] - right["end"]) > TIME_EQUAL_TOLERANCE
            ):
                _fail(ctx, key, f"SRT cue {index + 1} timing differs from srt_source")


def validate_media(ctx: StageContext, key: str) -> None:
    """T1 checks existence and size only; codec and duration checks belong to T3/T8."""
    path = ctx.path(key)
    try:
        if not path.is_file():
            _fail(ctx, key, "file is missing")
        if path.stat().st_size <= 0:
            _fail(ctx, key, "file is empty")
    except OSError as exc:
        raise ContractError(f"{key} ({path}): cannot stat file: {exc}") from exc


VALIDATORS = {
    "source_video": lambda ctx: validate_media(ctx, "source_video"),
    "audio": lambda ctx: validate_media(ctx, "audio"),
    "transcript": validate_transcript,
    "segments": validate_segments,
    "translation": validate_translation,
    "srt_source": lambda ctx: validate_srt(ctx, "srt_source"),
    "srt_target": lambda ctx: validate_srt(ctx, "srt_target"),
    "srt_bilingual": validate_srt_set,
    "output_video": lambda ctx: validate_media(ctx, "output_video"),
}


def validate_artifact(ctx: StageContext, key: str) -> None:
    validator = VALIDATORS.get(key)
    if validator is None:
        raise ContractError(f"no validator for artifact '{key}'")
    validator(ctx)