import math
import os
from pathlib import Path

from pipeline.contracts import ConfigError, StageError

DEFAULT_MAX_LENGTH = 75
DEFAULT_TARGET_MULTIPLIER = 1.2
DEFAULT_MIN_DURATION = 0.5
DEFAULT_MAX_DURATION = 10.0


def _positive(value, name, cast):
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be positive")
    try:
        number = cast(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be positive") from None
    if number <= 0:
        raise ConfigError(f"{name} must be positive")
    return number


def _settings(config):
    return {
        "max_length": _positive(
            config.get("subtitle.max_length", DEFAULT_MAX_LENGTH),
            "subtitle.max_length",
            int,
        ),
        "target_multiplier": _positive(
            config.get("subtitle.target_multiplier", DEFAULT_TARGET_MULTIPLIER),
            "subtitle.target_multiplier",
            float,
        ),
        "min_duration": _positive(
            config.get("subtitle.min_duration", DEFAULT_MIN_DURATION),
            "subtitle.min_duration",
            float,
        ),
        "max_duration": _positive(
            config.get("subtitle.max_duration", DEFAULT_MAX_DURATION),
            "subtitle.max_duration",
            float,
        ),
    }


def _normalize(text):
    return " ".join(str(text).split())


def _units(text):
    return text.split(" ") if " " in text else list(text)


def _needed_count(text, limit):
    return max(1, math.ceil(len(text) / limit))


def _split_into(text, count):
    text = _normalize(text)
    if count <= 1:
        return [text]
    units = _units(text)
    count = min(count, len(units))
    parts = []
    for index in range(count):
        start = round(index * len(units) / count)
        end = round((index + 1) * len(units) / count)
        part = " ".join(units[start:end]) if " " in text else "".join(units[start:end])
        part = part.strip()
        if part:
            parts.append(part)
    return parts


def _time_text(seconds):
    if seconds < 0:
        raise StageError("subtitle timestamp must not be negative")
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segment_cues(segment, index, settings):
    start = float(segment["start"])
    end = float(segment["end"])
    source = _normalize(segment["source"])
    target = _normalize(segment["target"])
    if not source or not target:
        raise StageError(f"translation segment {index} has empty source or target")
    if end <= start:
        raise StageError(f"translation segment {index} has invalid timing")

    source_units = len(_units(source))
    target_units = len(_units(target))
    cue_count = max(
        _needed_count(source, settings["max_length"]),
        _needed_count(target, round(settings["max_length"] * settings["target_multiplier"])),
        math.ceil((end - start) / settings["max_duration"]),
        1,
    )
    cue_count = max(1, min(cue_count, source_units, target_units))
    while cue_count > 1 and (end - start) / cue_count < settings["min_duration"]:
        cue_count -= 1

    source_parts = _split_into(source, cue_count)
    target_parts = _split_into(target, cue_count)
    if len(source_parts) != len(target_parts) or len(source_parts) != cue_count:
        raise StageError(f"translation segment {index} could not be split into {cue_count} cues")

    source_limit = settings["max_length"]
    target_limit = round(settings["max_length"] * settings["target_multiplier"])
    cues = []
    duration = end - start
    for part_index, (source_part, target_part) in enumerate(
        zip(source_parts, target_parts)
    ):
        if len(source_part) > source_limit:
            raise StageError(
                f"translation segment {index} source cue exceeds {source_limit} characters"
            )
        if len(target_part) > target_limit:
            raise StageError(
                f"translation segment {index} target cue exceeds {target_limit} characters"
            )
        cue_start = start + duration * part_index / cue_count
        cue_end = start + duration * (part_index + 1) / cue_count
        if part_index == cue_count - 1:
            cue_end = end
        cues.append(
            {
                "start": cue_start,
                "end": cue_end,
                "source": source_part,
                "target": target_part,
            }
        )
    return cues


def _srt_text(cues, bilingual):
    blocks = []
    for index, cue in enumerate(cues, start=1):
        text = (
            f"{cue['source']}\n{cue['target']}" if bilingual else cue["source"]
        )
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_time_text(cue['start'])} --> {_time_text(cue['end'])}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def _write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def run(ctx) -> None:
    settings = _settings(ctx.config)
    translation = ctx.read("translation")
    cues = []
    for index, segment in enumerate(translation["segments"]):
        cues.extend(_segment_cues(segment, index, settings))
    if not cues:
        raise StageError("subtitle stage produced no cues")

    previous_end = None
    for index, cue in enumerate(cues, start=1):
        if cue["end"] <= cue["start"]:
            raise StageError(f"subtitle cue {index} has invalid timing")
        if previous_end is not None and cue["start"] < previous_end - 0.001:
            raise StageError(f"subtitle cue {index} overlaps the previous cue")
        previous_end = cue["end"]

    source_srt = _srt_text(cues, bilingual=False)
    # Source and target must share the same cue timeline; use source text for target-only file.
    target_cues = [dict(cue, source=cue["target"]) for cue in cues]
    target_srt = _srt_text(target_cues, bilingual=False)
    bilingual_srt = _srt_text(cues, bilingual=True)

    _write_atomic(ctx.path("srt_source"), source_srt)
    _write_atomic(ctx.path("srt_target"), target_srt)
    _write_atomic(ctx.path("srt_bilingual"), bilingual_srt)