import json

from pipeline.contracts import StageError
from pipeline.llm.contracts import LlmError
from pipeline.llm.factory import create_llm_client

SYSTEM_PROMPT = (
    "You split an ordered ASR word stream into natural sentences. "
    "Return JSON only: {\"sentences\":[{\"start\":0,\"end\":3}, ...]}. "
    "Indices are half-open [start, end) and must cover every word exactly once, in order."
)


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
        raise StageError(f"segment: LLM returned invalid JSON: {exc}") from exc


def _join_words(words, language):
    parts = [str(word["text"]).strip() for word in words if str(word["text"]).strip()]
    if not parts:
        return ""
    prefix = str(language).lower().split("-")[0]
    if prefix in ("zh", "ja", "ko", "yue"):
        return "".join(parts)
    return " ".join(parts)


def _boundaries(data, word_count):
    sentences = data.get("sentences") if isinstance(data, dict) else None
    if not isinstance(sentences, list) or not sentences:
        raise StageError("segment: LLM response must contain a non-empty sentences list")
    boundaries = []
    expected = 0
    for index, item in enumerate(sentences):
        if not isinstance(item, dict):
            raise StageError(f"segment: sentence {index} must be an object")
        start = item.get("start")
        end = item.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise StageError(f"segment: sentence {index} boundaries must be integers")
        if start != expected:
            raise StageError(
                "segment: sentence boundaries must cover words without gaps or overlap"
            )
        if end <= start or end > word_count:
            raise StageError(f"segment: sentence {index} has invalid boundary")
        boundaries.append((start, end))
        expected = end
    if expected != word_count:
        raise StageError("segment: sentence boundaries do not cover all words")
    return boundaries


def _build_segments(transcript, boundaries, language):
    words = [
        word
        for transcript_segment in transcript["segments"]
        for word in transcript_segment["words"]
    ]
    if not words:
        raise StageError("segment: transcript has no words")
    segments = []
    previous_end = None
    for index, (start, end) in enumerate(boundaries, start=1):
        group = words[start:end]
        text = _join_words(group, language)
        if not text:
            raise StageError(f"segment: sentence {index} has empty text")
        segment_start = float(group[0]["start"])
        segment_end = float(group[-1]["end"])
        if segment_end <= segment_start:
            raise StageError(f"segment: sentence {index} has invalid timing")
        if previous_end is not None and segment_start < previous_end - 0.001:
            raise StageError(f"segment: sentence {index} overlaps the previous sentence")
        previous_end = segment_end
        segments.append(
            {
                "id": index,
                "start": segment_start,
                "end": segment_end,
                "text": text,
                "words": [
                    {
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                        "text": str(word["text"]).strip(),
                    }
                    for word in group
                ],
            }
        )
    return segments


def run(ctx) -> None:
    transcript = ctx.read("transcript")
    words = [
        str(word["text"]).strip()
        for transcript_segment in transcript["segments"]
        for word in transcript_segment["words"]
    ]
    if not words:
        raise StageError("segment: transcript has no words")

    client = create_llm_client(ctx.config, ctx.llm_fixture)
    request = {
        "language": transcript["language"],
        "words": [{"index": index, "text": text} for index, text in enumerate(words)],
    }
    try:
        raw = client.complete(SYSTEM_PROMPT, json.dumps(request, ensure_ascii=False))
    except LlmError as exc:
        raise StageError(f"segment: LLM call failed: {exc}") from exc

    boundaries = _boundaries(_parse_json(raw), len(words))
    segments = _build_segments(transcript, boundaries, transcript["language"])
    ctx.write("segments", {"schema_version": 1, "segments": segments})