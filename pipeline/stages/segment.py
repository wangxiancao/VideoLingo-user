import json

from pipeline.contracts import StageError

# V4-3: rule-based segmentation (maintainer-approved fallback, #32 decision
# 2026-10-23). The V4-1 spike showed HY-MT2-1.8B cannot segment reliably
# (skipped ranges / mechanical 4-word chunks), so boundaries now come from
# deterministic word-stream rules; the segments.json contract is unchanged.
SENTENCE_TERMINALS = (".", "!", "?", "。", "！", "？", "…")
DEFAULT_PAUSE_S = 0.45
DEFAULT_MAX_WORDS = 20
DEFAULT_MAX_DURATION_S = 8.0


def _join_words(words, language):
    parts = [str(word["text"]).strip() for word in words if str(word["text"]).strip()]
    if not parts:
        return ""
    prefix = str(language).lower().split("-")[0]
    if prefix in ("zh", "ja", "ko", "yue"):
        return "".join(parts)
    return " ".join(parts)


def _is_terminal(word):
    return str(word["text"]).strip().endswith(SENTENCE_TERMINALS)


def _rule_boundaries(words, pause_s, max_words, max_duration_s):
    """Half-open [start, end) word-index boundaries covering every word once.

    Cuts after a word when it ends a sentence (terminal punctuation), when
    the next word starts after a silence longer than `pause_s`, or when the
    running sentence would exceed `max_words` / `max_duration_s`. Best-effort
    by design: imperfect boundaries are acceptable, gaps and overlaps are not.
    """
    boundaries = []
    start = 0
    for index in range(len(words)):
        group_words = index - start + 1
        duration = float(words[index]["end"]) - float(words[start]["start"])
        last = index == len(words) - 1
        cut = False
        if last:
            cut = True
        elif _is_terminal(words[index]):
            cut = True
        elif float(words[index + 1]["start"]) - float(words[index]["end"]) > pause_s:
            cut = True
        elif group_words >= max_words or duration >= max_duration_s:
            cut = True
        if cut and group_words > 0:
            boundaries.append((start, index + 1))
            start = index + 1
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
        # Adjacent aligner words can overlap by a few ms (zero-duration
        # clamps, boundary artifacts); best-effort segmentation tolerates
        # that rather than failing the run, but anything larger is real
        # disorder and still fails the contract.
        if previous_end is not None and segment_start < previous_end - 0.05:
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
    all_words = [
        word
        for transcript_segment in transcript["segments"]
        for word in transcript_segment["words"]
    ]
    if not all_words:
        raise StageError("segment: transcript has no words")

    boundaries = _rule_boundaries(
        all_words,
        pause_s=float(ctx.config.get("segmentation.pause_s", DEFAULT_PAUSE_S)),
        max_words=int(ctx.config.get("segmentation.max_words", DEFAULT_MAX_WORDS)),
        max_duration_s=float(
            ctx.config.get("segmentation.max_duration_s", DEFAULT_MAX_DURATION_S)
        ),
    )
    segments = _build_segments(transcript, boundaries, transcript["language"])
    ctx.write("segments", {"schema_version": 1, "segments": segments})
