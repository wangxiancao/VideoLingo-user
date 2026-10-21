from pipeline.asr.glm import _clamp_words
from bench.asr_rtfx import chunk_audio


def test_word_slightly_past_chunk_end_is_clamped():
    words = _clamp_words(
        [{"text": "surface", "start_time": 30.08, "end_time": 30.08}],
        offset=270.0,
        chunk_duration=30.0,
    )

    assert len(words) == 1
    assert words[0].text == "surface"
    assert words[0].start < 300.0
    assert words[0].end == 300.0
    assert words[0].end > words[0].start


def test_benchmark_chunker_merges_short_tail():
    samples = [0.0] * 480064

    chunks = chunk_audio(samples, 16000, 30)

    assert len(chunks) == 1
    assert len(chunks[0]) == 480064