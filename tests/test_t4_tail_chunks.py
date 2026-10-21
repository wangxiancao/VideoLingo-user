from pipeline.asr.glm import _chunks


def test_short_tail_is_merged_into_previous_chunk():
    samples = [0.0] * int(30.016 * 16000)

    chunks = _chunks(samples, 16000, 30)

    assert len(chunks) == 1
    assert len(chunks[0]) == int(30.016 * 16000)


def test_short_tail_after_full_chunks_is_merged():
    samples = [0.0] * int(60.016 * 16000)

    chunks = _chunks(samples, 16000, 30)

    assert [len(chunk) for chunk in chunks] == [480000, int(30.016 * 16000)]


def test_normal_full_chunks_are_unchanged():
    samples = [0.0] * (16000 * 60)

    chunks = _chunks(samples, 16000, 30)

    assert [len(chunk) for chunk in chunks] == [480000, 480000]