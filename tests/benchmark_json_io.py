import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.codec import JsonCodec


def main() -> None:
    codec = JsonCodec()
    word_count = 0
    segments = []
    for segment_index in range(2000):
        segment_start = segment_index * 1.8
        words = []
        for word_index in range(10):
            start = segment_start + word_index * 0.18
            words.append(
                {
                    "start": start,
                    "end": start + 0.18,
                    "text": f"w{segment_index}_{word_index}",
                }
            )
            word_count += 1
        segments.append(
            {
                "start": segment_start,
                "end": segment_start + 1.8,
                "text": " ".join(word["text"] for word in words),
                "words": words,
            }
        )
    payload = {"schema_version": 1, "language": "en", "segments": segments}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "large_transcript.json"
        start = time.perf_counter()
        codec.write(path, payload)
        write_seconds = time.perf_counter() - start
        start = time.perf_counter()
        loaded = codec.read(path)
        read_seconds = time.perf_counter() - start
        size_mb = path.stat().st_size / (1024 * 1024)

    loaded_words = sum(len(segment["words"]) for segment in loaded["segments"])
    assert loaded_words == word_count
    print(
        f"json_io: {word_count} words, {size_mb:.2f} MiB, "
        f"write={write_seconds:.3f}s, read={read_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()