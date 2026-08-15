"""Focused tests for sequential timestamp alignment with degenerate ASR words.

Whisper can collapse a run of repeated words onto a single point in time
(start == end for every word in the run). Alignment must stay sequential and
best-effort: matched sentences get a positive-duration span instead of a
zero-duration one, and start times never regress.
"""

import unittest

import pandas as pd

from core._6_gen_sub import align_timestamp, get_sentence_timestamps


def make_words(rows):
    return pd.DataFrame(rows, columns=["text", "start", "end"])


class GetSentenceTimestampsTests(unittest.TestCase):
    def test_collapsed_repeated_words_get_extended_span(self):
        df_words = make_words([
            ("你好", 0.0, 0.5),
            ("世界", 0.5, 1.0),
            ("没有", 10.0, 10.0),
            ("没有", 10.0, 10.0),
            ("没有", 10.0, 10.0),
            ("没有", 10.0, 10.0),
            ("没有", 10.0, 10.0),
            ("没有", 10.0, 10.0),
            ("再见", 12.0, 13.0),
        ])
        df_sentences = pd.DataFrame({
            "Source": ["你好 世界", "没有 没有 没有", "没有 没有 没有", "再见"],
        })

        stamps = get_sentence_timestamps(df_words, df_sentences)

        self.assertEqual(stamps[0], (0.0, 1.0))
        # Collapsed matches borrow the next real word end (13.0).
        self.assertEqual(stamps[1], (10.0, 13.0))
        self.assertEqual(stamps[2], (10.0, 13.0))
        self.assertEqual(stamps[3], (12.0, 13.0))
        for start, end in stamps:
            self.assertGreater(end, start)
        starts = [start for start, _ in stamps]
        self.assertEqual(starts, sorted(starts))

    def test_collapsed_words_at_stream_end_use_fallback_duration(self):
        df_words = make_words([
            ("没有", 5.0, 5.0),
            ("没有", 5.0, 5.0),
            ("没有", 5.0, 5.0),
        ])
        df_sentences = pd.DataFrame({"Source": ["没有 没有 没有"]})

        stamps = get_sentence_timestamps(df_words, df_sentences)

        self.assertEqual(len(stamps), 1)
        start, end = stamps[0]
        self.assertEqual(start, 5.0)
        self.assertGreater(end, start)


class AlignTimestampTests(unittest.TestCase):
    def test_align_passes_validation_with_collapsed_words(self):
        df_text = make_words([
            ("没有", 10.0, 10.0),
            ("没有", 10.0, 10.0),
            ("没有", 10.0, 10.0),
            ("没有", 10.0, 10.0),
            ("没有", 10.0, 10.0),
            ("没有", 10.0, 10.0),
            ("再见", 12.0, 13.0),
        ])
        df_translate = pd.DataFrame({
            "Source": ["没有 没有 没有", "没有 没有 没有", "再见"],
            "Translation": ["不，不，不。", "不，不，不。", "再见。"],
        })

        df_time = align_timestamp(
            df_text, df_translate, [("trans.srt", ["Translation"])],
            output_dir=None, for_display=False,
        )

        self.assertTrue((df_time["duration"] > 0).all())
        self.assertEqual(len(df_time), 3)


if __name__ == "__main__":
    unittest.main()
