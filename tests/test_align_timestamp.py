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


class WideWordDistributionTests(unittest.TestCase):
    """Consecutive sentences matching one wide ASR word, followed by another
    sentence matching the SAME wide word (spilling into the next word), must
    not produce regressing starts — validate_positive_timestamps rejected the
    pre-fix output as 'Subtitle timestamps regress in N row(s)'."""

    def test_straddler_after_wide_word_group_does_not_regress(self):
        # Giant wide word spanning [100, 126]; two sentences inside it, then a
        # third that starts in its tail and spills into the next word.
        wide = "".join([
            "今天天氣很好我們去公園散步聊聊最近的事情",   # 20 chars
            "所以我決定明天再出發",                      # 10 chars
        ])
        df_words = pd.DataFrame({
            "text": ["開頭", wide, "結束"],
            "start": [0.0, 100.0, 126.5],
            "end":   [3.0, 126.0, 128.0],
        })
        s1, s2, s3 = "今天天氣很好我們去公園散步", "聊聊最近的事情", "所以我決定明天再出發結束"
        df_sentences = pd.DataFrame({
            "Source": [s1, s2, s3],
            "Translation": ["t1", "t2", "t3"],
        })

        stamps = get_sentence_timestamps(df_words, df_sentences)
        self.assertEqual(len(stamps), 3)
        starts = [s for s, _ in stamps]
        # Monotonic non-regressing starts.
        self.assertEqual(starts, sorted(starts))
        self.assertGreaterEqual(starts[0], 100.0)
        self.assertLessEqual(starts[2], 128.0)
        self.assertGreater(stamps[1][1], stamps[0][0])
        self.assertGreater(stamps[2][1], stamps[2][0])

        # End-to-end: validate_positive_timestamps must accept the output and
        # normalized rows must keep positive duration (no start+ε collapse).
        df_time = align_timestamp(
            df_words, df_sentences, [("trans.srt", ["Translation"])],
            output_dir=None, for_display=False,
        )
        self.assertTrue((df_time["duration"] > 0).all())
        self.assertEqual(len(df_time), 3)
        durs = df_time["duration"].tolist()
        # No cue may collapse to the MIN_DURATION_EPS floor (sub-second).
        self.assertTrue(all(d > 0.5 for d in durs), durs)


if __name__ == "__main__":
    unittest.main()
