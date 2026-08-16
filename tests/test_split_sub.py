"""Focused tests for equal-count, zip-safe parallel part handling in stage 5.

The LLM split/alignment can return a different number of target parts than the
source was split into, and a line can fail alignment entirely. Anything that
`zip`s the two sides would silently truncate the longer one and drop text, so
align_subs and both defensive passes must pad the shorter side with '' pieces
(stream-preserving under "".join, skipped by stage 6). These tests pin the
deterministic behavior; the LLM-dependent align_subs path is covered by the
same helper.
"""

import unittest
from unittest import mock

from core._5_split_sub import _pad_equal_parts, split_pair_to_fit, align_subs


class PadEqualPartsTests(unittest.TestCase):
    def test_pads_target_when_source_is_longer(self):
        src_parts = ["Hello ", "world, ", "goodbye!"]
        tr_parts = ["你好世界！"]  # shorter -> padded, never source-truncated
        padded_src, padded_tr = _pad_equal_parts(src_parts, tr_parts)

        self.assertEqual(len(padded_src), len(padded_tr))
        self.assertEqual(padded_tr, ["你好世界！", "", ""])
        # text stream preserved on BOTH sides
        self.assertEqual("".join(padded_src), "Hello world, goodbye!")
        self.assertEqual("".join(padded_tr), "你好世界！")

    def test_pads_source_when_target_is_longer(self):
        src_parts = ["one"]
        tr_parts = ["a", "b", "c"]  # shorter side improved -> no truncation
        padded_src, padded_tr = _pad_equal_parts(src_parts, tr_parts)

        self.assertEqual(len(padded_src), len(padded_tr))
        self.assertEqual(padded_src, ["one", "", ""])
        self.assertEqual("".join(padded_src), "one")
        self.assertEqual("".join(padded_tr), "abc")

    def test_equal_counts_passthrough_and_zip_is_lossless(self):
        src = ["a", "b"]
        tr = ["x", "y"]
        padded_src, padded_tr = _pad_equal_parts(src, tr)

        self.assertEqual(padded_src, src)
        self.assertEqual(padded_tr, tr)
        # the exact operation that used to drop text must now be lossless
        self.assertEqual(list(zip(padded_src, padded_tr)), [("a", "x"), ("b", "y")])


class SplitPairToFitTests(unittest.TestCase):
    def test_equal_counts_and_stream_preservation(self):
        src = "Hello world, this is a quite long source line to split up."
        tr = "这是一个非常长的目标文本，需要被分割成多个部分以满足字幕长度限制。"
        src_parts, tr_parts = split_pair_to_fit(src, tr, max_len=12, multiplier=1.75)

        self.assertEqual(len(src_parts), len(tr_parts))
        self.assertEqual("".join(src_parts), src)
        self.assertEqual("".join(tr_parts), tr)
        self.assertGreater(len(src_parts), 1)

    def test_short_pair_stays_single_part(self):
        src_parts, tr_parts = split_pair_to_fit("hi", "你好", max_len=20, multiplier=1.75)
        self.assertEqual(src_parts, ["hi"])
        self.assertEqual(tr_parts, ["你好"])


class AlignSubsTests(unittest.TestCase):
    """align_subs' multi-line LLM success path.

    align_subs splits the LLM source line on '\\n' (src_parts) and maps the
    returned target parts onto it, so a successful multi-part split must be
    able to pass the stream-preservation asserts. Regression guard: before the
    newline-normalized comparison, `"".join(src_parts) == src_part` could never
    hold for a multi-line split (join() strips the '\\n'), making the assert
    fail and silently routing every line to the deterministic fallback.

    ask_gpt, get_align_prompt, load_key and get_joiner_safe are stubbed so the
    test pins align_subs' own invariant logic only, with no network/config.
    """

    @staticmethod
    def _stub_align(canned_align):
        return (
            mock.patch("core._5_split_sub.ask_gpt", return_value=canned_align),
            mock.patch("core._5_split_sub.get_align_prompt", return_value="align prompt"),
            mock.patch("core._5_split_sub.load_key", return_value="zh"),
            mock.patch("core._5_split_sub.get_joiner_safe", return_value=" "),
        )

    def _run_align(self, canned_align, src_sub, tr_sub, src_part):
        patches = self._stub_align(canned_align)
        for patch in patches:
            patch.start()
        try:
            return align_subs(src_sub, tr_sub, src_part)
        finally:
            for patch in patches:
                patch.stop()

    def test_multi_line_llm_split_success_path_preserves_both_streams(self):
        src_sub = "Hello world. Goodbye moon."
        tr_sub = "你好世界。再见月亮。"
        # split_sentence-style multi-line output: one line per source part
        src_part = "Hello world.\nGoodbye moon."
        canned = {
            "align": [
                {"target_part_1": "你好世界。"},
                {"target_part_2": "再见月亮。"},
            ]
        }

        src_parts, tr_parts, tr_remerged = self._run_align(canned, src_sub, tr_sub, src_part)

        # no exception: the streaming asserts must pass on the multi-line path
        self.assertEqual(len(src_parts), len(tr_parts))
        self.assertEqual(src_parts, ["Hello world.", "Goodbye moon."])
        self.assertEqual(tr_parts, ["你好世界。", "再见月亮。"])
        # newline-normalized stream preserved on BOTH sides
        self.assertEqual("".join(src_parts), src_part.replace("\n", ""))
        self.assertEqual("".join(tr_parts), "你好世界。再见月亮。")
        # remerged = joiner-joined real (unpadded) parts
        self.assertEqual(tr_remerged, "你好世界。 再见月亮。")

    def test_multi_line_split_with_mismatched_part_count_pads_not_truncates(self):
        # The LLM returns ONE target part for a two-line source split: the
        # shorter target side must be padded with '' (step-6 skip), never the
        # source truncated, and both streams still line up under the
        # normalized invariant (no exception).
        src_part = "Hello world.\nGoodbye moon."
        canned = {"align": [{"target_part_1": "你好世界。再见月亮。"}]}

        src_parts, tr_parts, tr_remerged = self._run_align(
            canned, "Hello world. Goodbye moon.", "你好世界。再见月亮。", src_part
        )

        self.assertEqual(len(src_parts), len(tr_parts))
        self.assertEqual(src_parts, ["Hello world.", "Goodbye moon."])
        self.assertEqual(tr_parts, ["你好世界。再见月亮。", ""])
        self.assertEqual("".join(src_parts), src_part.replace("\n", ""))
        self.assertEqual("".join(tr_parts), "你好世界。再见月亮。")
        # '' padding never wraps a stray joiner separator into the remerged text
        self.assertEqual(tr_remerged, "你好世界。再见月亮。")


if __name__ == "__main__":
    unittest.main()