# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
import unittest

import numpy as np

from phoneme_lane.scorer import BLANK, NEG, Batch, extended
from tests.fakes import ids, posteriors


class ExtendedTest(unittest.TestCase):
    def test_labels_are_separated_by_blanks_and_outer_blanks_are_dropped(self):
        self.assertEqual(extended([5, 6, 7]), [5, BLANK, 6, BLANK, 7])
        self.assertEqual(extended([5]), [5])


class BatchTest(unittest.TestCase):
    def test_a_sequence_spoken_in_the_clip_scores_zero_on_its_exact_span(self):
        word = ids("b", "a", "k", "o", "t")
        lp = posteriors(60, spans=[(20, word)])  # 3 frames per label: frames 20 to 34
        best, start, end = Batch([(0, word)], min_frames=[1.5 * len(word)]).score(lp)
        self.assertAlmostEqual(float(best[0]), 0.0, places=5)
        # the span sits inside the frames where the word is spoken, give or take the length of one phone
        self.assertTrue(20 <= int(start[0]) <= 22 and 32 <= int(end[0]) <= 34, (int(start[0]), int(end[0])))

    def test_a_sequence_that_was_not_spoken_scores_far_below_zero(self):
        spoken = ids("b", "a", "k", "o", "t")
        other = ids("m", "e", "l", "i", "n")
        lp = posteriors(60, spans=[(20, spoken)])
        best, _, _ = Batch([(0, other)], min_frames=[7.5]).score(lp)
        self.assertLess(float(best[0]), -1.0)

    def test_duration_floor_rejects_a_fit_that_is_too_fast(self):
        word = ids("b", "a", "k")
        lp = posteriors(40, spans=[(10, word)], per_label=1)  # one frame per label
        fast, s, e = Batch([(0, word)]).score(lp)  # without a floor the fit is perfect
        self.assertAlmostEqual(float(fast[0]), 0.0, places=5)
        self.assertEqual((int(s[0]), int(e[0])), (10, 12))
        held, _, _ = Batch([(0, word)], min_frames=[4.5]).score(lp)  # 1.5 frames per label
        self.assertLess(float(held[0]), -0.2)  # the floor forces a longer, worse fit

    def test_a_clip_too_short_for_the_sequence_has_no_valid_span(self):
        lp = posteriors(2)
        best, _, _ = Batch([(0, ids("b", "a", "k"))], min_frames=[4.5]).score(lp)
        self.assertEqual(float(best[0]), float(NEG))

    def test_identical_neighbour_labels_need_a_blank_between_them(self):
        t3 = ids("t", "t", "t")
        # one long run of t does not say "t t t": CTC would collapse it into a single t
        run = posteriors(40, spans=[(10, ids("t", "t", "t", "t"))])
        collapsed, _, _ = Batch([(0, t3)], min_frames=[4.5]).score(run)
        self.assertLess(float(collapsed[0]), -0.2)
        # three separate t sounds with a gap between them do say it
        gaps = posteriors(60, spans=[(10, ids("t")), (16, ids("t")), (22, ids("t"))])
        spoken, _, _ = Batch([(0, t3)], min_frames=[4.5]).score(gaps)
        self.assertAlmostEqual(float(spoken[0]), 0.0, places=4)

    def test_every_variant_in_the_batch_gets_its_own_answer(self):
        a, b = ids("b", "a", "k", "o", "t"), ids("m", "e", "l", "i", "n")
        lp = posteriors(90, spans=[(10, a), (50, b)])
        best, start, end = Batch([(0, a), (1, b)], min_frames=[7.5, 7.5]).score(lp)
        (s0, s1), (e0, e1) = [int(x) for x in start], [int(x) for x in end]
        self.assertTrue(10 <= s0 <= 12 and 22 <= e0 <= 24, (s0, e0))
        self.assertTrue(50 <= s1 <= 52 and 62 <= e1 <= 64, (s1, e1))
        self.assertTrue(np.all(best > -1e-4))


if __name__ == "__main__":
    unittest.main()
