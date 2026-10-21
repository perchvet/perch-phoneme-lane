# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phoneme_lane.lane import Lane, Null
from phoneme_lane.lexicon import Lexicon
from tests.fakes import VOCAB, ids, posteriors

SH = "ʃ"
NAMES = ["Bakot", "Melin", "Pashi"]
BAKOT = ids("b", "a", "k", "o", "t")
PASHI = ids("p", "a", SH, "i")


def lexicon():
    return Lexicon.from_bench_json({
        "names": {
            "Bakot": {"en-us": {"ipa": "", "tokens": ["b", "a", "k", "o", "t"]}},
            "Melin": {"en-us": {"ipa": "", "tokens": ["m", "e", "l", "i", "n"]}},
            "Pashi": {"en-us": {"ipa": "", "tokens": ["p", "a", SH, "i"]}},
        }
    })


class LaneTest(unittest.TestCase):
    def test_the_spoken_name_ranks_first_with_its_span_in_seconds(self):
        lane = Lane(lexicon(), VOCAB, fold=0)
        scores = lane.run(posteriors(80, spans=[(20, BAKOT), (50, PASHI, 0.3)]))
        top = scores.hits()
        self.assertEqual([h.name for h in top[:2]], ["Bakot", "Pashi"])
        self.assertEqual(top[0].raw, 0.0)  # float32 rounding noise around 0 is not reported
        # the span sits inside the frames where the word is spoken, give or take the length of one phone
        self.assertTrue(20 <= top[0].start_frame <= 22 and 32 <= top[0].end_frame <= 34)
        # frames are 20 ms and the end of a span is the end of its last frame
        self.assertAlmostEqual(top[0].start, top[0].start_frame * 0.02)
        self.assertAlmostEqual(top[0].end, (top[0].end_frame + 1) * 0.02)
        self.assertEqual(top[0].heard, "b a k o t")
        # Pashi is weak evidence: the blank wins its frames, so the greedy path does not hear it ...
        self.assertEqual(scores.heard_string(), "b a k o t")
        # ... but the lane still finds it, with a worse ratio
        self.assertLess(top[1].raw, top[0].raw - 0.2)

    def test_top_limits_the_list(self):
        scores = Lane(lexicon(), VOCAB, fold=0).run(posteriors(80, spans=[(20, BAKOT), (50, PASHI, 0.3)]))
        self.assertEqual(len(scores.hits(top=2)), 2)
        self.assertEqual(len(scores.hits()), 3)

    def test_a_name_with_no_stretch_long_enough_has_no_span_and_still_serialises(self):
        scores = Lane(lexicon(), VOCAB, fold=0).run(posteriors(3))
        self.assertTrue(all(h.raw is None and h.start is None for h in scores.hits()))
        record = scores.to_record("tiny")
        json.dumps(record, allow_nan=False)  # no NaN or infinity leaks into the output
        self.assertEqual(record["frames"], 3)

    def test_broad_class_fold_hears_the_same_name(self):
        scores = Lane(lexicon(), VOCAB, fold=1).run(posteriors(80, spans=[(20, BAKOT), (50, PASHI, 0.3)]))
        self.assertEqual(scores.hits()[0].name, "Bakot")

    def test_a_lexicon_with_nothing_long_enough_is_refused(self):
        tiny = Lexicon.from_bench_json({"names": {"Ab": {"en-us": {"ipa": "", "tokens": ["a", "b"]}}}})
        with self.assertRaises(ValueError):
            Lane(tiny, VOCAB, fold=0)

    def test_names_without_a_scorable_pronunciation_are_listed(self):
        mixed = Lexicon.from_bench_json({"names": {
            "Bakot": {"en-us": {"ipa": "", "tokens": ["b", "a", "k", "o", "t"]}},
            "Ab": {"en-us": {"ipa": "", "tokens": ["a", "b"]}},
        }})
        self.assertEqual(Lane(mixed, VOCAB, fold=0).unscorable, ["Ab"])

    def test_extra_pronunciations_can_be_switched_off(self):
        lex = Lexicon.from_bench_json({"names": {"Bakot": {
            "en-us": {"ipa": "", "tokens": ["b", "a", "k", "o", "t"]},
            "llm0": {"ipa": "", "tokens": ["b", "a", "k", "o"]},
        }}})
        self.assertEqual(len(Lane(lex, VOCAB, fold=0, use_extra=True).owner), 2)
        self.assertEqual(len(Lane(lex, VOCAB, fold=0, use_extra=False).owner), 1)

    def test_posteriors_of_the_wrong_width_are_refused(self):
        with self.assertRaises(ValueError):
            Lane(lexicon(), VOCAB, fold=1).run(np.zeros((10, 7), np.float32))


class NullTest(unittest.TestCase):
    RAW = np.array([[-0.50, -0.40, -0.60], [-0.60, -0.50, -0.70], [-0.40, -0.30, -0.50], [-0.55, -0.45, -0.65]])

    def test_z_is_the_distance_from_the_median_in_scaled_mad_units(self):
        null = Null.fit(NAMES, self.RAW, fold=0)
        med = float(np.median(self.RAW[:, 0]))
        mad = float(np.median(np.abs(self.RAW[:, 0] - med))) * 1.4826
        z = null.z(NAMES, np.array([-0.10, -0.40, -0.60], np.float32))
        self.assertAlmostEqual(z[0], (-0.10 - med) / mad, places=4)
        self.assertAlmostEqual(z[1], (-0.40 - float(np.median(self.RAW[:, 1]))) / null.stats["Melin"][1], places=4)

    def test_mad_has_a_floor_so_z_stays_finite(self):
        flat = np.full((6, 3), -0.5)
        null = Null.fit(NAMES, flat, fold=0)
        self.assertAlmostEqual(null.stats["Bakot"][1], 1e-3)
        self.assertTrue(np.isfinite(null.z(NAMES, np.array([-0.4, -0.4, -0.4], np.float32))).all())

    def test_clips_where_the_name_is_spoken_can_be_left_out(self):
        raw = np.array([[-0.5], [-0.5], [-0.5], [-0.5], [0.0]])
        exclude = np.array([[False], [False], [False], [False], [True]])
        self.assertAlmostEqual(Null.fit(["Bakot"], raw, exclude, fold=0).stats["Bakot"][0], -0.5)

    def test_clips_without_a_match_are_left_out_of_the_null(self):
        raw = np.array([[-0.5], [-0.5], [-1e9]])
        self.assertAlmostEqual(Null.fit(["Bakot"], raw, fold=0).stats["Bakot"][0], -0.5)

    def test_names_unknown_to_the_null_or_without_a_match_have_no_z(self):
        null = Null.fit(["Bakot"], self.RAW[:, :1], fold=0)
        z = null.z(["Bakot", "Melin"], np.array([-0.3, -0.3], np.float32))
        self.assertFalse(np.isnan(z[0]))
        self.assertTrue(np.isnan(z[1]))
        self.assertTrue(np.isnan(null.z(["Bakot"], np.array([-1e9], np.float32))[0]))

    def test_ranking_by_z_can_differ_from_ranking_by_raw(self):
        lane = Lane(lexicon(), VOCAB, fold=0)
        scores = lane.run(posteriors(80, spans=[(20, BAKOT), (50, PASHI, 0.3)]))
        raw = scores.raw.astype(float)
        # a null under which Pashi's raw score is far above its usual level and Bakot's is not
        null = Null({"Bakot": (-3.0, 1.0), "Melin": (-3.0, 1.0), "Pashi": (float(raw[2]) - 10.0, 0.1)}, fold=0)
        self.assertEqual(scores.hits()[0].name, "Bakot")
        self.assertEqual(scores.hits(null)[0].name, "Pashi")

    def test_save_load_round_trip_and_settings_check(self):
        lane = Lane(lexicon(), VOCAB, fold=0)
        null = Null.fit(NAMES, self.RAW, fold=0, fpp=1.5, use_extra=True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "null.json"
            null.save(path)
            back = Null.load(path)
        self.assertEqual(back.stats.keys(), null.stats.keys())
        for name in NAMES:
            self.assertAlmostEqual(back.stats[name][0], null.stats[name][0], places=5)
        back.check(lane)  # same settings: no error
        with self.assertRaises(ValueError):
            Null.fit(NAMES, self.RAW, fold=1).check(lane)
        with self.assertRaises(ValueError):
            Null.fit(NAMES, self.RAW, fold=0, fpp=2.0).check(lane)
        with self.assertRaises(ValueError):
            Null.fit(NAMES, self.RAW, fold=0, use_extra=False).check(lane)

    def test_a_file_that_is_not_a_null_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "other.json"
            path.write_text(json.dumps({"names": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                Null.load(path)


if __name__ == "__main__":
    unittest.main()
