# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from phoneme_lane.lexicon import Fold, Lexicon, Tokenizer
from tests.fakes import SIZE, VOCAB, ids

R, SH, SCHWA, DIPH = "ɹ", "ʃ", "ə", "aɪ"
STRESS, LONG = "ˈ", "ː"

# what a stand in for espeak says for each name (the real call needs phonemizer and espeak)
FAKE_IPA = {
    "en-us": {"Bakot": STRESS + "bakot", "Tashi": "t" + SH + "a" + SH + "i", "Rakat": R + "akat"},
    "en-gb": {"Bakot": "bakot", "Tashi": "ta" + SH + "i", "Rakat": "rakat"},
}


def fake_espeak(names, language):
    return [FAKE_IPA[language][n] for n in names]


class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer(VOCAB)

    def test_longest_match_wins(self):
        self.assertEqual(self.tok.tokenize("b" + DIPH + "k")[0], ["b", DIPH, "k"])
        self.assertEqual(self.tok.tokenize("t" + SH + "a")[0], ["t" + SH, "a"])

    def test_spaces_and_stress_marks_are_ignored_and_unknown_characters_are_reported(self):
        tokens, bad = self.tok.tokenize(STRESS + "ba k" + LONG + "o")
        self.assertEqual(tokens, ["b", "a", "k", "o"])
        self.assertEqual(bad, [LONG])

    def test_hand_written_ipa_also_ignores_dots_slashes_brackets_and_reads_ascii_g(self):
        tokens, bad = self.tok.tokenize_written("/b.a.g.o/")
        self.assertEqual(tokens, ["b", "a", "ɡ", "o"])
        self.assertEqual(bad, [])


class FoldTest(unittest.TestCase):
    def test_broad_classes_merge_r_variants_and_keep_affricates(self):
        f = Fold(VOCAB, 1)
        self.assertEqual(f.cls(R), "r")
        self.assertEqual(f.cls("r"), "r")
        self.assertEqual(f.cls("t" + SH), "t" + SH)
        self.assertEqual(f.cls(SCHWA), SCHWA)
        self.assertEqual(f.idx["r"], f.idx[f.cls(R)])

    def test_special_tokens_all_fold_into_the_blank_class(self):
        f = Fold(VOCAB, 1)
        self.assertEqual([int(f.of[VOCAB[t]]) for t in ("<pad>", "<s>", "</s>", "<unk>")], [0, 0, 0, 0])
        self.assertEqual(f.names[0], "<pad>")

    def test_consonant_skeleton_turns_every_vowel_into_one_class(self):
        f = Fold(VOCAB, 2)
        self.assertEqual({f.cls(v) for v in ("a", "i", "e", "o", SCHWA)}, {"V"})
        self.assertNotEqual(f.cls("k"), "V")

    def test_exact_mode_is_the_identity(self):
        f = Fold(VOCAB, 0)
        lp = np.log(np.full((3, SIZE), 1.0 / SIZE)).astype(np.float32)
        self.assertTrue(np.array_equal(f.logp(lp), lp))
        self.assertEqual(f.variants(["b", "a", "k"]), [tuple(ids("b", "a", "k"))])

    def test_class_posterior_is_the_sum_of_its_members(self):
        f = Fold(VOCAB, 1)
        p = np.full((1, SIZE), 1e-6)
        p[0, VOCAB["r"]], p[0, VOCAB[R]] = 0.3, 0.2
        p /= p.sum()
        folded = f.logp(np.log(p).astype(np.float32))
        self.assertAlmostEqual(float(np.exp(folded[0, f.idx["r"]])), float(p[0, VOCAB["r"]] + p[0, VOCAB[R]]), places=5)

    def test_posteriors_of_the_wrong_width_are_refused(self):
        with self.assertRaises(ValueError):
            Fold(VOCAB, 1).logp(np.zeros((4, SIZE + 3), np.float32))

    def test_a_diphthong_is_also_offered_split(self):
        f = Fold(VOCAB, 1)
        forms = f.variants(["b", DIPH, "k"])
        self.assertEqual(len(forms), 2)
        self.assertEqual(forms[0], (f.idx["b"], f.idx["a"], f.idx["k"]))  # folded as it is: the diphthong is class a
        self.assertEqual(forms[1], (f.idx["b"], f.idx["a"], f.idx["i"], f.idx["k"]))  # split into a and i


class LexiconTest(unittest.TestCase):
    def build(self, **kw):
        with mock.patch("phoneme_lane.lexicon.espeak_ipa", side_effect=fake_espeak):
            return Lexicon.from_names(["Bakot", "Tashi", "Rakat"], VOCAB, **kw)

    def test_names_get_one_pronunciation_per_espeak_language(self):
        lex = self.build()
        self.assertEqual(lex.names, ["Bakot", "Tashi", "Rakat"])
        self.assertEqual([p.key for p in lex.prons["Bakot"]], ["en-us", "en-gb"])
        self.assertEqual(lex.prons["Bakot"][0].tokens, ["b", "a", "k", "o", "t"])
        self.assertEqual(lex.prons["Tashi"][0].tokens, ["t" + SH, "a", SH, "i"])

    def test_extra_pronunciations_are_added_when_they_are_new_and_long_enough(self):
        lex = self.build(extra={"Bakot": ["bakot", "b.a.g.o.t", "ba"]})
        keys = [p.key for p in lex.prons["Bakot"]]
        self.assertEqual(keys, ["en-us", "en-gb", "extra1"])  # same as espeak is dropped, "ba" is too short
        self.assertEqual(lex.prons["Bakot"][2].tokens, ["b", "a", "ɡ", "o", "t"])

    def test_add_extra_to_a_built_lexicon_numbers_new_keys_after_the_old_ones_and_reports_unknown_names(self):
        lex = self.build(extra={"Bakot": ["b.a.g.o.t"]})
        unknown = lex.add_extra({"Bakot": ["b.a.d.o.t"], "Nobody": ["bakot"]}, VOCAB)
        self.assertEqual(unknown, ["Nobody"])
        self.assertEqual([p.key for p in lex.prons["Bakot"]], ["en-us", "en-gb", "extra0", "extra1"])
        self.assertNotIn("Nobody", lex.prons)

    def test_variants_skip_short_and_duplicate_sequences_and_carry_a_duration_floor(self):
        lex = self.build(extra={"Bakot": ["b.a.g.o.t"]})
        fold = Fold(VOCAB, 0)
        variants, floor = lex.variants(fold, fpp=2.0)
        # en-us and en-gb of Bakot are the same sequence, so Bakot has two sequences: espeak and the extra
        self.assertEqual(sum(1 for i, _ in variants if i == 0), 2)
        self.assertEqual(floor, [2.0 * len(seq) for _, seq in variants])
        only_espeak, _ = lex.variants(fold, use_extra=False)
        self.assertEqual(sum(1 for i, _ in only_espeak if i == 0), 1)

    def test_text_file_ignores_blank_lines_comments_and_repeats(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("phoneme_lane.lexicon.espeak_ipa", side_effect=fake_espeak):
            p = Path(d) / "names.txt"
            p.write_text("# formulary\nBakot\n\n  Tashi  \nBakot\nRakat\n", encoding="utf-8")
            self.assertEqual(Lexicon.load(p, VOCAB).names, ["Bakot", "Tashi", "Rakat"])

    def test_json_list_and_json_object_of_extra_pronunciations(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("phoneme_lane.lexicon.espeak_ipa", side_effect=fake_espeak):
            a, b = Path(d) / "list.json", Path(d) / "extra.json"
            a.write_text(json.dumps(["Bakot", "Tashi"]), encoding="utf-8")
            b.write_text(json.dumps({"Bakot": ["b.a.g.o.t"], "Tashi": []}), encoding="utf-8")
            self.assertEqual(Lexicon.load(a, VOCAB).names, ["Bakot", "Tashi"])
            self.assertEqual([p.key for p in Lexicon.load(b, VOCAB).prons["Bakot"]], ["en-us", "en-gb", "extra0"])

    def test_bench_layout_is_read_as_is_in_gate_then_other_order(self):
        obj = {
            "gate": ["Bakot"], "other": ["Tashi"],
            "names": {
                "Tashi": {"en-us": {"ipa": "x", "tokens": ["t", "a", "i"]}},
                "Bakot": {"en-us": {"ipa": "x", "tokens": ["b", "a", "k", "o", "t"]},
                          "llm0": {"ipa": "y", "tokens": ["b", "a", "k", "o"]}},
            },
        }
        lex = Lexicon.from_bench_json(obj)
        self.assertEqual(lex.names, ["Bakot", "Tashi"])
        self.assertEqual([p.key for p in lex.prons["Bakot"]], ["en-us", "llm0"])
        variants, _ = lex.variants(Fold(VOCAB, 0), use_extra=False)
        self.assertEqual(len(variants), 2)  # Bakot en-us and Tashi; llm0 is an extra
        with self.assertRaises(ValueError):
            Lexicon.from_bench_json({"gate": ["Missing"], "names": {}})

    def test_save_and_load_round_trip(self):
        lex = self.build()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lexicon.json"
            lex.save(p)
            back = Lexicon.load(p, VOCAB)
        self.assertEqual(back.names, lex.names)
        self.assertEqual([q.tokens for q in back.prons["Rakat"]], [q.tokens for q in lex.prons["Rakat"]])


if __name__ == "__main__":
    unittest.main()
