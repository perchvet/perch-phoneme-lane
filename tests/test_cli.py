# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
"""Command line plumbing, run without the model: score and demo on stored posteriors, and the
errors dump reports before it loads anything."""

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from phoneme_lane import cli
from tests.fakes import VOCAB, ids, posteriors

LEX = {
    "gate": ["Bakot", "Melin"], "other": [],
    "names": {
        "Bakot": {"en-us": {"ipa": "", "tokens": ["b", "a", "k", "o", "t"]}},
        "Melin": {"en-us": {"ipa": "", "tokens": ["m", "e", "l", "i", "n"]}},
    },
}


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
            mock.patch("phoneme_lane.cli.load_vocab", return_value=VOCAB):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.lex = self.dir / "names.json"
        self.lex.write_text(json.dumps(LEX), encoding="utf-8")
        self.logits = self.dir / "logits"
        self.logits.mkdir()
        bakot, melin = ids("b", "a", "k", "o", "t"), ids("m", "e", "l", "i", "n")
        # each clip holds both names: one spoken cleanly, the other with weak evidence
        clips = {
            "clip_a": [(10, bakot), (50, melin, 0.3)],
            "clip_b": [(10, melin), (50, bakot, 0.3)],
        }
        for i, (name, spans) in enumerate(clips.items()):
            np.save(self.logits / f"{name}.npy", posteriors(90, spans=spans, seed=i).astype(np.float16))

    def test_score_writes_one_json_line_per_clip_best_name_first(self):
        out = self.dir / "scores.jsonl"
        code, stdout, _ = run(["score", "--logits", str(self.logits), "--names", str(self.lex),
                               "--out", str(out), "--fold", "0", "--top", "1"])
        self.assertEqual(code, 0)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["clip"] for r in rows], ["clip_a", "clip_b"])
        self.assertEqual([r["scores"][0]["name"] for r in rows], ["Bakot", "Melin"])
        self.assertTrue(all(len(r["scores"]) == 1 for r in rows))
        first = rows[0]["scores"][0]
        self.assertTrue(10 <= first["start_frame"] <= 12 and 22 <= first["end_frame"] <= 24)
        self.assertAlmostEqual(first["end"], round((first["end_frame"] + 1) * 0.02, 2))
        self.assertIn("scored 2 clips against 2 names", stdout)

    def test_top_zero_keeps_every_name(self):
        out = self.dir / "all.jsonl"
        run(["score", "--logits", str(self.logits), "--names", str(self.lex), "--out", str(out), "--fold", "0", "--top", "0"])
        self.assertEqual(len(json.loads(out.read_text(encoding="utf-8").splitlines()[0])["scores"]), 2)

    def test_write_null_then_use_it_gives_z_scores(self):
        out, null = self.dir / "s.jsonl", self.dir / "null.json"
        code, _, err = run(["score", "--logits", str(self.logits), "--names", str(self.lex), "--out", str(out),
                            "--fold", "0", "--write-null", str(null)])
        self.assertEqual(code, 0)
        self.assertIn("fewer than 10 clips", err)  # two clips is not a usable null, and the tool says so
        self.assertTrue(json.loads(null.read_text(encoding="utf-8"))["names"])
        out2 = self.dir / "s2.jsonl"
        code, _, _ = run(["score", "--logits", str(self.logits), "--names", str(self.lex), "--out", str(out2),
                          "--fold", "0", "--null", str(null)])
        self.assertEqual(code, 0)
        first = json.loads(out2.read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(all(s["z"] is not None for s in first["scores"]))

    def test_a_null_made_with_other_settings_is_an_error(self):
        out, null = self.dir / "s.jsonl", self.dir / "null.json"
        run(["score", "--logits", str(self.logits), "--names", str(self.lex), "--out", str(out), "--fold", "0",
             "--write-null", str(null)])
        code, _, err = run(["score", "--logits", str(self.logits), "--names", str(self.lex), "--out", str(out),
                            "--fold", "1", "--null", str(null)])
        self.assertEqual(code, 1)
        self.assertIn("different settings", err)

    def test_demo_prints_the_heard_string_and_the_top_name_from_stored_posteriors(self):
        code, stdout, _ = run(["demo", str(self.logits / "clip_a.npy"), "--names", str(self.lex), "--fold", "0", "--top", "2"])
        self.assertEqual(code, 0)
        lines = stdout.splitlines()
        self.assertEqual(lines[0], "HEARD (phonemes): b a k o t")  # the second name is weak evidence, not heard
        self.assertTrue(lines[2].lstrip().startswith("Bakot"))
        start, end = re.search(r"(\d+\.\d+)-\s*(\d+\.\d+)s", lines[2]).groups()
        self.assertTrue(0.20 <= float(start) <= 0.24 and 0.46 <= float(end) <= 0.50)  # inside frames 10 to 24
        self.assertTrue(lines[2].endswith("heard there: b a k o t"), lines[2])

    def test_variants_file_adds_spoken_forms_and_names_it_does_not_know_are_reported(self):
        variants = self.dir / "variants.json"
        variants.write_text(json.dumps({"Bakot": ["b.a.g.o.t"], "Nobody": ["bakot"]}), encoding="utf-8")
        out = self.dir / "v.jsonl"
        code, _, err = run(["score", "--logits", str(self.logits), "--names", str(self.lex), "--out", str(out),
                            "--fold", "0", "--variants", str(variants)])
        self.assertEqual(code, 0)
        self.assertIn("ignored: Nobody", err)

    def test_variants_file_of_the_wrong_shape_is_an_error(self):
        variants = self.dir / "variants.json"
        variants.write_text(json.dumps(["Bakot"]), encoding="utf-8")
        code, _, err = run(["score", "--logits", str(self.logits), "--names", str(self.lex),
                            "--out", str(self.dir / "v.jsonl"), "--variants", str(variants)])
        self.assertEqual(code, 1)
        self.assertIn("expected a JSON object", err)

    def test_a_negative_top_and_a_zero_fpp_are_refused(self):
        base = ["score", "--logits", str(self.logits), "--names", str(self.lex), "--out", str(self.dir / "x.jsonl")]
        code, _, err = run(base + ["--top", "-1"])
        self.assertEqual((code, "--top cannot be negative" in err), (1, True))
        code, _, err = run(base + ["--fpp", "0"])
        self.assertEqual((code, "--fpp must be above 0" in err), (1, True))

    def test_score_without_posteriors_is_an_error_not_a_traceback(self):
        empty = self.dir / "empty"
        empty.mkdir()
        code, _, err = run(["score", "--logits", str(empty), "--names", str(self.lex), "--out", str(self.dir / "x.jsonl")])
        self.assertEqual(code, 1)
        self.assertIn("no .npy posteriors", err)

    def test_dump_refuses_two_inputs_with_the_same_file_name_before_loading_the_model(self):
        one, two = self.dir / "a" / "clip.wav", self.dir / "b" / "clip.wav"
        for p in (one, two):
            p.parent.mkdir()
            sf.write(str(p), np.zeros(1600, np.float32), 16000)
        code, _, err = run(["dump", str(one), str(two), "--out", str(self.dir / "o")])
        self.assertEqual(code, 1)
        self.assertIn("share the file name", err)

    def test_dump_reports_a_missing_audio_file(self):
        code, _, err = run(["dump", str(self.dir / "nope.wav"), "--out", str(self.dir / "o")])
        self.assertEqual(code, 1)
        self.assertIn("no such audio file", err)

    def test_no_subcommand_prints_help_and_exits_2(self):
        code, stdout, _ = run([])
        self.assertEqual(code, 2)
        self.assertIn("dump", stdout)

    def test_wildcards_are_expanded_when_the_shell_leaves_them_alone(self):
        for n in ("x1.wav", "x2.wav"):
            sf.write(str(self.dir / n), np.zeros(1600, np.float32), 16000)
        found = cli._expand([str(self.dir / "x*.wav")])
        self.assertEqual([p.name for p in found], ["x1.wav", "x2.wav"])


if __name__ == "__main__":
    unittest.main()
