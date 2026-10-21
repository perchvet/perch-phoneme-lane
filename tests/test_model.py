# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
"""The guards in model.py that run before any network is touched. The model itself is not loaded."""

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from phoneme_lane import model


def bare_model():
    """A PhonemeModel without weights: the feature extractor stops the call after the guards."""
    m = object.__new__(model.PhonemeModel)
    m.device = "cpu"
    m._torch = None
    m.extractor = mock.Mock(side_effect=RuntimeError("stop after the guards"))
    return m


class GuardTest(unittest.TestCase):
    def test_audio_shorter_than_25_ms_is_refused(self):
        with self.assertRaises(ValueError):
            bare_model().log_posteriors(np.zeros(100, np.float32))

    def test_a_long_clip_warns_before_the_model_runs(self):
        with warnings.catch_warnings(record=True) as caught, self.assertRaises(RuntimeError):
            warnings.simplefilter("always")
            bare_model().log_posteriors(np.zeros(int(121 * model.SAMPLE_RATE), np.float32))
        self.assertTrue(any("split long recordings" in str(w.message) for w in caught))

    def test_a_normal_clip_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught, self.assertRaises(RuntimeError):
            warnings.simplefilter("always")
            bare_model().log_posteriors(np.zeros(2 * model.SAMPLE_RATE, np.float32))
        self.assertEqual(caught, [])


class AudioAndStorageTest(unittest.TestCase):
    def test_stereo_is_averaged_and_other_rates_are_resampled_to_16_khz(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "stereo.wav"
            t = np.arange(24000) / 24000.0
            left, right = np.sin(2 * np.pi * 440 * t), np.zeros_like(t)
            sf.write(str(path), np.stack([left, right], axis=1).astype(np.float32), 24000)
            wav = model.read_audio(path)
        self.assertEqual(wav.dtype, np.float32)
        self.assertEqual(wav.ndim, 1)
        self.assertAlmostEqual(len(wav) / model.SAMPLE_RATE, 1.0, places=2)
        self.assertAlmostEqual(float(np.abs(wav).max()), 0.5, delta=0.05)  # two channels averaged: half amplitude

    def test_posteriors_round_trip_as_float16_and_create_the_folder(self):
        lp = np.log(np.full((5, 392), 1.0 / 392)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "new" / "folder" / "clip.npy"
            model.save_posteriors(path, lp)
            self.assertEqual(np.load(path).dtype, np.float16)
            back = model.load_posteriors(path)
        self.assertEqual(back.dtype, np.float32)
        self.assertTrue(np.allclose(back, lp, atol=2e-2))


if __name__ == "__main__":
    unittest.main()
