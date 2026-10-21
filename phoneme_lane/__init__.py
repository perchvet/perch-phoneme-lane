# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
"""Perch phoneme lane: score a list of names against the sounds in a recording.

A multilingual phoneme recognizer turns audio into a probability for every phone at every 20 ms
frame. The lane asks, for each name on a list, how well that name's phoneme sequence fits some
stretch of those probabilities, so a name can be found even when a transcript never wrote it.

Modules
    model    audio to phoneme log posteriors (torch, transformers)
    lexicon  names to pronunciations to label sequences, and the phone class fold
    scorer   the CTC Viterbi keyword scorer, vendored from the Perch bench
    lane     scoring a clip against a lexicon, the null and the z score
    cli      python -m phoneme_lane dump | score | demo
"""

__version__ = "0.1.0"

from .lane import ClipScores, Hit, Lane, Null  # noqa: E402
from .lexicon import Fold, Lexicon  # noqa: E402

__all__ = ["ClipScores", "Fold", "Hit", "Lane", "Lexicon", "Null", "__version__"]
