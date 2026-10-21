# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
"""A small made up vocabulary and posteriors builder, so the tests need no model and no network."""

import numpy as np

# id order matters: 0 is the CTC blank (<pad>)
VOCAB = {
    "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
    "a": 4, "i": 5, "k": 6, "t": 7, "r": 8, "ɹ": 9,  # ɹ is the IPA r
    "aɪ": 10,  # a + small capital I, a diphthong
    "ʃ": 11, "tʃ": 12,  # sh and ch
    "e": 13, "o": 14, "n": 15, "s": 16, "ə": 17,  # ə is schwa
    "b": 18, "p": 19, "l": 20, "d": 21, "ɡ": 22, "m": 23, "h": 24,
}
SIZE = max(VOCAB.values()) + 1


def posteriors(frames, spans=(), blank_p=0.9, hit_p=0.97, per_label=3, seed=0):
    """Log posteriors (frames, SIZE) for a clip of mostly blank frames.

    spans: [(first_frame, [label ids])] or [(first_frame, [label ids], hit_p)]. Each label holds
    its own frame for per_label frames (a spoken phone lasts several 20 ms frames) and takes
    probability hit_p there; the blank takes the rest. With hit_p below one half the blank wins
    the frame, which is what weak evidence looks like to the scorer: the ratio is taken against the
    best label, so a label that is the best label scores 0 however confident the model is."""
    rng = np.random.default_rng(seed)
    p = np.full((frames, SIZE), (1.0 - blank_p) / (SIZE - 1))
    p[:, 0] = blank_p
    for span in spans:
        first, labels = span[0], span[1]
        hp = span[2] if len(span) > 2 else hit_p
        for k, lab in enumerate(labels):
            row = np.full(SIZE, 1e-4)
            row[lab] = hp
            row[0] = 1.0 - hp - 1e-4 * (SIZE - 2)
            p[first + k * per_label: first + (k + 1) * per_label] = row
    p *= 1.0 + 0.001 * rng.random(p.shape)  # tiny jitter so ties are not exact
    p /= p.sum(axis=1, keepdims=True)
    return np.log(p).astype(np.float32)


def ids(*labels):
    return [VOCAB[x] for x in labels]
