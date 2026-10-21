# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
#
# Vendored from the Perch bench file scribe/bench/ctc_kws.py (Talcoe LLC): the constants
# BLANK and NEG, the helper extended() and the class Batch. The code below is copied line
# for line. Only the Batch docstring is reworded, because the original was written for a
# character CTC model at 40 ms per frame and this lane scores phoneme labels at 20 ms per
# frame.
#
# What it computes. Every candidate label sequence is scored against the posteriors of a
# clip with a free start, free end Viterbi over the CTC extended sequence. The result is a
# per frame log likelihood ratio against the unconstrained best path over the same span.
# The ratio is at most 0, it is 0 when the sequence is what the model wanted to say there,
# and it gets more negative the harder the lattice has to be bent to fit the sequence.
"""CTC keyword scorer: free start, free end Viterbi over a batch of label sequences."""

import numpy as np

BLANK = 0
NEG = np.float32(-1e30)


def extended(seq):
    """CTC extended sequence: labels separated by optional blanks. The outer
    blanks are dropped on purpose. This is a sub-segment match, so the path may
    begin on the first label and end on the last; requiring leading and trailing
    blanks would only lengthen the span and depress the normalised score."""
    ext = []
    for i, y in enumerate(seq):
        if i:
            ext.append(BLANK)
        ext.append(y)
    return ext


class Batch:
    """All candidate label sequences, padded into one array so a clip is scored
    against the entire lexicon in a single pass over its frames.

    MINIMUM DURATION. Without a duration floor the score is invalid. A short
    sequence can align to a single frame, and if that frame's best label happens
    to be the sequence's one label, the ratio against the free path is exactly 0,
    a perfect score. Nothing was heard; the sequence simply fit in a gap. So each
    variant carries a minimum span in frames (min_frames). The lane passes 1.5
    frames per phoneme, which is 30 ms per phoneme at 20 ms frames. That floor is
    deliberately permissive, well under the time a spoken phoneme really takes,
    because its job is to exclude the degenerate case, not to model speech rate.
    """

    def __init__(self, variants, min_frames=None):
        # variants: list of (row_index_into_names, [token ids])
        self.owner = np.array([o for o, _ in variants], dtype=np.int64)
        self.minspan = (np.ones(len(variants), dtype=np.float32) if min_frames is None
                        else np.asarray(min_frames, dtype=np.float32))
        exts = [extended(s) for _, s in variants]
        S = max(len(e) for e in exts)
        N = len(exts)
        self.ext = np.zeros((N, S), dtype=np.int64)
        self.canskip = np.zeros((N, S), dtype=bool)
        self.last = np.zeros(N, dtype=np.int64)
        for i, e in enumerate(exts):
            self.ext[i, :len(e)] = e
            self.last[i] = len(e) - 1
            # A state may be entered from s-2 only if it is a label whose
            # predecessor label differs; identical neighbours need the blank
            # between them or CTC would collapse the two into one.
            for s in range(2, len(e)):
                if s % 2 == 0 and e[s] != e[s - 2]:
                    self.canskip[i, s] = True
        self.N, self.S = N, S

    def score(self, logp):
        """Viterbi with free start and free end. Returns, per variant, the
        best normalised score and the frame span it was found in."""
        T = logp.shape[0]
        N, S = self.N, self.S
        alpha = np.full((N, S), NEG, dtype=np.float32)
        start = np.zeros((N, S), dtype=np.int64)
        free = logp.max(axis=1)
        F = np.concatenate([[np.float32(0)], np.cumsum(free, dtype=np.float32)])
        best = np.full(N, NEG, dtype=np.float32)
        best_s = np.zeros(N, dtype=np.int64)
        best_e = np.zeros(N, dtype=np.int64)
        idx = np.arange(N)
        for t in range(T):
            emit = logp[t][self.ext]

            prev = alpha.copy()
            ps = start.copy()

            a1 = np.full((N, S), NEG, dtype=np.float32)
            a1[:, 1:] = alpha[:, :-1]
            s1 = np.zeros((N, S), dtype=np.int64)
            s1[:, 1:] = start[:, :-1]
            m = a1 > prev
            prev = np.where(m, a1, prev)
            ps = np.where(m, s1, ps)

            a2 = np.full((N, S), NEG, dtype=np.float32)
            a2[:, 2:] = alpha[:, :-2]
            a2 = np.where(self.canskip, a2, NEG)
            s2 = np.zeros((N, S), dtype=np.int64)
            s2[:, 2:] = start[:, :-2]
            m = a2 > prev
            prev = np.where(m, a2, prev)
            ps = np.where(m, s2, ps)

            # free start: entering the first label at this frame costs nothing
            # before it. Every alpha is <= 0, so beginning here always beats
            # having dwelt in state 0 since earlier; that is correct for
            # Viterbi and it is what makes the search position independent.
            prev[:, 0] = 0.0
            ps[:, 0] = t

            alpha = prev + emit
            start = ps

            fin = alpha[idx, self.last]
            st = start[idx, self.last]
            span = np.maximum(t - st + 1, 1).astype(np.float32)
            norm = (fin - (F[t + 1] - F[st])) / span
            # unreached states carry NEG and a meaningless start frame; without
            # this guard a negative span would flip the sign of a huge negative
            # number and manufacture a perfect score.
            ok = (fin > -1e20) & (st <= t) & (span >= self.minspan) & (norm > best)
            best = np.where(ok, norm, best)
            best_s = np.where(ok, st, best_s)
            best_e = np.where(ok, t, best_e)
        return best, best_s, best_e
