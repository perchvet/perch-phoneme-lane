# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
"""The lane: score a clip's phoneme posteriors against every name in a lexicon.

For each name the lane keeps the best raw ratio over all of the name's pronunciation variants,
with the frame span where that variant fits. The raw ratio is at most 0. It is 0 when the model
wanted to say exactly those phones in that stretch, and it falls as the lattice has to be bent to
fit them. Optionally the raw ratio is standardised against a per name null (median and MAD of the
name's raw ratio over clips that do not contain it), which gives a z score.

Time spans. A frame is 20 ms. A span covers frames start_frame to end_frame inclusive, so in
seconds it runs from start_frame * 0.02 to (end_frame + 1) * 0.02. The edges are approximate:
the scorer keeps the single best path per state, which tends to begin on the last frames of the
first phone, so expect an error of about one phone at each end.

Heard phones. The heard string is the greedy path: the best label at every frame, repeats
collapsed, blanks dropped. The phones "heard there" for a hit are the runs of that path that
overlap the hit's span.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lexicon import Fold
from .model import FRAME_SECONDS
from .scorer import Batch

NO_MATCH = np.float32(-1e9)  # raw score of a name that has no valid span in the clip
MATCHED = -1e8  # raw scores above this are real
MAD_SCALE = 1.4826  # makes the MAD comparable to a standard deviation
MAD_FLOOR = 1e-3  # a name whose raw ratio never varies still gets a finite z
DEFAULT_FPP = 1.5  # minimum frames per phoneme: 30 ms at 20 ms frames


@dataclass
class Hit:
    name: str
    raw: float | None
    z: float | None
    start_frame: int | None
    end_frame: int | None  # inclusive
    start: float | None  # seconds
    end: float | None  # seconds, the end of the last frame in the span
    heard: str  # the phones the greedy path put inside the span

    def to_dict(self):
        return {
            "name": self.name,
            "raw": None if self.raw is None else round(self.raw, 4) + 0.0,  # + 0.0 turns -0.0 into 0.0
            "z": None if self.z is None else round(self.z, 2) + 0.0,
            "start": None if self.start is None else round(self.start, 2),
            "end": None if self.end is None else round(self.end, 2),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "heard": self.heard,
        }


class Null:
    """Per name median and MAD of the raw ratio, measured on clips that do not contain the name.

    The settings the scores were made with (fold, fpp, extra pronunciations) are stored with the
    null, because a z score is only meaningful against a null made the same way."""

    def __init__(self, stats, fold=1, fpp=DEFAULT_FPP, use_extra=True, clips=0):
        self.stats = stats  # {name: (median, mad)}
        self.fold = int(fold)
        self.fpp = float(fpp)
        self.use_extra = bool(use_extra)
        self.clips = int(clips)

    @classmethod
    def fit(cls, names, raw, exclude=None, **settings):
        """raw: (clips, names) matrix of raw ratios. exclude: optional boolean matrix, True where
        the name is known to be spoken in the clip, left out of that name's null."""
        raw = np.asarray(raw, dtype=np.float64)
        stats = {}
        for j, name in enumerate(names):
            ok = raw[:, j] > MATCHED
            if exclude is not None:
                ok &= ~np.asarray(exclude, dtype=bool)[:, j]
            values = raw[ok, j]
            if values.size == 0:
                continue
            med = float(np.median(values))
            mad = max(float(np.median(np.abs(values - med))) * MAD_SCALE, MAD_FLOOR)
            stats[name] = (med, mad)
        return cls(stats, clips=raw.shape[0], **settings)

    def z(self, names, raw):
        """z score per name, NaN for a name without a null entry or without a match."""
        out = np.full(len(names), np.nan)
        for j, name in enumerate(names):
            if name in self.stats and raw[j] > MATCHED:
                med, mad = self.stats[name]
                out[j] = (float(raw[j]) - med) / mad
        return out

    def check(self, lane):
        """Refuse a null made with different settings than the lane that is about to use it."""
        diffs = []
        if self.fold != lane.fold.mode:
            diffs.append(f"fold {self.fold} in the null, {lane.fold.mode} in use")
        if abs(self.fpp - lane.fpp) > 1e-9:
            diffs.append(f"fpp {self.fpp} in the null, {lane.fpp} in use")
        if self.use_extra != lane.use_extra:
            diffs.append(f"extra pronunciations {self.use_extra} in the null, {lane.use_extra} in use")
        if diffs:
            raise ValueError("the null was made with different settings: " + "; ".join(diffs))

    def to_dict(self):
        return {
            "format": "perch-phoneme-lane-null/1",
            "fold": self.fold,
            "fpp": self.fpp,
            "extra": self.use_extra,
            "clips": self.clips,
            "names": {n: {"median": round(m, 6), "mad": round(d, 6)} for n, (m, d) in self.stats.items()},
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if not str(obj.get("format", "")).startswith("perch-phoneme-lane-null"):
            raise ValueError(f"{path}: not a null file written by this tool")
        stats = {n: (float(v["median"]), float(v["mad"])) for n, v in obj["names"].items()}
        return cls(stats, obj["fold"], obj["fpp"], obj["extra"], obj.get("clips", 0))


class ClipScores:
    """The result of scoring one clip: one raw ratio and one frame span per name."""

    def __init__(self, names, raw, start, end, runs, frames):
        self.names = names
        self.raw = raw  # (names,) float32, NO_MATCH where the name has no valid span
        self.start = start  # (names,) first frame of the best span
        self.end = end  # (names,) last frame of the best span, inclusive
        self.runs = runs  # [(first frame, last frame, symbol)] of the greedy path, blanks dropped
        self.frames = frames

    @property
    def heard(self):
        """[(first frame, symbol)] of the greedy path."""
        return [(a, sym) for a, _, sym in self.runs]

    @property
    def seconds(self):
        return self.frames * FRAME_SECONDS

    def heard_string(self):
        return " ".join(sym for _, _, sym in self.runs)

    def heard_between(self, start_frame, end_frame):
        """The phones of the greedy path that overlap frames start_frame to end_frame."""
        return " ".join(sym for a, b, sym in self.runs if a <= end_frame and b >= start_frame)

    def hits(self, null=None, top=None):
        """Names ranked best first: by z when a null is given, by raw ratio otherwise."""
        z = null.z(self.names, self.raw) if null is not None else None
        key = np.asarray(z if z is not None else self.raw, dtype=np.float64)
        key = np.where(np.isnan(key) | (self.raw <= MATCHED), -np.inf, key)
        order = np.argsort(-key, kind="stable")
        if top:
            order = order[:top]
        out = []
        for j in order:
            if self.raw[j] <= MATCHED:
                out.append(Hit(self.names[j], None, None, None, None, None, None, ""))
                continue
            s, e = int(self.start[j]), int(self.end[j])
            zj = None if z is None or np.isnan(z[j]) else float(z[j])
            # the ratio cannot be above 0; a few 1e-7 either side of 0 is float32 rounding, not signal
            raw = round(float(self.raw[j]), 6) + 0.0
            out.append(
                Hit(
                    self.names[j], raw, zj, s, e,
                    s * FRAME_SECONDS, (e + 1) * FRAME_SECONDS, self.heard_between(s, e),
                )
            )
        return out

    def to_record(self, clip, null=None, top=None):
        """One JSON serialisable record for a clip, the line written by the score command."""
        return {
            "clip": clip,
            "frames": self.frames,
            "seconds": round(self.seconds, 2),
            "heard": self.heard_string(),
            "scores": [h.to_dict() for h in self.hits(null, top)],
        }


class Lane:
    """A lexicon compiled for scoring, with the settings that produced it."""

    def __init__(self, lexicon, vocab, fold=1, fpp=DEFAULT_FPP, use_extra=True):
        self.lexicon = lexicon
        self.names = lexicon.names
        self.fold = Fold(vocab, fold)
        self.fpp = float(fpp)
        self.use_extra = bool(use_extra)
        variants, min_frames = lexicon.variants(self.fold, self.fpp, self.use_extra)
        if not variants:
            raise ValueError("the lexicon has no pronunciation long enough to score")
        self.batch = Batch(variants, min_frames=min_frames)
        self.owner = self.batch.owner
        scored = set(self.owner.tolist())
        self.unscorable = [n for i, n in enumerate(self.names) if i not in scored]

    def run(self, logp):
        """Score one clip. logp: (frames, 392) log posteriors from the model, any float type."""
        lp = self.fold.logp(logp)
        best, best_start, best_end = self.batch.score(lp)
        n = len(self.names)
        raw = np.full(n, NO_MATCH, np.float32)
        start = np.zeros(n, np.int64)
        end = np.zeros(n, np.int64)
        for v in range(len(self.owner)):
            o = self.owner[v]
            if best[v] > raw[o]:
                raw[o], start[o], end[o] = best[v], best_start[v], best_end[v]
        am = lp.argmax(1)
        first = np.flatnonzero(np.diff(am, prepend=-1) != 0)  # first frame of every run of the same label
        last = np.append(first[1:] - 1, lp.shape[0] - 1)
        runs = [(int(a), int(b), self.fold.names[am[a]]) for a, b in zip(first, last) if am[a] != 0]
        return ClipScores(self.names, raw, start, end, runs, lp.shape[0])
