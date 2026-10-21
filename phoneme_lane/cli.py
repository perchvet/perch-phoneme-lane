# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
"""Command line: dump, score, demo.

  python -m phoneme_lane dump  clip1.wav clip2.wav --out logits/
  python -m phoneme_lane score --logits logits/ --names names.txt --out scores.jsonl
  python -m phoneme_lane demo  clip1.wav --names names.txt
"""

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import __version__
from .lane import DEFAULT_FPP, Lane, Null
from .lexicon import Lexicon
from .model import FRAME_SECONDS, PhonemeModel, load_posteriors, load_vocab, save_posteriors

MIN_NULL_CLIPS = 10


def _count(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _utf8_streams():
    """IPA does not survive the default Windows console code page, so write UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _expand(patterns):
    """Expand wildcards that the shell left alone (cmd and PowerShell do not expand them)."""
    out = []
    for p in patterns:
        if not Path(p).exists() and any(ch in p for ch in "*?["):
            out += sorted(glob.glob(p))
        else:
            out.append(p)
    return [Path(p) for p in out]


def _add_lane_args(p):
    p.add_argument("--names", required=True, help="lexicon: a text file with one name per line, a JSON list of names, "
                   "a JSON object {name: [IPA, ...]} of extra pronunciations, or a bench style names.json")
    p.add_argument("--variants", help="JSON object {name: [IPA, ...]} of extra spoken forms added to the lexicon")
    p.add_argument("--null", help="null.json written by score --write-null; adds a z score per name")
    p.add_argument("--fold", type=int, choices=(0, 1, 2), default=1,
                   help="0 exact phones, 1 broad classes (default), 2 consonant skeleton")
    p.add_argument("--fpp", type=float, default=DEFAULT_FPP, help="minimum frames per phoneme (default 1.5, 30 ms)")
    p.add_argument("--no-extra", action="store_true", help="use the espeak pronunciations only, ignore extra ones")


def _check_lane_args(args):
    if args.fpp <= 0:
        raise ValueError("--fpp must be above 0")
    if args.top < 0:
        raise ValueError("--top cannot be negative")


def _build_lane(args):
    _check_lane_args(args)
    vocab = load_vocab()
    lexicon = Lexicon.load(args.names, vocab)
    if args.variants:
        extra = json.loads(Path(args.variants).read_text(encoding="utf-8"))
        if not isinstance(extra, dict) or not all(isinstance(v, list) for v in extra.values()):
            raise ValueError(f"{args.variants}: expected a JSON object {{name: [IPA, ...]}}")
        unknown = lexicon.add_extra(extra, vocab)
        if unknown:
            print(f"note: variants for names that are not in the lexicon were ignored: {', '.join(unknown[:8])}",
                  file=sys.stderr)
    if lexicon.untokenizable:
        dropped = ", ".join(f"{c} x{n}" for c, n in sorted(lexicon.untokenizable.items()))
        print(f"note: some characters in the pronunciations are not in the model vocabulary and were skipped: {dropped}",
              file=sys.stderr)
    lane = Lane(lexicon, vocab, fold=args.fold, fpp=args.fpp, use_extra=not args.no_extra)
    if lane.unscorable:
        print(f"note: {len(lane.unscorable)} names have no pronunciation long enough to score: "
              f"{', '.join(lane.unscorable[:8])}{' ...' if len(lane.unscorable) > 8 else ''}", file=sys.stderr)
    null = None
    if args.null:
        null = Null.load(args.null)
        null.check(lane)
    return lane, null


def cmd_dump(args):
    wavs = _expand(args.wav)
    missing = [str(w) for w in wavs if not w.is_file()]
    if missing:
        raise FileNotFoundError(f"no such audio file: {missing[0]}")
    if not wavs:
        raise FileNotFoundError("no audio files given")
    clash = [s for s, n in Counter(w.stem for w in wavs).items() if n > 1]
    if clash:
        raise ValueError(f"two inputs share the file name {clash[0]!r}; posteriors are stored by file name, rename one")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [w for w in wavs if not (args.skip_existing and (out / f"{w.stem}.npy").exists())]
    model = PhonemeModel(device=args.device, threads=args.threads) if todo else None
    for w in wavs:
        target = out / f"{w.stem}.npy"
        if w not in todo:
            print(f"skipped {w.as_posix()} (already dumped)", flush=True)
            continue
        t0 = time.time()
        try:
            lp = model.file_posteriors(w)
        except ValueError as e:
            raise ValueError(f"{w.as_posix()}: {e}") from e
        save_posteriors(target, lp)
        print(f"dumped {w.as_posix()} -> {target.as_posix()}  {lp.shape[0]} frames ({lp.shape[0] * FRAME_SECONDS:.1f} s of audio) "
              f"in {time.time() - t0:.1f} s on {model.device}", flush=True)


def cmd_score(args):
    files = sorted(Path(args.logits).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no .npy posteriors in {args.logits}")
    lane, null = _build_lane(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for k, path in enumerate(files):
            scores = lane.run(load_posteriors(path))
            f.write(json.dumps(scores.to_record(path.stem, null, args.top), ensure_ascii=False) + "\n")
            raw_rows.append(scores.raw)
            if (k + 1) % 25 == 0:
                print(f"scored {k + 1}/{len(files)}", file=sys.stderr, flush=True)
    print(f"scored {_count(len(files), 'clip')} against {_count(len(lane.names), 'name')} -> {out.as_posix()}")
    if args.write_null:
        fitted = Null.fit(lane.names, np.vstack(raw_rows), fold=lane.fold.mode, fpp=lane.fpp, use_extra=lane.use_extra)
        fitted.save(args.write_null)
        print(f"null from {_count(len(files), 'clip')} -> {args.write_null}")
        if len(files) < MIN_NULL_CLIPS:
            print(f"warning: a null from fewer than {MIN_NULL_CLIPS} clips has an unstable median and spread; "
                  "use many clips that do not contain the names", file=sys.stderr)


def cmd_demo(args):
    lane, null = _build_lane(args)
    src = Path(args.audio)
    if not src.is_file():
        raise FileNotFoundError(f"no such file: {src}")
    if src.suffix.lower() == ".npy":
        logp = load_posteriors(src)
    else:
        model = PhonemeModel(device=args.device, threads=args.threads)
        # round through float16 so the demo and dump plus score agree to the last digit
        logp = model.file_posteriors(src).astype(np.float32)
    scores = lane.run(logp)
    print("HEARD (phonemes):", scores.heard_string())
    hits = scores.hits(null, args.top)
    if null is None:
        print(f"TOP {len(hits)} names by phoneme match (raw ratio, 0 is a perfect fit)")
    else:
        print(f"TOP {len(hits)} names by phoneme match (z against the null, raw ratio beside it)")
    for h in hits:
        if h.raw is None:
            print(f"  {h.name:<22} no span fits in this clip")
            continue
        z = "" if null is None else (f"z={round(h.z, 2) + 0.0:6.2f}  " if h.z is not None else "z=   n/a  ")
        print(f"  {h.name:<22} {z}raw={round(h.raw, 3) + 0.0:6.3f}  {h.start:5.2f}-{h.end:5.2f}s  heard there: {h.heard}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="phoneme_lane",
        description="Score a list of names against the sounds in a recording with a phoneme recognizer.",
    )
    p.add_argument("--version", action="version", version=f"phoneme_lane {__version__}")
    sub = p.add_subparsers(dest="command")

    d = sub.add_parser("dump", help="run the phoneme model on audio files and store the log posteriors")
    d.add_argument("wav", nargs="+", help="audio files (wildcards are expanded); any rate, resampled to 16 kHz mono")
    d.add_argument("--out", required=True, help="folder for the .npy posteriors, one per audio file name")
    d.add_argument("--device", default="auto", help="auto (default), cuda or cpu")
    d.add_argument("--threads", type=int, help="CPU threads for torch (default: torch's own choice)")
    d.add_argument("--skip-existing", action="store_true", help="do not redo files that already have a .npy")
    d.set_defaults(func=cmd_dump)

    s = sub.add_parser("score", help="score stored posteriors against a lexicon, one JSON line per clip")
    s.add_argument("--logits", required=True, help="folder of .npy posteriors written by dump")
    _add_lane_args(s)
    s.add_argument("--out", required=True, help="scores.jsonl to write")
    s.add_argument("--top", type=int, default=20, help="names kept per clip, best first (default 20, 0 keeps all)")
    s.add_argument("--write-null", help="also fit a null from these clips and write it here")
    s.set_defaults(func=cmd_score)

    m = sub.add_parser("demo", help="one clip: the phoneme string the model heard and the top names with spans")
    m.add_argument("audio", help="an audio file, or a .npy written by dump")
    _add_lane_args(m)
    m.add_argument("--top", type=int, default=8, help="names to show (default 8)")
    m.add_argument("--device", default="auto", help="auto (default), cuda or cpu")
    m.add_argument("--threads", type=int, help="CPU threads for torch (default: torch's own choice)")
    m.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    _utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return args.func(args) or 0
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
