# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
"""Lexicon: names, their pronunciations, and the label sequences the scorer searches for.

A name becomes one or more label sequences in four steps.

1. Pronunciations. espeak (through phonemizer and espeakng_loader) writes each name in IPA for
   en-us and en-gb. Extra pronunciations can be supplied by hand, also in IPA.
2. Tokens. The IPA string is cut into the model's phone labels by longest match against the
   model vocabulary. Characters the vocabulary cannot carry are dropped and counted.
3. Fold. mode 0 keeps the exact phone labels. mode 1 merges them into broad classes (all r
   variants to r, theta to t, vowel heights merged, and so on). mode 2 keeps only the consonant
   skeleton (every vowel becomes one class). The model's posteriors are folded the same way, by
   summing the probabilities of the members of each class, so the scorer runs unchanged.
4. Variants. In the folded modes each pronunciation yields two sequences: the tokens as they
   are, and a second form with diphthongs and syllabic l split into their parts, which is how
   an accented or syllable timed speaker says them.

The fold tables are carried over from the Perch bench file fold.py, and the tokenizer and the
phonemizer call from the bench file names.py.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_KEYS = ("en-us", "en-gb")  # pronunciation keys written by espeak
MIN_LABELS = 3  # shorter label sequences are never scored

# Fold tables, carried over from the bench file fold.py.
LEAD_MARKS = ('ˈ', 'ˌ', 'ː', '̃', '̪')  # a label that starts with one of these is classed by its next character
AFFRICATE_SECOND = ('ʃ', 'ʒ', 'ɕ', 'ʑ', 's', 'z')
VOWELS = set('aeiouəäøœũɨɯɵʉ')
MAP = {'ɹ': 'r', 'ɾ': 'r', 'ʁ': 'r', 'ɽ': 'r', 'θ': 't', 'ð': 'd', 'ɐ': 'a', 'ʌ': 'a', 'ɑ': 'a', 'æ': 'a',
       'ɛ': 'e', 'ɪ': 'i', 'ᵻ': 'i', 'y': 'i', 'ɔ': 'o', 'ɒ': 'o', 'ʊ': 'u', 'ɚ': 'ə', 'ɝ': 'ə', 'ɜ': 'ə',
       'ɡ': 'g', 'ɲ': 'n', 'ɱ': 'm', 'ʋ': 'v', 'β': 'v', 'x': 'k', 'ɣ': 'g', 'q': 'k', 'ʔ': '', 'ʰ': ''}

# Diphthongs and syllabic l, offered split as a second form of the same pronunciation.
SPLIT = {'əl': ['ə', 'l'], 'aɪ': ['a', 'i'], 'ɔɪ': ['o', 'i'], 'aʊ': ['a', 'u'], 'oɪ': ['o', 'i'], 'ju': ['j', 'u'],
         'aɪə': ['a', 'i', 'ə'], 'aɪɚ': ['a', 'i', 'ə'], 'iə': ['i', 'ə'], 'eɪ': ['e', 'i'], 'oʊ': ['o', 'u'],
         'ɑːɹ': ['a', 'r'], 'ɔːɹ': ['o', 'r'], 'oːɹ': ['o', 'r'], 'ɛɹ': ['e', 'r'], 'ɪɹ': ['i', 'r'], 'ʊɹ': ['u', 'r']}


class Fold:
    """Maps the model's phone labels onto classes. mode 0 is the identity."""

    def __init__(self, vocab, mode=1):
        if mode not in (0, 1, 2):
            raise ValueError("fold mode must be 0, 1 or 2")
        self.mode = mode
        self.vocab = vocab
        size = max(vocab.values()) + 1
        if mode == 0:
            inverse = {i: t for t, i in vocab.items()}
            self.names = [inverse.get(i, "?") for i in range(size)]
            self.of = np.arange(size, dtype=np.int64)
            self.idx = dict(vocab)
        else:
            names = sorted({self.cls(t) for t in vocab if not t.startswith("<")})
            self.names = ["<pad>"] + [n for n in names if n != "<pad>"]
            self.idx = {n: i for i, n in enumerate(self.names)}
            self.of = np.zeros(size, np.int64)
            for t, i in vocab.items():
                # special tokens (pad, bos, eos, unk) all fold into the blank class 0
                self.of[i] = self.idx.get(self.cls(t), 0) if not t.startswith("<") or t == "<pad>" else 0
        self._members = [np.flatnonzero(self.of == c) for c in range(len(self.names))]

    def cls(self, tok):
        """Class name of one phone label (modes 1 and 2)."""
        if tok.startswith("<"):
            return tok
        c = tok[0]
        if c in LEAD_MARKS:
            c = tok[1] if len(tok) > 1 else c
        c = MAP.get(c, c)
        if self.mode == 2 and c in VOWELS:
            c = "V"
        if c in ("t", "d") and len(tok) > 1 and tok[1] in AFFRICATE_SECOND:  # affricates stay
            return tok[:2]
        return c or "?"

    def logp(self, lp):
        """Fold log posteriors (frames, vocab) into class log posteriors (frames, classes)."""
        lp = np.asarray(lp, dtype=np.float32)
        if lp.ndim != 2 or lp.shape[1] != len(self.of):
            raise ValueError(f"posteriors have shape {lp.shape}, expected (frames, {len(self.of)})")
        if self.mode == 0:
            return lp
        out = np.full((lp.shape[0], len(self.names)), -1e9, np.float32)
        for c, members in enumerate(self._members):
            if members.size:
                out[:, c] = np.logaddexp.reduce(lp[:, members], axis=1)
        return out

    def _ids(self, tokens):
        if self.mode == 0:
            return [self.vocab[t] for t in tokens if t in self.vocab]
        out = []
        for t in tokens:
            c = self.cls(t)
            if c in self.idx:
                out.append(self.idx[c])
        return out

    def variants(self, tokens):
        """Label id sequences for one pronunciation: the tokens as they are and, when different,
        the form with diphthongs and syllabic l split into their parts (modes 1 and 2)."""
        asis = self._ids(tokens)
        if self.mode == 0:
            return [tuple(asis)]
        split = []
        for t in tokens:
            for u in SPLIT.get(t, [t]):
                split += self._ids([u])
        out = [tuple(asis)]
        if tuple(split) != tuple(asis):
            out.append(tuple(split))
        return out


class Tokenizer:
    """Cuts an IPA string into the model's phone labels by longest match."""

    def __init__(self, vocab):
        self.tokens = sorted((k for k in vocab if not k.startswith("<")), key=len, reverse=True)

    def tokenize(self, ipa):
        """Returns (labels, dropped characters). Spaces and stress marks are ignored."""
        ipa = ipa.replace(" ", "").replace("ˈ", "").replace("ˌ", "")
        out, bad, i = [], [], 0
        while i < len(ipa):
            for t in self.tokens:
                if ipa.startswith(t, i):
                    out.append(t)
                    i += len(t)
                    break
            else:
                bad.append(ipa[i])
                i += 1
        return out, bad

    def tokenize_written(self, ipa):
        """Same, for IPA typed by hand: also ignores syllable dots, slashes and brackets, and
        reads the ASCII letter g as the IPA letter ɡ."""
        ipa = re.sub(r"[ .ˈˌ/\[\]]", "", ipa).replace("g", "ɡ")
        return self.tokenize(ipa)


@dataclass
class Pronunciation:
    key: str  # 'en-us', 'en-gb', or the key of an extra pronunciation
    ipa: str
    tokens: list


def wire_espeak():
    """Point phonemizer at the espeak-ng library and data that espeakng_loader ships."""
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError as e:
        raise ImportError(
            "phonemizing names needs phonemizer-fork and espeakng-loader (see requirements.txt)"
        ) from e
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


_wired = False


def espeak_ipa(names, language):
    """IPA for each name from espeak, exactly as the bench file names.py calls it."""
    global _wired
    if not _wired:
        wire_espeak()
        _wired = True
    from phonemizer import phonemize

    return phonemize(list(names), language=language, backend="espeak", strip=True, with_stress=False, njobs=1)


class Lexicon:
    """Names in a fixed order, each with its pronunciations."""

    def __init__(self, names, prons, untokenizable=None):
        self.names = list(names)
        self.prons = prons  # {name: [Pronunciation]}
        self.untokenizable = dict(untokenizable or {})  # dropped character -> count

    def __len__(self):
        return len(self.names)

    @classmethod
    def from_names(cls, names, vocab, languages=BASE_KEYS, extra=None):
        """Phonemize plain names with espeak. extra is {name: [IPA string, ...]} of added forms."""
        names = list(dict.fromkeys(n.strip() for n in names if n and n.strip()))
        tok = Tokenizer(vocab)
        prons = {n: [] for n in names}
        bad = Counter()
        for lang in languages:
            for name, ipa in zip(names, espeak_ipa(names, lang)):
                tokens, dropped = tok.tokenize(ipa)
                bad.update(dropped)
                prons[name].append(Pronunciation(lang, ipa, tokens))
        lexicon = cls(names, prons, bad)
        if extra:
            lexicon.add_extra(extra, vocab)
        return lexicon

    def add_extra(self, extra, vocab):
        """Add hand written pronunciations, {name: [IPA string, ...]}, to names already in the lexicon.

        A form is skipped when it is too short or gives the same labels as a pronunciation the name
        already has. Returns the names in extra that are not in the lexicon."""
        tok = Tokenizer(vocab)
        for name, forms in extra.items():
            if name not in self.prons:
                continue
            have = self.prons[name]
            seen = {tuple(p.tokens) for p in have}
            offset = sum(1 for p in have if p.key.startswith("extra"))
            for i, ipa in enumerate(forms):
                tokens, dropped = tok.tokenize_written(ipa)
                for c in dropped:
                    self.untokenizable[c] = self.untokenizable.get(c, 0) + 1
                if len(tokens) >= MIN_LABELS and tuple(tokens) not in seen:
                    have.append(Pronunciation(f"extra{offset + i}", ipa, tokens))
                    seen.add(tuple(tokens))
        return [n for n in extra if n not in self.prons]

    @classmethod
    def from_bench_json(cls, obj):
        """The bench layout: {'gate': [...], 'other': [...], 'names': {name: {key: {'ipa', 'tokens'}}}}.
        Names are gate then other. Without those two lists, the keys of 'names' in file order."""
        listed = list(obj.get("gate", [])) + list(obj.get("other", []))
        order = list(dict.fromkeys(listed)) if listed else list(obj["names"])
        missing = [n for n in order if n not in obj["names"]]
        if missing:
            raise ValueError(f"names without pronunciations in the lexicon file: {missing[:5]}")
        prons = {
            n: [Pronunciation(k, v.get("ipa", ""), list(v["tokens"])) for k, v in obj["names"][n].items()]
            for n in order
        }
        return cls(order, prons)

    @classmethod
    def load(cls, path, vocab, languages=BASE_KEYS):
        """Read a lexicon file. Accepted layouts:

        * text file, one name per line (blank lines and lines starting with # are ignored)
        * JSON list of names
        * JSON object {name: [IPA, ...]}: names with extra pronunciations written by hand
        * JSON object in the bench layout (see from_bench_json), used as is
        """
        path = Path(path)
        if path.suffix.lower() != ".json":
            lines = path.read_text(encoding="utf-8").splitlines()
            return cls.from_names([ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")], vocab, languages)
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return cls.from_names([str(n) for n in obj], vocab, languages)
        if isinstance(obj, dict) and isinstance(obj.get("names"), dict):
            return cls.from_bench_json(obj)
        if isinstance(obj, dict) and all(isinstance(v, list) for v in obj.values()):
            return cls.from_names(list(obj), vocab, languages, extra=obj)
        raise ValueError(f"{path}: not a recognised lexicon layout")

    def to_dict(self):
        """The bench layout, so a lexicon built from names can be saved and reloaded."""
        return {
            "names": {
                n: {p.key: {"ipa": p.ipa, "tokens": p.tokens} for p in self.prons[n]} for n in self.names
            }
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=0), encoding="utf-8")

    def variants(self, fold, fpp=1.5, use_extra=True):
        """The label sequences to score: ([(name index, [label ids])], [minimum frames]).

        A sequence needs at least MIN_LABELS labels and fpp frames per label. Duplicate sequences of one
        name are dropped. With use_extra False only the espeak pronunciations are used."""
        variants, min_frames = [], []
        for i, name in enumerate(self.names):
            seen = set()
            for p in self.prons[name]:
                if not use_extra and p.key not in BASE_KEYS:
                    continue
                for ids in fold.variants(p.tokens):
                    if len(ids) < MIN_LABELS or ids in seen:
                        continue
                    seen.add(ids)
                    variants.append((i, list(ids)))
                    min_frames.append(fpp * len(ids))
        return variants, min_frames
