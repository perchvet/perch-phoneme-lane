# Perch phoneme lane

Scores a list of names directly against the sounds in a recording, so a name can be found even when a speech to text transcript never wrote it. It runs a multilingual phoneme recognizer over the audio and, for every name on your list, finds the stretch of the clip where that name's phones fit best.

It is a nomination source. It reports where a name's sounds are. It does not report that the name was said, and it belongs behind the context rules you already apply to a transcript (see Limits). Everything in this repository was tested on synthetic speech.

Made by Talcoe LLC for [Perch](https://perch.talcoe.com), a veterinary scribe. Code is Apache 2.0.

## Why a phoneme lattice can reach names a transcript lost

A speech to text engine commits to one path through the audio and writes the words on that path. If the path never wrote "Cerenia", nothing that works on the text afterwards, a spell checker or a formulary matcher, can write it back.

A phoneme recognizer keeps more. For every 20 ms frame it holds a probability for each of its 392 output labels. That table is the lattice. The lane asks it a different question: does the phone sequence of this name fit anywhere in the clip? The question can be asked for any name, including names no decoder ever proposed, and the answer comes from the sound, not from how a transcript spelled it.

It holds across scripts too. When a Hindi sentence with an English drug name inside comes back in Devanagari, the Latin name is not in the text. Its sounds are still in the audio. The Hindi example below shows it.

On the English sample, a transcript reads "She's on famitidine and meripitin, the morphine CRI was stopped overnight ... a dose of Entice before rounds" (engine and settings in Method notes). The lane's four best fits are famotidine, morphine, Entyce and maropitant, the four drug names that were spoken. That is one clip, not a measurement.

## Install

Python 3.11 or newer.

```
git clone https://github.com/perchvet/perch-phoneme-lane
cd perch-phoneme-lane
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS and Linux activate with `source .venv/bin/activate`. `requirements.txt` pins the versions we built with. The first command that needs the phoneme model downloads it (1.26 GB) into the Hugging Face cache. The espeak library comes with the `espeakng-loader` package, so there is no separate install. With a CUDA build of torch the lane uses the GPU on its own, and `--device cpu` forces the CPU.

The unit tests need no model and no network:

```
python -m unittest discover -s tests -t .
```

## The three commands

All three run as `python -m phoneme_lane`.

### dump: audio to phoneme posteriors

Runs the phoneme model on audio files and stores the log posteriors as float16 `.npy` files, one per audio file name. Any sample rate is accepted and resampled to 16 kHz mono. Wildcards are expanded, which helps on Windows shells.

```
$ python -m phoneme_lane dump samples/r001.wav --out logits
dumped samples/r001.wav -> logits/r001.npy  832 frames (16.6 s of audio) in 2.4 s on cpu
```

That took 2.4 s on a 20 thread CPU and about 0.2 s on an RTX 2070 SUPER. `--skip-existing` leaves files that already have a `.npy`.

### score: posteriors against a lexicon

Scores every stored clip against every name and writes one JSON line per clip.

```
$ python -m phoneme_lane score --logits logits --names samples/names.txt --out scores.jsonl
note: some characters in the pronunciations are not in the model vocabulary and were skipped: ː x14
scored 1 clip against 265 names -> scores.jsonl
```

The note is expected. For names such as butorphanol and albuterol espeak writes the vowel as `juː`. The model has a label for `ju` but none for the length mark, so the mark is skipped.

The first line of `scores.jsonl`, shortened (the heard string is cut, and 18 of the 20 kept names are left out):

```json
{
 "clip": "r001",
 "frames": 832,
 "seconds": 16.64,
 "heard": "m o n i ŋ h a n d o v ə f ə k e n o t r i ə n e t i ə r o l  ...",
 "scores": [
  {
   "name": "famotidine",
   "raw": 0.0,
   "z": null,
   "start": 7.66,
   "end": 8.28,
   "start_frame": 383,
   "end_frame": 413,
   "heard": "f a m ə t i d i n"
  },
  {
   "name": "morphine",
   "raw": 0.0,
   "z": null,
   "start": 9.38,
   "end": 9.74,
   "start_frame": 469,
   "end_frame": 486,
   "heard": "m o f i n"
  }
 ]
}
```

`--top N` keeps the best N names per clip (default 20, `0` keeps all). `z` is `null` until you give a null (see How the score works).

### demo: one clip, what the model heard

Takes an audio file (or a `.npy` from `dump`), prints the phone string the model heard and the top names with their time spans. This is the sample output we got on the English clip. It was rendered with Kokoro, voice am_echo, and says: "Morning handover for kennel three, an eight year old spayed female Labrador who is one day post op from her G D V surgery. She's on famotidine and maropitant, the morphine C R I was stopped overnight, appetite is still nil, and the plan this morning is to try a dose of Entyce before rounds."

```
$ python -m phoneme_lane demo samples/r001.wav --names samples/names.txt
note: some characters in the pronunciations are not in the model vocabulary and were skipped: ː x14
HEARD (phonemes): m o n i ŋ h a n d o v ə f ə k e n o t r i ə n e t i ə r o l d s p e d f i m e l l a b r ə d o h u i z w a n d e p o s t a p f r a m h ə dʒ i d i v i s ə dʒ ə r i ʃ i z a n f a m ə t i d i n a n d m e r ə p i t ə n t d ə m o f i n s s i a a w a z s t a p t o v ə n a t a p ə t a t i z s t i l n i l a n d ə p l a n d i s m o n i ŋ i z t ə t r a a d o s a v e n t a s b i f o r a n d z
TOP 8 names by phoneme match (raw ratio, 0 is a perfect fit)
  famotidine             raw= 0.000   7.66- 8.28s  heard there: f a m ə t i d i n
  morphine               raw= 0.000   9.38- 9.74s  heard there: m o f i n
  Entyce                 raw= 0.000  15.36-15.74s  heard there: e n t a s
  maropitant             raw= 0.000   8.48- 9.02s  heard there: m e r ə p i t ə n t
  Elura                  raw=-0.312   1.18- 1.90s  heard there: o t r i ə
  atenolol               raw=-0.317   2.04- 2.62s  heard there: e t i ə r o l
  Tylan                  raw=-0.329  12.52-13.40s  heard there: t i l n i l a n
  Atopica                raw=-0.346  11.26-12.12s  heard there: a t a p ə t a
```

How to read it:

- The heard string is the model's best phone at each frame, repeats collapsed and silence dropped. It is written in broad classes (see How the score works), which is why `r` stands for every r sound.
- `raw` is the lane's score for the best stretch of the clip for that name. 0 is a perfect fit. It falls as the lattice has to be bent to fit the name.
- The four spoken names all score 0.000 to three decimals, so nothing separates them and their order is not meaningful. It can change with the device or the lexicon. The next name down, Elura, is at -0.312.
- The two numbers before `s` are the start and the end of the stretch in seconds. The edges are approximate (see Limits).
- `heard there` is the part of the heard string that overlaps the stretch.

## Hindi example: a name inside another language

`samples/hi_plan_hf_alpha.wav` is a Hindi sentence with two English drug names in it, rendered with the Kokoro Hindi voice hf_alpha:

> ब्रूनो को Apoquel सोलह मिलीग्राम दिन में दो बार दें, और Cerenia एक गोली रोज़ाना तीन दिन तक।
> (Give Bruno Apoquel, sixteen milligrams, twice a day, and Cerenia, one tablet daily for three days.)

A speech to text transcript of this clip is written entirely in Devanagari (engine and settings in Method notes):

> ब्रूनो को अपक्वल सोलह मिलीग्राम दिन में दो बार दे और सरेनिय एक गोली रोजाना तीन दिन तक।

Apoquel came out as अपक्वल and Cerenia as सरेनिय. There is no Latin letter in the text, so a rule that looks for "Apoquel" or "Cerenia" in the transcript has nothing to match. The phoneme lane reads the sound instead:

```
$ python -m phoneme_lane demo samples/hi_plan_hf_alpha.wav --names samples/names.txt
note: some characters in the pronunciations are not in the model vocabulary and were skipped: ː x14
HEARD (phonemes): b r u n o k o a p o k w o l s o l a m e l i g r a m d e n m e d o b a r d e o r s e v e n i a e g l i r o z a n a t i n d e n t e k
TOP 8 names by phoneme match (raw ratio, 0 is a perfect fit)
  Unasyn                 raw=-0.230   5.52- 6.08s  heard there: a n a t i n
  Apoquel                raw=-0.290   0.72- 1.20s  heard there: a p o k w o l
  Vetoryl                raw=-0.406   4.24- 5.12s  heard there: v e n i a e g l
  methadone              raw=-0.431   2.64- 3.98s  heard there: m e d o b a r d e o r
  Cerenia                raw=-0.432   4.10- 4.72s  heard there: s e v e n i a e
  Ventolin               raw=-0.457   4.24- 5.30s  heard there: v e n i a e g l i
  Orbax                  raw=-0.464   2.98- 3.90s  heard there: o b a r d e o
  Entyce                 raw=-0.491   4.28- 4.96s  heard there: e n i a e g
```

Both names are in the top 8 of 265, and each stretch sits on its word. Apoquel is second at 0.72 to 1.20 s and the transcript puts the word at 0.58 to 1.14 s. Cerenia is fifth at 4.10 to 4.72 s and the transcript puts it at 3.94 to 4.42 s. Neither is first. Unasyn is first because the Hindi words "rozana tin" (daily, three) are heard as `a n a t i n`, and espeak writes Unasyn as `a n a s i n`, one phone away. This is what a nomination source looks like, and it is why the lane needs something behind it.

All four Hindi clips in `samples/` (two sentences, two voices), with the rank of each spoken name among the 265, the transcript's version of the name, and the phones the model heard there:

| clip | name spoken | transcript wrote | rank of 265 | raw | phones heard there |
|---|---|---|---|---|---|
| hi_plan_hf_alpha | Apoquel | अपक्वल | 2 | -0.290 | a p o k w o l |
| hi_plan_hf_alpha | Cerenia | सरेनिय | 5 | -0.432 | s e v e n i a e |
| hi_plan_hm_omega | Apoquel | अपक्वल | 2 | -0.295 | a p o k w o l |
| hi_plan_hm_omega | Cerenia | सरेनिय | 5 | -0.433 | s a r e n i |
| hi_exam_hf_alpha | Heartgard | हाथकाद | 1 | 0.000 | h a t g a d |
| hi_exam_hf_alpha | Bravecto | ब्रेव तो | 2 | -0.242 | b r e v d o |
| hi_exam_hm_omega | Heartgard | हाथकाद | 1 | 0.000 | h a t g a d |
| hi_exam_hm_omega | Bravecto | ब्रेफ तो | 8 | -0.454 | b r e f t o |

The voices differ in how an English r comes out: the model hears `s e v e n i a e` for Cerenia in one voice and `s a r e n i` in the other. In our bench the same first sentence, read by a Hindi neural voice from another text to speech engine (not included here, this repository ships Kokoro audio only), was heard as `s e r i n i a`, and Cerenia ranked first at a raw ratio of 0.00. Apoquel did not reach the top 8 on that clip: the speaker said "apokel" with no w, and every pronunciation in our bench lexicon had one.

That kind of gap is closed by adding the spoken form to the lexicon, not by changing the model. `samples/variants_example.json` holds one extra form for Cerenia:

```
$ python -m phoneme_lane demo samples/hi_plan_hm_omega.wav --names samples/names.txt --variants samples/variants_example.json
note: some characters in the pronunciations are not in the model vocabulary and were skipped: ː x14
HEARD (phonemes): b r u n o k o a p o k w o l s o l e m e l i g r a m d e n m e d o b a r d e o r s a r e n i e e k g o l i r o z a n a t i n d i n t a k
TOP 8 names by phoneme match (raw ratio, 0 is a perfect fit)
  Cerenia                raw=-0.205   4.50- 4.92s  heard there: s a r e n i e
  Unasyn                 raw=-0.261   6.06- 6.60s  heard there: a n a t i n
  Apoquel                raw=-0.295   0.78- 1.22s  heard there: a p o k w o l
  Orbax                  raw=-0.405   3.22- 4.28s  heard there: o b a r d e o
  toceranib              raw=-0.421   3.68- 5.18s  heard there: e o r s a r e n i e
  Entyce                 raw=-0.439   6.70- 7.10s  heard there: i n t a k
  methadone              raw=-0.442   2.88- 4.36s  heard there: m e d o b a r d e o r
  lotilaner              raw=-0.487   3.68- 5.22s  heard there: e o r s a r e n i e e
```

Cerenia moves from fifth (-0.433) to first (-0.205). We wrote that form after reading what this clip's heard string said, so it shows the mechanism and is not a gain to expect on other speech.

## Lexicon files

`--names` takes any of these:

- A text file, one name per line (`samples/names.txt`). Blank lines and lines that start with `#` are ignored.
- A JSON list of names.
- A JSON object `{"Cerenia": ["sarenia"]}`: names with extra spoken forms written in IPA. Spaces, stress marks, dots, slashes and square brackets are ignored, and the ASCII letter g is read as the IPA ɡ.
- The layout our bench uses, `{"gate": [...], "other": [...], "names": {name: {key: {"ipa": ..., "tokens": [...]}}}}`, with the phone tokens already filled in. It is used as is and needs no phonemizer. Keys other than `en-us` and `en-gb` count as extra forms.

`--variants file.json` adds extra forms, `{name: [IPA, ...]}`, to whichever lexicon you loaded. `--no-extra` ignores every extra form and uses espeak's pronunciations only.

Names are written to phones by espeak (en-us and en-gb) through `phonemizer-fork`, cut into the model's labels by longest match, and each pronunciation becomes one or two label sequences. A name is scored against all of its sequences and keeps the best.

## How the score works

For a name written as the labels y1 to yL, the lane runs a Viterbi search over the CTC extended sequence (the labels with optional blanks between them). The search is free to start and to end anywhere in the clip. The raw ratio is

```
raw = ( log P(best alignment of the name over frames s..e)
        - sum over t in s..e of the best single label's log P at t ) / (e - s + 1)
```

- It is at most 0, up to float rounding. It is 0 when, in every frame of the stretch, the name's label is the model's top choice, however confident the model is. It falls as the lattice has to be bent to fit the name.
- Every sequence must span at least 1.5 frames per label (`--fpp`, 30 ms). Without that floor a short name can fit a single frame and score a perfect 0 for nothing.
- Spans are in 20 ms frames: `start_frame * 0.02` to `(end_frame + 1) * 0.02` seconds.
- `--fold` sets the label alphabet. `1` (default) merges labels into broad classes: every r variant to `r`, theta to `t`, vowel heights merged, and diphthongs and syllabic l also tried split into their parts. `0` keeps the exact phones. `2` keeps only the consonant skeleton and turns every vowel into one class. The model's posteriors are folded the same way, by adding the probabilities of the members of each class.
- `z` is optional. `score --write-null null.json` fits a null from the clips it scores: for each name, the median of its raw ratio and the MAD scaled by 1.4826 (floor 0.001). Pass it back with `--null null.json` to get `z = (raw - median) / MAD`. The null records the fold, `--fpp` and extra form settings it was made with, and the tool refuses to use it with different ones. Fit the null on your own audio, on many clips that mostly do not contain the names. The median and MAD tolerate a few clips where a name is spoken.

As Python:

```python
from phoneme_lane import Lane, Lexicon
from phoneme_lane.model import load_vocab, PhonemeModel

vocab = load_vocab()
lexicon = Lexicon.load("samples/names.txt", vocab)
lane = Lane(lexicon, vocab)                          # fold=1, fpp=1.5

model = PhonemeModel()                               # downloads the phoneme model on first use
logp = model.file_posteriors("samples/r001.wav")     # float16 log posteriors, shape (frames, 392)
scores = lane.run(logp)
for hit in scores.hits(top=4):
    print(hit.name, hit.raw, hit.start, hit.end, hit.heard)
```

## Limits

- **Synthetic speech only.** Every clip we scored, here and in our bench, is text to speech. No recording of a real clinician in a real room has been scored with this lane. Real audio will move the scores. In our bench, moving to a synthetic voice the settings had not been tuned on already did.
- **The voices and the lexicon share a pronunciation source.** Kokoro takes its phones for English words from espeak, the same tool that writes the lexicon's pronunciations. In these clips a name is said the way the lexicon expects. A real speaker will say it differently. The samples flatter the lane, and they are not a test of how people say brand names.
- **Brand names are a pronunciation gap.** Pronunciations come from espeak's English rules plus whatever forms you add. A brand said the way its maker says it, or by a speaker of another English, can miss, as Apoquel did above. espeak also writes Bravecto as `bɹeɪvktoʊ`, with no vowel between v and k. Fixing a gap means adding a form.
- **It nominates, it does not decide.** The lane says these sounds are here. It cannot say that a word was meant. Ordinary English fits drug names. Here is a clip with no drug name in it: "He is such a pro, heart and soul, that the whole clinic trusts him with every case." (`samples/collision_proheart.wav`)

  ```
  $ python -m phoneme_lane demo samples/collision_proheart.wav --names samples/names.txt --top 4
  note: some characters in the pronunciations are not in the model vocabulary and were skipped: ː x14
  HEARD (phonemes): h i i z s a tʃ a p r o h a t t ə n s o l d a t d ə h o l k l i n i k t r a s t s h i m w i d e v r i k e s
  TOP 4 names by phoneme match (raw ratio, 0 is a perfect fit)
    ProHeart               raw= 0.000   0.60- 1.44s  heard there: p r o h a t
    Percorten              raw=-0.247   0.60- 1.62s  heard there: p r o h a t t ə n
    Proin                  raw=-0.276   0.60- 1.62s  heard there: p r o h a t t ə n
    carprofen              raw=-0.299   0.42- 1.62s  heard there: tʃ a p r o h a t t ə n
  ```

  ProHeart scores a perfect 0.000, and the sound really is there. Only context can refuse it: a dose, a species, a span of plain English around it. Put the lane behind the same rules you use for a transcript, and treat a hit as a candidate. In our bench it feeds a nomination step that has those rules after it.
- **The z score needs your own null.** A null built on one set of synthetic voices produced many wrong names on a voice it had not seen, because z alone does not carry across voice sets. Our bench used a floor on the raw ratio together with z, and chose both on a separate corpus. Choose yours on your own audio.
- **Long and multiword names.** In our bench a single 22 phoneme name caused a large share of the wrong names on an unseen voice, at weak raw fits. Score the words of a long name separately, or leave names much longer than about 14 phonemes out of the lexicon.
- **Fits can straddle words.** Unasyn above fits across two Hindi words. The lane has no notion of word boundaries.
- **Span edges are approximate.** The scorer keeps the single best path per state, which tends to start on the last frames of a name's first phone. On the 13 stretches in this README the start was 0.08 to 0.20 s after the start of the same word in a transcript with word times, and the end was within 0.3 s of the word's end (the transcript's word times are approximate too). Expect an error of about one phone at each end.
- **Whole clip attention.** The model attends over the whole clip and memory grows with the square of its length. Split recordings longer than a couple of minutes. `dump` warns above 120 s.

## Samples

`samples/` holds synthetic speech only. Every audio file was rendered with Kokoro, and `samples/samples.json` lists the text, the voice and how each file was made. `tools/render_samples.py` renders the Hindi files and `collision_proheart.wav` again; it needs `pip install kokoro-onnx==0.5.0` and the Kokoro v1.0 ONNX model and voices files. `samples/names.txt` is the 265 name lexicon used in this README.

```
.
├── phoneme_lane/        the package
│   ├── model.py         audio to log posteriors
│   ├── lexicon.py       names to label sequences, and the phone class fold
│   ├── scorer.py        the CTC Viterbi scorer, vendored from the Perch bench
│   ├── lane.py          score a clip against a lexicon, null and z
│   └── cli.py           dump, score, demo
├── samples/             audio, names.txt, samples.json, variants_example.json
├── tools/render_samples.py
└── tests/               unit tests, no model needed
```

## Method notes

The speech recognition models are named here and nowhere else in this README.

- **Phoneme model.** `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (Apache 2.0): wav2vec 2.0 XLSR-53 fine tuned on Common Voice to recognize espeak phone labels in many languages. 392 output labels, one frame per 20 ms. Xu, Baevski and Auli, "Simple and Effective Zero-shot Cross-lingual Phoneme Recognition", 2021, https://arxiv.org/abs/2109.11680. It is loaded with `Wav2Vec2FeatureExtractor` and `AutoModelForCTC`. The processor is not used, because its tokenizer needs espeak wired in before it will load; the lane reads `vocab.json` directly.
- **Posteriors.** The log softmax of the logits, stored as float16. GPU and CPU runs of the same clip differ by at most 0.016 in a stored value and pick the same best label in every frame.
- **Scorer.** `phoneme_lane/scorer.py` is the CTC keyword scorer from the Perch bench, copied line for line. Its source is `scribe/bench/ctc_kws.py`. Only the class docstring is reworded.
- **Pronunciations.** espeak-ng through `phonemizer-fork` and `espeakng-loader`, languages en-us and en-gb, no stress marks.
- **Transcripts quoted above.** Whisper large v3 through faster whisper 1.2.1, on CPU, int8, beam size 5, no voice activity filter, word timestamps on, language detected by the model (Hindi for the Hindi clips, English for `r001.wav` and `collision_proheart.wav`). The word times quoted in the Hindi example and in Limits come from the same runs. In our bench, a hosted Whisper large v3 turbo transcript of the same Hindi sentences, read by another synthetic Hindi voice, was also entirely Devanagari.
- **Speed.** On the machine we built it on (20 CPU threads, RTX 2070 SUPER): `dump` of the 16.6 s clip took 2.4 s on CPU and 0.2 s on the GPU. Scoring it against 265 names took 0.3 s on CPU.

## Licence

Code: Apache 2.0, see `LICENSE` and `NOTICE`. The sample audio, `names.txt` and `samples.json` are data and are CC BY 4.0, see `samples/LICENSE`. The voices are Kokoro, which is Apache 2.0: Kokoro 82M by hexgrad, run through kokoro-onnx. The phoneme model is Apache 2.0 and is downloaded on first use. `phonemizer-fork` (GPL 3) and the espeak-ng library inside `espeakng-loader` (GPL 3 or later) are installed by pip and only write pronunciations of names; they are not part of this repository.

## Citation

See `CITATION.cff`. In short: Talcoe LLC, "Perch phoneme lane: scoring a list of names against the phonemes in a recording", version 0.1.0, 2026. https://github.com/perchvet/perch-phoneme-lane. DOI pending (Zenodo mints it after the first release).

Related Perch releases: [perch-trapset](https://github.com/perchvet/perch-trapset) (clips that catch a scribe inventing drug names), [vet-drug-pronunciations](https://github.com/perchvet/vet-drug-pronunciations) (also on [Hugging Face](https://huggingface.co/datasets/Talcoe/vet-drug-pronunciations)), [perch-vetdictation](https://github.com/perchvet/perch-vetdictation) (also on [Hugging Face](https://huggingface.co/datasets/Talcoe/perch-vetdictation)).

---

Talcoe LLC, makers of Perch. Questions: mo@talcoe.com
