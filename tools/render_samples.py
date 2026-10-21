# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
"""Render the sample sentences with Kokoro (ONNX).

    python tools/render_samples.py --model kokoro-v1.0.onnx --voices voices-v1.0.bin --out samples

Two Hindi code mixed sentences: Hindi in Devanagari with English drug names in Latin script. Each
is rendered in two Kokoro Hindi voices, hf_alpha and hm_omega, and written as <sentence>_<voice>.wav.
One English sentence, collision_proheart.wav, in the voice am_echo. All files are 16 kHz mono 16 bit.

Needs kokoro-onnx, which the lane itself does not use and which is not in requirements.txt:
    pip install kokoro-onnx==0.5.0
The model and voices files are the Kokoro v1.0 ONNX release (Apache 2.0),
https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0

How the text reaches the voice. The sentence is phonemized by espeak with the Hindi rules. The
Latin script words come out with espeak's English pronunciation. espeak marks the switch with the
flags (en) and (hi); Kokoro would read those aloud as phones, so the flags are removed and the
phonemes are passed to Kokoro directly (is_phonemes=True). Only characters in the Kokoro
vocabulary are kept, as Kokoro's own tokenizer does.

Resampling. Kokoro writes 24 kHz. The audio is brought to 16 kHz by cutting the spectrum above
8 kHz with an FFT (numpy only), which needs no extra package.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000
VOICES = ("hf_alpha", "hm_omega")
ENGLISH = {
    "collision_proheart": ("am_echo", "He is such a pro, heart and soul, that the whole clinic trusts him with every case."),
}
SENTENCES = {
    "hi_plan": "ब्रूनो को Apoquel सोलह मिलीग्राम दिन में दो बार दें, और Cerenia एक गोली रोज़ाना तीन दिन तक।",
    "hi_exam": "पेट नरम है, दर्द नहीं, लिम्फ नोड्स नॉर्मल हैं। Heartgard और Bravecto अप टू डेट हैं।",
}


def resample_fft(x, sr_in, sr_out):
    """Band limited resampling by truncating (or zero padding) the spectrum."""
    n_out = int(round(len(x) * sr_out / sr_in))
    spec = np.fft.rfft(x)
    cut = np.zeros(n_out // 2 + 1, dtype=spec.dtype)
    m = min(len(spec), len(cut))
    cut[:m] = spec[:m]
    return np.fft.irfft(cut, n_out) * (n_out / len(x))


def hindi_phonemes(text, vocab):
    import espeakng_loader
    import phonemizer
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    ph = phonemizer.phonemize(
        text, language="hi", backend="espeak", preserve_punctuation=True,
        with_stress=True, language_switch="remove-flags",
    )
    return "".join(c for c in ph if c in vocab).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="kokoro-v1.0.onnx")
    ap.add_argument("--voices", required=True, help="voices-v1.0.bin")
    ap.add_argument("--out", required=True, help="folder for the wav files")
    args = ap.parse_args()

    from kokoro_onnx import Kokoro

    kokoro = Kokoro(args.model, args.voices)
    available = set(kokoro.get_voices())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for sid, text in SENTENCES.items():
        phonemes = hindi_phonemes(text, kokoro.tokenizer.vocab)
        for voice in VOICES:
            if voice not in available:
                print(f"voice {voice} is not in {args.voices}", file=sys.stderr)
                return 1
            audio, sr = kokoro.create(phonemes, voice=voice, speed=1.0, lang="hi", is_phonemes=True)
            audio16 = np.clip(resample_fft(audio.astype(np.float64), sr, TARGET_SR), -1.0, 1.0)
            path = out / f"{sid}_{voice}.wav"
            sf.write(str(path), audio16, TARGET_SR, subtype="PCM_16")
            report.append({"file": path.name, "text": text, "voice": voice, "lang": "hi", "speed": 1.0,
                           "phonemes": phonemes, "seconds": round(len(audio16) / TARGET_SR, 2)})
            print(f"wrote {path}  {len(audio16) / TARGET_SR:.2f} s")
    for sid, (voice, text) in ENGLISH.items():
        audio, sr = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        audio16 = np.clip(resample_fft(audio.astype(np.float64), sr, TARGET_SR), -1.0, 1.0)
        path = out / f"{sid}.wav"
        sf.write(str(path), audio16, TARGET_SR, subtype="PCM_16")
        report.append({"file": path.name, "text": text, "voice": voice, "lang": "en-us", "speed": 1.0,
                       "seconds": round(len(audio16) / TARGET_SR, 2)})
        print(f"wrote {path}  {len(audio16) / TARGET_SR:.2f} s")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
