# Copyright 2026 Talcoe LLC. Licensed under the Apache License, Version 2.0.
"""Phoneme log posteriors from a multilingual wav2vec2 CTC model.

The model is facebook/wav2vec2-xlsr-53-espeak-cv-ft (Apache 2.0): 392 espeak phone labels,
one frame every 20 ms. It is loaded with Wav2Vec2FeatureExtractor and AutoModelForCTC and never
with Wav2Vec2Processor, because the processor builds a phoneme tokenizer that needs espeak wired
in before it will load. The lane needs only the vocabulary file, which load_vocab() reads directly.

Posteriors are stored as float16 .npy arrays of shape (frames, 392): the log softmax of the logits.
torch, transformers, soundfile and librosa are imported inside the functions that need them, so
importing this module costs nothing and scoring stored posteriors does not need torch.
"""

import json
import warnings
from pathlib import Path

import numpy as np

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
SAMPLE_RATE = 16000
FRAME_SECONDS = 0.02  # one frame per 320 samples at 16 kHz
BLANK_ID = 0  # <pad> is the CTC blank in this vocabulary
MIN_SAMPLES = 400  # the first convolution needs at least 25 ms of audio
WARN_SECONDS = 120.0  # attention memory grows with the square of the clip length


def load_vocab(model_id=MODEL_ID):
    """Return the model vocabulary as {token: id}.

    Read from vocab.json in the Hugging Face cache; the file is downloaded on first use."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, "vocab.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pick_device(device="auto"):
    """'auto' means cuda when torch can see a GPU and cpu otherwise."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch cannot see a GPU")
    return device


def read_audio(path):
    """Read an audio file as 16 kHz mono float32. Other rates are resampled, channels averaged."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wav


class PhonemeModel:
    """The phoneme recognizer, ready to turn audio into log posteriors."""

    def __init__(self, model_id=MODEL_ID, device="auto", threads=None):
        import torch
        from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor

        if threads:
            torch.set_num_threads(int(threads))
        self._torch = torch
        self.device = pick_device(device)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).eval().to(self.device)

    def log_posteriors(self, wav):
        """wav: 16 kHz mono float32. Returns a float16 array (frames, 392): log softmax of the logits."""
        wav = np.asarray(wav, dtype=np.float32)
        if wav.shape[0] < MIN_SAMPLES:
            raise ValueError("audio is shorter than 25 ms, the model cannot frame it")
        if wav.shape[0] > WARN_SECONDS * SAMPLE_RATE:
            warnings.warn(
                f"clip is {wav.shape[0] / SAMPLE_RATE:.0f} s long; the model attends over the whole clip "
                "and memory grows with the square of the length, so split long recordings"
            )
        x = self.extractor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(self.device)
        with self._torch.inference_mode():
            logits = self.model(x).logits[0]
        return self._torch.log_softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float16)

    def file_posteriors(self, path):
        return self.log_posteriors(read_audio(path))


def save_posteriors(path, logp):
    """Store posteriors as float16 .npy, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(logp).astype(np.float16))


def load_posteriors(path):
    """Read a stored posteriors file as float32 (frames, 392)."""
    return np.load(path).astype(np.float32)
