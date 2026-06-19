"""
alignment.py — forced alignment of KNOWN text against a generated waveform.

We already have the exact text we fed into the TTS model, so we don't transcribe
(Whisper-style ASR) — we *align*. WhisperX wraps a wav2vec2 CTC model and uses the
known transcript to find, per word, the start/end time in the audio. This is more
accurate than re-transcribing and never invents words.

The align model is language-specific. For our Devanagari / Hindi pipeline we use the
Hindi wav2vec2 model (its CTC vocabulary is Devanagari characters), so the phonetic
Devanagari text produced upstream aligns directly.

Loaded once at startup (see server.py). align_words() is cheap relative to generation.
"""

import os

import numpy as np
import torch

# wav2vec2 CTC align models run at 16 kHz. Our TTS waveform is at the model sample rate,
# so we resample to this before aligning.
ALIGN_SR = 16000

# Language code passed to WhisperX. "hi" -> Hindi wav2vec2 (Devanagari vocabulary).
ALIGN_LANGUAGE = os.environ.get("ALIGN_LANGUAGE", "hi")
# Optional override of the HF align model id (defaults to WhisperX's pick for the language).
ALIGN_MODEL = os.environ.get("ALIGN_MODEL", "") or None


class Aligner:
    """Holds the loaded wav2vec2 align model + its metadata."""

    def __init__(self, model, metadata, device: str, language: str):
        self.model = model
        self.metadata = metadata
        self.device = device
        self.language = language


def load_aligner(device: str | None = None, language: str = ALIGN_LANGUAGE) -> Aligner:
    """Load the wav2vec2 forced-alignment model for `language`. Call once at startup."""
    import whisperx

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[align] loading align model lang={language} "
          f"model={ALIGN_MODEL or 'default'} device={device}", flush=True)
    model, metadata = whisperx.load_align_model(
        language_code=language, device=device, model_name=ALIGN_MODEL
    )
    print("[align] align model loaded", flush=True)
    return Aligner(model, metadata, device, language)


def _to_16k_mono(wav, sample_rate: int) -> np.ndarray:
    """Force the TTS waveform to mono float32 @ 16 kHz for the align model."""
    a = np.asarray(wav, dtype=np.float32).reshape(-1)  # mono 1-D
    if sample_rate == ALIGN_SR:
        return a
    import torchaudio.functional as AF
    t = torch.from_numpy(a)
    t = AF.resample(t, orig_freq=sample_rate, new_freq=ALIGN_SR)
    return t.numpy().astype(np.float32)


def align_words(aligner: Aligner, wav, sample_rate: int, text: str) -> list[dict]:
    """Align `text` to `wav`, returning [{word, start, end, score}] in seconds.

    Words whose characters aren't in the align model's vocabulary (e.g. stray Latin
    letters / digits in a Devanagari transcript) can't be timed and are dropped by
    WhisperX, so the returned list may be shorter than the literal word count.
    """
    import whisperx

    text = (text or "").strip()
    if not text:
        return []

    audio = _to_16k_mono(wav, sample_rate)
    duration = len(audio) / ALIGN_SR
    # One segment spanning the whole clip; WhisperX subdivides it into word timings.
    segments = [{"start": 0.0, "end": duration, "text": text}]

    result = whisperx.align(
        segments,
        aligner.model,
        aligner.metadata,
        audio,
        aligner.device,
        return_char_alignments=False,
    )

    out = []
    for w in result.get("word_segments", []):
        # WhisperX may emit a word with no timing if it couldn't be placed; skip those.
        if w.get("start") is None or w.get("end") is None:
            continue
        out.append({
            "word": w["word"],
            "start": round(float(w["start"]), 3),
            "end": round(float(w["end"]), 3),
            "score": round(float(w["score"]), 3) if w.get("score") is not None else None,
        })
    return out
