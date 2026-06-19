"""Quick manual test of forced alignment against a real lecture clip.

Usage:
    python test_alignment.py            # aligns the first manifest entry
    python test_alignment.py 5          # aligns manifest entry index 5
    python test_alignment.py <file.wav> # aligns the entry whose audioFile matches
"""
import json
import os
import sys

import soundfile as sf

from alignment import align_words, load_aligner

ROOT = "/Users/shubhammishra/TTS"
MANIFEST = os.path.join(ROOT, "lecture_devanagari_manifest.json")


def pick_entry(entries, arg):
    if arg is None:
        return entries[0]
    if arg.isdigit():
        return entries[int(arg)]
    return next(e for e in entries if arg in e["audioFile"])


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    entries = json.load(open(MANIFEST))
    entry = pick_entry(entries, arg)

    wav_path = os.path.join(ROOT, entry["audioFile"])
    text = entry["text"]
    wav, sr = sf.read(wav_path, dtype="float32")

    print(f"file:  {entry['audioFile']}")
    print(f"sr:    {sr} Hz   duration: {len(wav)/sr:.2f}s")
    print(f"text:  {text}\n")

    aligner = load_aligner(device="cpu")
    words = align_words(aligner, wav, sr, text)

    print(f"\n{len(words)} words aligned:\n")
    print(f"{'start':>7} {'end':>7} {'score':>6}  word")
    print("-" * 40)
    for w in words:
        print(f"{w['start']:7.2f} {w['end']:7.2f} {str(w['score']):>6}  {w['word']}")


if __name__ == "__main__":
    main()
