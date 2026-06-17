"""
benchmark_rtf.py — Measure the Real-Time Factor (RTF) of VoxCPM2 + LoRA
generation on the current GPU (here: the E2E T4). Run this BEFORE serving so
you know the per-request latency the T4 will give you.

    RTF = generation_time / audio_duration

RTF < 1.0 = faster than real-time (good); lower is better. On a T4 expect RTF
noticeably higher than on a 4090. This script:
  - loads the model once (same loader the server uses),
  - runs a discarded warm-up generation (lazy CUDA init / kernel caching),
  - times N runs across texts of varying length with torch.cuda.synchronize()
    so async GPU work is actually finished,
  - records per-run audio duration, generation time, RTF and peak VRAM,
  - prints a summary (mean / median RTF) and appends every run to a CSV.

Note: RTF here is pure model generation. MP3/WAV encoding adds only a few ms
per clip and is not part of the number.

Usage (on the box, with .env already filled or assets in /workspace/assets):
    python benchmark_rtf.py
    python benchmark_rtf.py --runs 5 --timesteps 10 --cfg 2.0
    python benchmark_rtf.py --lora-dir /workspace/assets/lora_only --step 100
"""

import argparse
import csv
import os
import statistics
import time

import torch

from model_loader import load_model

# A spread of lengths so RTF is reported across short / medium / long inputs.
DEFAULT_TEXTS = [
    "नमस्ते, आज का लेसन शुरू करते हैं।",
    "आज हम डिफरेंशिएशन पढ़ेंगे, जो कैलकुलस का एक बहुत इम्पोर्टेंट टॉपिक है।",
    (
        "आज हम डिटरमिनेंट्स पढ़ेंगे, जिनका यूज़ मैट्रिक्स इक्वेशन्स और "
        "लीनियर सिस्टम्स को सॉल्व करने के लिए व्यापक रूप से होता है।"
    ),
    (
        "बच्चों, फोटोसिंथेसिस वो प्रोसेस है जिसमें प्लांट्स सनलाइट को यूज़ करके "
        "कार्बन डाइऑक्साइड और वॉटर से ग्लूकोज़ बनाते हैं, और इस रिएक्शन में "
        "ऑक्सीजन एक बायप्रोडक्ट के रूप में रिलीज़ होती है, जो हमारे लिए ज़रूरी है।"
    ),
]


def gpu_name(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def time_one(model, text, reference, cfg, timesteps, device):
    """Generate once and return (audio_seconds, gen_seconds, peak_vram_mb)."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sync(device)
    t0 = time.perf_counter()
    wav = model.generate(
        text=text,
        reference_wav_path=reference,
        cfg_value=cfg,
        inference_timesteps=timesteps,
    )
    sync(device)
    gen_seconds = time.perf_counter() - t0

    audio_seconds = len(wav) / model.tts_model.sample_rate
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
    )
    return audio_seconds, gen_seconds, peak_vram_mb


def main():
    p = argparse.ArgumentParser(description="VoxCPM2 + LoRA RTF benchmark (E2E T4)")
    p.add_argument("--lora-dir", default=os.environ.get("LORA_DIR", "/workspace/assets/lora_only"),
                   help="Directory containing step_XXXXXXX/ checkpoints")
    p.add_argument("--step", type=int, default=int(os.environ.get("LORA_STEP", 100)),
                   help="Checkpoint step to benchmark (default: 100)")
    p.add_argument("--reference", default=os.environ.get("REFERENCE_WAV", "/workspace/assets/reference.wav"),
                   help="Reference speaker wav")
    p.add_argument("--runs", type=int, default=3, help="Measured runs per text (default: 3)")
    p.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance (default: 2.0)")
    p.add_argument("--timesteps", type=int, default=10, help="Diffusion timesteps (default: 10)")
    p.add_argument("--no-denoiser", action="store_true", help="Skip loading the denoiser")
    p.add_argument("--csv", default="rtf_log.csv", help="CSV file to append results to")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = gpu_name(device)
    print(f"[device] {device} ({name})")
    if device == "cpu":
        print("[warn] No CUDA GPU detected — RTF will be far above 1.0 and not "
              "representative of the T4 target.")

    model = load_model(args.lora_dir, args.step, load_denoiser=not args.no_denoiser)

    # Warm-up (discarded): first generation pays one-time init costs.
    print("[warmup] running one discarded generation...")
    time_one(model, DEFAULT_TEXTS[0], args.reference, args.cfg, args.timesteps, device)

    rows = []
    rtfs = []
    print(f"\n{'text_len':>8}  {'audio_s':>8}  {'gen_s':>8}  {'RTF':>7}  {'VRAM_MB':>8}")
    print("-" * 48)
    for text in DEFAULT_TEXTS:
        for _ in range(args.runs):
            audio_s, gen_s, vram = time_one(
                model, text, args.reference, args.cfg, args.timesteps, device
            )
            rtf = gen_s / audio_s if audio_s > 0 else float("nan")
            rtfs.append(rtf)
            print(f"{len(text):>8}  {audio_s:>8.2f}  {gen_s:>8.2f}  {rtf:>7.3f}  {vram:>8.0f}")
            rows.append({
                "gpu": name,
                "device": device,
                "step": args.step,
                "cfg": args.cfg,
                "timesteps": args.timesteps,
                "text_len": len(text),
                "audio_s": round(audio_s, 4),
                "gen_s": round(gen_s, 4),
                "rtf": round(rtf, 5),
                "peak_vram_mb": round(vram, 1),
            })

    mean_rtf = statistics.mean(rtfs)
    median_rtf = statistics.median(rtfs)
    print("-" * 48)
    print(f"[summary] runs={len(rtfs)}  mean RTF={mean_rtf:.3f}  median RTF={median_rtf:.3f}")
    if mean_rtf > 0:
        faster = mean_rtf < 1
        print(f"[summary] {'faster' if faster else 'SLOWER'} than real-time "
              f"({1/mean_rtf:.2f}x {'real-time' if faster else 'i.e. takes longer than the audio'})")
    print(f"[summary] peak VRAM ~{max(r['peak_vram_mb'] for r in rows):.0f} MB")

    # Append to CSV (write header only if the file is new).
    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] appended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
