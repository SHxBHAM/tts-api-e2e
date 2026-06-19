"""
handler.py — RunPod Serverless (queue-based) worker for VoxCPM2 + Hinglish LoRA TTS.

Reuses the same model_loader / audio_encode / alignment modules as the FastAPI server.
The model loads ONCE per worker (module level → paid on cold start), then every job runs
through handler(event).

Request  (event["input"]):
    Single:  {"text": "...", "format": "mp3"|"wav", "cfg": 2.0, "timesteps": 10,
              "bitrate": 128, "align": false}
    Batch:   {"texts": ["...", "..."], ...same options...}   (texts wins if both present)

Response:
    Single:  {format, contentType, sampleRate, durationSec, audioBase64, [words]}
    Batch:   {sampleRate, format, count, succeeded, failed, items:[{index,text,...}]}
    Errors:  {"error": "..."} (whole request) or per-item {"index","text","error"}.

audioBase64 is the raw mp3/wav bytes, base64-encoded (serverless responses are JSON).

Local test:  python handler.py        # uses test_input.json
"""

import base64
import os

import runpod
import torch

from audio_encode import CONTENT_TYPE, encode
from model_loader import load_model

HF_REPO = os.environ.get("HF_REPO", "")
ASSET_DIR = os.environ.get("ASSET_DIR", "/runpod-volume/assets")
LORA_DIR = os.environ.get("LORA_DIR", os.path.join(ASSET_DIR, "lora_only"))
LORA_STEP = int(os.environ.get("LORA_STEP", 100))
REFERENCE_WAV = os.environ.get("REFERENCE_WAV", os.path.join(ASSET_DIR, "reference.wav"))
LOAD_DENOISER = os.environ.get("LOAD_DENOISER", "1") != "0"
DEFAULT_CFG = float(os.environ.get("DEFAULT_CFG", 2.0))
DEFAULT_TIMESTEPS = int(os.environ.get("DEFAULT_TIMESTEPS", 10))
DEFAULT_BITRATE = int(os.environ.get("DEFAULT_BITRATE", 128))
MAX_BATCH = int(os.environ.get("MAX_BATCH", 200))
# Alignment (whisperx) is heavy and adds to cold start; default OFF for serverless.
ENABLE_ALIGN = os.environ.get("ENABLE_ALIGN", "0") != "0"


def _ensure_assets():
    """Download the LoRA + reference from HF if not already on disk (e.g. first cold start
    on a fresh worker). With a network volume mounted at /runpod-volume they persist and
    later workers skip this."""
    if os.path.isdir(LORA_DIR):
        return
    if not HF_REPO:
        raise RuntimeError(f"{LORA_DIR} missing and HF_REPO unset — cannot fetch the LoRA")
    from huggingface_hub import snapshot_download
    os.makedirs(ASSET_DIR, exist_ok=True)
    print(f"[init] downloading {HF_REPO} -> {ASSET_DIR}", flush=True)
    snapshot_download(repo_id=HF_REPO, repo_type="model", local_dir=ASSET_DIR,
                      token=os.environ.get("HF_TOKEN"))


# ── one-time worker init (cold start) ───────────────────────────────────────────
torch.set_float32_matmul_precision("high")
_ensure_assets()
MODEL = load_model(LORA_DIR, LORA_STEP, load_denoiser=LOAD_DENOISER)
SR = MODEL.tts_model.sample_rate
ALIGNER = None
if ENABLE_ALIGN:
    try:
        from alignment import load_aligner
        ALIGNER = load_aligner()
    except Exception as e:  # noqa: BLE001 — alignment is optional
        print(f"[warn] align model failed to load: {e}", flush=True)
print(f"[init] ready. sample_rate={SR} align={ALIGNER is not None}", flush=True)


def _gen_one(text, cfg, timesteps, fmt, bitrate, align):
    wav = MODEL.generate(text=text, reference_wav_path=REFERENCE_WAV,
                         cfg_value=cfg, inference_timesteps=timesteps)
    data = encode(wav, SR, fmt, bitrate)
    item = {
        "format": fmt,
        "contentType": CONTENT_TYPE[fmt],
        "sampleRate": SR,
        "durationSec": round(len(wav) / SR, 3),
        "audioBase64": base64.b64encode(data).decode("ascii"),
    }
    if align:
        if ALIGNER is None:
            item["alignError"] = "alignment disabled (set ENABLE_ALIGN=1 and redeploy)"
        else:
            from alignment import align_words
            item["words"] = align_words(ALIGNER, wav, SR, text)
    return item


def handler(event):
    inp = event.get("input") or {}
    fmt = inp.get("format", "mp3")
    if fmt not in ("mp3", "wav"):
        return {"error": "format must be 'mp3' or 'wav'"}
    cfg = float(inp.get("cfg", DEFAULT_CFG))
    timesteps = int(inp.get("timesteps", DEFAULT_TIMESTEPS))
    bitrate = int(inp.get("bitrate", DEFAULT_BITRATE))
    align = bool(inp.get("align", False))

    texts = inp.get("texts")
    if texts is not None:
        if not isinstance(texts, list) or not texts:
            return {"error": "texts must be a non-empty array"}
        if len(texts) > MAX_BATCH:
            return {"error": f"texts exceeds MAX_BATCH={MAX_BATCH}"}
        items = []
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                items.append({"index": i, "text": t, "error": "empty text"})
                continue
            try:
                it = _gen_one(t, cfg, timesteps, fmt, bitrate, align)
                it.update(index=i, text=t)
                items.append(it)
            except Exception as e:  # noqa: BLE001 — one bad line shouldn't kill the batch
                items.append({"index": i, "text": t, "error": str(e)})
        ok = sum(1 for it in items if "audioBase64" in it)
        return {"sampleRate": SR, "format": fmt, "count": len(items),
                "succeeded": ok, "failed": len(items) - ok, "items": items}

    text = inp.get("text")
    if not text or not str(text).strip():
        return {"error": "provide 'text' (string) or 'texts' (array)"}
    if len(text) > 2000:
        return {"error": "text exceeds 2000 chars"}
    return _gen_one(text, cfg, timesteps, fmt, bitrate, align)


runpod.serverless.start({"handler": handler})
