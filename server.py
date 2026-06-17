"""
server.py — VoxCPM2 + Hinglish LoRA TTS service for the E2E T4 box.

Loads the model ONCE at startup, then exposes:
    GET  /            -> minimal web frontend (index.html)
    GET  /health      -> {"status": "ok", ...}        (no auth)
    POST /tts         -> single text  -> audio bytes  (mp3 default, or wav)   (Bearer auth)
    POST /tts/batch   -> array of texts -> JSON of base64 audio               (Bearer auth)

Audio format is chosen per request via "format": "mp3" (default) | "wav".

Config via environment variables:
    API_TOKEN          shared bearer token (required to enable auth)
    LORA_DIR           dir containing step_XXXXXXX/ checkpoints (default: /workspace/assets/lora_only)
    LORA_STEP          checkpoint step to serve (default: 100)
    REFERENCE_WAV      reference speaker wav (default: /workspace/assets/reference.wav)
    LOAD_DENOISER      "1"/"0" (default: 1)
    DEFAULT_CFG        default classifier-free guidance (default: 2.0)
    DEFAULT_TIMESTEPS  default diffusion timesteps (default: 10)
    DEFAULT_BITRATE    default mp3 bitrate kbps (default: 128)
    MAX_BATCH          max texts per /tts/batch call (default: 200)

Run:  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import base64
import os
import threading
from typing import List, Literal

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

from audio_encode import CONTENT_TYPE, EXT, encode
from model_loader import load_model

API_TOKEN = os.environ.get("API_TOKEN", "")
LORA_DIR = os.environ.get("LORA_DIR", "/workspace/assets/lora_only")
LORA_STEP = int(os.environ.get("LORA_STEP", 100))
REFERENCE_WAV = os.environ.get("REFERENCE_WAV", "/workspace/assets/reference.wav")
LOAD_DENOISER = os.environ.get("LOAD_DENOISER", "1") != "0"
DEFAULT_CFG = float(os.environ.get("DEFAULT_CFG", 2.0))
DEFAULT_TIMESTEPS = int(os.environ.get("DEFAULT_TIMESTEPS", 10))
DEFAULT_BITRATE = int(os.environ.get("DEFAULT_BITRATE", 128))
MAX_BATCH = int(os.environ.get("MAX_BATCH", 200))
# Max requests allowed in flight (queued + generating). Beyond this we return 503 fast
# instead of letting threads pile up on a single GPU.
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", 16))

HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="VoxCPM2 Hinglish TTS (E2E)")
STATE = {"model": None, "sample_rate": None}

# One GPU, one model instance -> generation MUST be serialized. This lock guarantees only
# one model.generate() runs at a time (concurrent calls would race / corrupt / spike VRAM).
_gpu_lock = threading.Lock()
# Admission control: cap total in-flight requests so overload fails fast (503) rather than
# growing an unbounded queue behind the lock.
_inflight = threading.BoundedSemaphore(MAX_INFLIGHT)


@app.on_event("startup")
def _startup():
    torch.set_float32_matmul_precision("high")  # silences the warmup warning, minor speedup
    model = load_model(LORA_DIR, LORA_STEP, load_denoiser=LOAD_DENOISER)
    STATE["model"] = model
    STATE["sample_rate"] = model.tts_model.sample_rate
    if not API_TOKEN:
        print("[warn] API_TOKEN is empty — endpoints are UNAUTHENTICATED. "
              "Set API_TOKEN before exposing the box.", flush=True)


def require_token(authorization: str = Header(default="")):
    if not API_TOKEN:
        return
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def admit():
    """Admission gate: take an in-flight slot or 503 immediately if the server is saturated.
    The slot is released after the response is produced."""
    if not _inflight.acquire(blocking=False):
        raise HTTPException(status_code=503,
                            detail="Server busy — too many requests in flight, retry shortly")
    try:
        yield
    finally:
        _inflight.release()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    format: Literal["mp3", "wav"] = "mp3"
    cfg: float = Field(default=DEFAULT_CFG, ge=1.0, le=5.0)
    timesteps: int = Field(default=DEFAULT_TIMESTEPS, ge=1, le=50)
    bitrate: int = Field(default=DEFAULT_BITRATE, ge=32, le=320)


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    format: Literal["mp3", "wav"] = "mp3"
    cfg: float = Field(default=DEFAULT_CFG, ge=1.0, le=5.0)
    timesteps: int = Field(default=DEFAULT_TIMESTEPS, ge=1, le=50)
    bitrate: int = Field(default=DEFAULT_BITRATE, ge=32, le=320)

    @field_validator("texts")
    @classmethod
    def _check_items(cls, v):
        for i, t in enumerate(v):
            if not t or not t.strip():
                raise ValueError(f"texts[{i}] is empty")
            if len(t) > 2000:
                raise ValueError(f"texts[{i}] exceeds 2000 chars")
        return v


def _synth(text: str, cfg: float, timesteps: int):
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    # Serialize GPU access: only one generation at a time on the single T4.
    with _gpu_lock:
        return model.generate(
            text=text,
            reference_wav_path=REFERENCE_WAV,
            cfg_value=cfg,
            inference_timesteps=timesteps,
        )


@app.get("/health")
def health():
    return {
        "status": "ok" if STATE["model"] is not None else "loading",
        "step": LORA_STEP,
        "sample_rate": STATE["sample_rate"],
        "auth": bool(API_TOKEN),
        "max_batch": MAX_BATCH,
        "max_inflight": MAX_INFLIGHT,
        "formats": ["mp3", "wav"],
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.post("/tts", dependencies=[Depends(require_token), Depends(admit)])
def tts(req: TTSRequest):
    """Single text -> audio bytes (mp3 by default, wav if format='wav')."""
    wav = _synth(req.text, req.cfg, req.timesteps)
    data = encode(wav, STATE["sample_rate"], req.format, req.bitrate)
    return Response(
        content=data,
        media_type=CONTENT_TYPE[req.format],
        headers={"Content-Disposition": f'inline; filename="output.{EXT[req.format]}"'},
    )


@app.post("/tts/batch", dependencies=[Depends(require_token), Depends(admit)])
def tts_batch(req: BatchRequest):
    """Array of texts -> JSON list of base64-encoded audio (one per input).

    Generation is sequential (single GPU). A failing item gets an `error` field
    instead of `audioBase64`, so one bad line never fails the whole batch.
    """
    sr = STATE["sample_rate"]
    items = []
    for i, text in enumerate(req.texts):
        try:
            wav = _synth(text, req.cfg, req.timesteps)
            data = encode(wav, sr, req.format, req.bitrate)
            duration = len(wav) / sr if sr else None
            items.append({
                "index": i,
                "text": text,
                "format": req.format,
                "contentType": CONTENT_TYPE[req.format],
                "durationSec": round(duration, 3) if duration else None,
                "audioBase64": base64.b64encode(data).decode("ascii"),
            })
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 — one bad line shouldn't kill the batch
            items.append({"index": i, "text": text, "error": str(e)})
    ok = sum(1 for it in items if "audioBase64" in it)
    return {
        "sampleRate": sr,
        "format": req.format,
        "count": len(items),
        "succeeded": ok,
        "failed": len(items) - ok,
        "items": items,
    }
