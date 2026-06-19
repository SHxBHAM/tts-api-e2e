# Deploy on RunPod Serverless

Run the same VoxCPM2 + Hinglish LoRA TTS as a **queue-based Serverless endpoint** —
scales to zero when idle, autoscales workers under load. Files: `handler.py` (worker),
`Dockerfile` (image), `test_input.json` (local test).

The handler returns **base64 audio in JSON** (serverless responses are JSON, not raw bytes).
Single + batch + optional word alignment, mirroring the FastAPI server.

## 1. (Optional) Test the handler locally
On a machine with the deps + GPU:
```bash
pip install runpod -r requirements.txt
HF_REPO=Shxbhxm21/voxcpm2-hinglish-lora HF_TOKEN=hf_xxx \
ASSET_DIR=./assets python handler.py        # runs handler once with test_input.json
```

## 2. Deploy — two paths

### A) GitHub integration (no local Docker)
RunPod builds the repo for you:
1. RunPod console → **Serverless → New Endpoint → GitHub**.
2. Pick `SHxBHAM/tts-api-e2e`, branch `main`, **Dockerfile path** `Dockerfile`.
3. Configure (see §3), deploy. RunPod builds the image and rolls it out.

### B) Build & push the image yourself
```bash
docker build -t <dockerhub-user>/voxcpm-tts-serverless:latest .
docker push <dockerhub-user>/voxcpm-tts-serverless:latest
```
Then RunPod console → **Serverless → New Endpoint → Docker image** → paste the tag.

## 3. Endpoint configuration

**GPU:** 16GB+ (RTX 4090 / A5000) — base model ~3–4 GB, plenty of headroom.

**Network volume (recommended):** attach one so the ~3 GB base model + LoRA + denoiser are
downloaded **once** and reused — without it, every cold start re-downloads them. The handler
defaults `ASSET_DIR=/runpod-volume/assets`, `HF_HOME=/runpod-volume/hf`,
`MODELSCOPE_CACHE=/runpod-volume/modelscope` (all on the volume).

**Workers:** `min=0` for true scale-to-zero (cheapest, but cold starts), or `min=1` +
**FlashBoot** for low-latency. `max` = your desired parallelism (each worker = 1 concurrent job).

**Env vars:**
| Var | Value |
|---|---|
| `HF_REPO` | `Shxbhxm21/voxcpm2-hinglish-lora` |
| `HF_TOKEN` | HF read token (private LoRA repo) |
| `LORA_STEP` | `100` (optional) |
| `ENABLE_ALIGN` | `1` to load whisperx for word timestamps (heavier cold start); default `0` |
| `DEFAULT_CFG` / `DEFAULT_TIMESTEPS` / `DEFAULT_BITRATE` / `MAX_BATCH` | optional overrides |

> No app-level token needed — RunPod protects the endpoint with your **RunPod API key**.

## 4. Call it

Async (`/run` → poll `/status/<id>`) or sync (`/runsync`, waits for the result):
```bash
ENDPOINT_ID=xxxxxxxx
RUNPOD_API_KEY=rpa_xxx

# single -> mp3
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"text":"नमस्ते बच्चों।","format":"mp3"}}'

# batch + word alignment
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"texts":["पहली लाइन।","दूसरी लाइन।"],"format":"mp3","align":true}}'
```

Decode the audio (Python):
```python
import base64, requests
r = requests.post(f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    json={"input": {"text": "नमस्ते।", "format": "mp3"}}, timeout=300)
out = r.json()["output"]
open("out.mp3", "wb").write(base64.b64decode(out["audioBase64"]))
```
(RunPod wraps the handler's return value under `"output"`.)

## ⚠️ Payload size limits (affects batch)
RunPod caps the payload — **including the response** — at **20 MB for `/runsync`** and
**10 MB for `/run`**. Since this worker returns audio as base64 in JSON, a large batch can
exceed it. Rough ceilings:

| format | `/run` (10 MB) | `/runsync` (20 MB) |
|---|---|---|
| mp3 (128 kbps) | ~250 clips | ~500 clips |
| wav (48 kHz 16-bit) | **~50 clips** | **~100 clips** |

Guidance:
- **Single / small batch:** base64-in-JSON is fine.
- **Large jobs:** prefer **mp3**, keep batches modest, and use **`/run` + `/status`** (async)
  rather than `/runsync` (a long batch on one worker can also hit `/runsync` timeouts).
- **For real scale / true parallelism:** send many **single `/run` jobs** and let RunPod fan
  them across workers — don't push one giant batch through a single worker.
- If you truly need huge outputs in one call, switch to the "write to a network volume / S3
  and return URLs" pattern instead of base64 (ask and I'll add it).

## Notes
- **Cold start:** first request after scale-to-zero boots a worker + loads the model
  (~30–90s, more with `ENABLE_ALIGN=1`). Use `min workers ≥1` + FlashBoot to avoid it.
- **Best fit:** batch / offline lecture-audio generation (scale up, then to zero). For
  always-on low-latency, the FastAPI pod (`start.sh`) is still simpler.
- No GPU lock needed here — each worker processes one job at a time (verified in RunPod docs);
  RunPod adds parallelism by adding workers.
