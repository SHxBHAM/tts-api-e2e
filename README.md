# VoxCPM2 Hinglish TTS — E2E API (mp3/wav, single + batch)

Standalone TTS service for the **E2E T4** box. Same VoxCPM2 + Hinglish LoRA voice as
`voxcpm_api`, but adds **batch generation** and **MP3 output**.

```
GET  /            minimal web frontend (single + batch tabs)
GET  /health      JSON status (no auth)
POST /tts         single text   -> audio bytes (mp3 default / wav)        Bearer auth
POST /tts/batch   array of texts -> JSON of base64 audio (one per input)  Bearer auth
```

Output format is per-request: `"format": "mp3"` (default) or `"format": "wav"`.
MP3 is encoded with `lameenc` (a pip wheel — **no system ffmpeg needed**).

## Files
| File | Purpose |
|---|---|
| `server.py` | FastAPI app: `/tts`, `/tts/batch`, `/health`, `/`. Loads model once at startup. |
| `audio_encode.py` | float32 waveform → WAV/MP3 bytes (lameenc, libsndfile fallback). |
| `model_loader.py` | VoxCPM2 + LoRA loading (self-contained copy). |
| `benchmark_rtf.py` | Measure Real-Time Factor (RTF) + peak VRAM on the GPU before serving. |
| `index.html` | Minimal frontend: token, format/cfg/timesteps, Single + Batch tabs. |
| `requirements.txt`, `start.sh` | Deps + boot script. |

## Run on the E2E T4 box
Config lives in a `.env` file (auto-loaded by `start.sh`) — set secrets once, no inline
exports on every restart.
```bash
git clone https://github.com/SHxBHAM/voxcpm-tts-api.git   # or copy this folder
cd voxcpm-tts-api/tts_api_e2e        # adjust if hosted elsewhere
pip install -r requirements.txt

cp .env.example .env && nano .env    # fill in API_TOKEN + HF_TOKEN (and PORT if not 8000)

fuser -k 8000/tcp 2>/dev/null || true                       # free the port (match PORT)
tmux new-session -d -s tts 'bash start.sh 2>&1 | tee /workspace/tts.log'
until curl -s localhost:8000/health | grep -q '"status":"ok"'; do sleep 5; done
curl -s localhost:8000/health; echo
```
Open the chosen `PORT` in the E2E **firewall / security group** (E2E has no auto-proxy).
The `.env` is gitignored, so secrets never get committed.
> On the T4 the model takes ~1 min to warm up, and generation is roughly real-time
> (slower than a 4090). Batch runs **sequentially** on the single GPU.

## Benchmark RTF (before serving)
Get latency/VRAM numbers on the T4 first:
```bash
cd tts_api_e2e && python benchmark_rtf.py --runs 5
```
Loads the model once, warms up, then times generation across short→long texts.
Prints per-run audio duration, gen time, **RTF** (`gen_time / audio_duration`; <1 = faster
than real-time) and peak VRAM, plus mean/median RTF, and appends to `rtf_log.csv`.
Reads `--lora-dir` / `--reference` from env (defaults to `/workspace/assets/...`).

## API

### `POST /tts`  → audio bytes
Request body:
| Field | Type | Default | Notes |
|---|---|---|---|
| `text` | string (1–2000) | — | Devanagari; spell English phonetically. |
| `format` | `"mp3"` \| `"wav"` | `"mp3"` | response audio format |
| `cfg` | float 1–5 | 2.0 | guidance |
| `timesteps` | int 1–50 | 10 | diffusion steps |
| `bitrate` | int 32–320 | 128 | mp3 only (kbps) |

Returns raw bytes: `audio/mpeg` (mp3) or `audio/wav`. Example:
```bash
curl -X POST "$URL/tts" -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"नमस्ते बच्चों।","format":"mp3"}' --output out.mp3
```

### `POST /tts/batch`  → JSON
Request: `{"texts": ["...", "..."], "format":"mp3", "cfg":2.0, "timesteps":10}`
(`texts` capped at `MAX_BATCH`, default 200; each ≤2000 chars.)

Response:
```json
{
  "sampleRate": 48000, "format": "mp3",
  "count": 2, "succeeded": 2, "failed": 0,
  "items": [
    {"index":0, "text":"...", "format":"mp3", "contentType":"audio/mpeg",
     "durationSec": 3.21, "audioBase64": "SUQz...=="},
    {"index":1, "text":"...", "error":"..."}      // a failed line carries `error` instead
  ]
}
```
Decode each `audioBase64` to bytes and save with the matching extension. One bad line
never fails the whole batch.

Python batch client:
```python
import base64, requests
r = requests.post(f"{URL}/tts/batch",
    headers={"Authorization": f"Bearer {API_TOKEN}"},
    json={"texts": ["पहली लाइन।", "दूसरी लाइन।"], "format": "mp3"}, timeout=600)
r.raise_for_status()
for it in r.json()["items"]:
    if "audioBase64" in it:
        open(f"clip_{it['index']}.mp3", "wb").write(base64.b64decode(it["audioBase64"]))
```

## Config (env vars)
`API_TOKEN`, `HF_REPO`, `HF_TOKEN`, `LORA_DIR`, `LORA_STEP` (100), `REFERENCE_WAV`,
`DEFAULT_CFG` (2.0), `DEFAULT_TIMESTEPS` (10), `DEFAULT_BITRATE` (128), `MAX_BATCH` (200),
`PORT` (8000), `HF_HOME`.
