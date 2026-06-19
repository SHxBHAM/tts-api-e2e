# RunPod Serverless worker for VoxCPM2 + Hinglish LoRA TTS.
# Build:  docker build -t <user>/voxcpm-tts-serverless .
# (or use RunPod's GitHub integration to build this repo directly.)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    # Cache big downloads on the network volume so cold starts don't re-download:
    HF_HOME=/runpod-volume/hf \
    MODELSCOPE_CACHE=/runpod-volume/modelscope

WORKDIR /app

# ffmpeg is needed by whisperx (alignment); git for any VCS-based pip installs.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir runpod -r requirements.txt

COPY . .

# RunPod invokes the handler; runpod.serverless.start() keeps the worker alive.
CMD ["python", "-u", "handler.py"]
