"""
audio_encode.py — turn a float32 model waveform into WAV or MP3 bytes.

MP3 uses `lameenc` (a pip wheel bundling LAME — no system ffmpeg needed, which keeps
deployment on the E2E box simple). Falls back to libsndfile MP3 if lameenc is missing.
"""

import io

import numpy as np
import soundfile as sf

CONTENT_TYPE = {"wav": "audio/wav", "mp3": "audio/mpeg"}
EXT = {"wav": "wav", "mp3": "mp3"}


def _to_int16(wav) -> np.ndarray:
    a = np.asarray(wav, dtype=np.float32).reshape(-1)  # force mono 1-D
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767.0).astype("<i2")


def encode_wav(wav, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32).reshape(-1), sample_rate,
             format="WAV", subtype="PCM_16")
    return buf.getvalue()


def encode_mp3(wav, sample_rate: int, bitrate: int = 128) -> bytes:
    pcm = _to_int16(wav)
    try:
        import lameenc
        enc = lameenc.Encoder()
        enc.set_bit_rate(bitrate)
        enc.set_in_sample_rate(sample_rate)
        enc.set_channels(1)
        enc.set_quality(2)  # 2 = high quality / slower; 7 = fast
        data = enc.encode(pcm.tobytes())
        data += enc.flush()
        return bytes(data)  # lameenc returns a bytearray; Starlette Response needs bytes
    except ImportError:
        # libsndfile >= 1.1 can write MP3 directly.
        buf = io.BytesIO()
        sf.write(buf, np.asarray(wav, dtype=np.float32).reshape(-1), sample_rate, format="MP3")
        return buf.getvalue()


def encode(wav, sample_rate: int, fmt: str = "mp3", bitrate: int = 128) -> bytes:
    if fmt == "wav":
        return encode_wav(wav, sample_rate)
    if fmt == "mp3":
        return encode_mp3(wav, sample_rate, bitrate)
    raise ValueError(f"unsupported format: {fmt!r} (use 'mp3' or 'wav')")
