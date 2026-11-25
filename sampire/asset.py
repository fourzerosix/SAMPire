"""Audio assets: loading, saving, hashing. Source audio is never modified after import."""

from __future__ import annotations

import hashlib
import os
import uuid
import wave
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import dsp
from .config import SAMPLE_RATE

try:  # soundfile reads wav, flac, aiff, ogg and (with a recent libsndfile) mp3
    import soundfile as _sf
except Exception:  # pragma: no cover - exercised only on machines without libsndfile
    _sf = None


@dataclass
class AudioAsset:
    id: str
    name: str
    data: np.ndarray                      # float32 (n, 2) at self.sr
    sr: int = SAMPLE_RATE
    path: Optional[str] = None            # path relative to the project (or absolute before saving)
    sha1: str = ""
    meta: dict = field(default_factory=dict)       # origin, parent asset id, generation, take info
    analysis: dict = field(default_factory=dict)   # cached analysis results (json-safe)

    @property
    def frames(self) -> int:
        return int(len(self.data))

    @property
    def duration(self) -> float:
        return self.frames / float(self.sr)

    def region(self, start: int, end: int) -> np.ndarray:
        start = max(int(start), 0)
        end = min(int(end), self.frames)
        return self.data[start:end]

    def content_hash(self) -> str:
        return hashlib.sha1(np.ascontiguousarray(self.data).tobytes()).hexdigest()


def new_id(prefix: str = "a") -> str:
    return "%s_%s" % (prefix, uuid.uuid4().hex[:8])


def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_wav_fallback(path: str):
    """Minimal PCM wav reader used only when soundfile/libsndfile is missing."""
    try:
        from scipy.io import wavfile

        sr, d = wavfile.read(path)
        if d.dtype == np.int16:
            d = d.astype(np.float32) / 32768.0
        elif d.dtype == np.int32:
            d = d.astype(np.float32) / 2147483648.0
        elif d.dtype == np.uint8:
            d = (d.astype(np.float32) - 128.0) / 128.0
        else:
            d = d.astype(np.float32)
        return d, sr
    except Exception:
        with wave.open(path, "rb") as w:
            ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        if sw == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            v = np.where(v & 0x800000, v - 0x1000000, v).astype(np.float32) / 8388608.0
        else:
            raise
        return v.reshape(-1, ch), sr


def read_audio(path: str):
    """Return (float32 array shaped (n, ch), sample_rate)."""
    if _sf is not None:
        d, sr = _sf.read(path, dtype="float32", always_2d=True)
        return d, sr
    if path.lower().endswith(".wav"):
        d, sr = _read_wav_fallback(path)
        return (d[:, None] if d.ndim == 1 else d), sr
    raise RuntimeError("reading %s needs the 'soundfile' package (pip install soundfile)" % os.path.splitext(path)[1])


def load_audio(path: str, sr: int = SAMPLE_RATE, name: Optional[str] = None) -> AudioAsset:
    d, file_sr = read_audio(path)
    d = dsp.ensure_2d(d)
    if file_sr != sr:
        d = dsp.resample_sr(d, file_sr, sr)
    return AudioAsset(
        id=new_id("a"),
        name=name or os.path.splitext(os.path.basename(path))[0],
        data=d,
        sr=sr,
        path=os.path.abspath(path),
        sha1=sha1_of_file(path),
        meta={"origin": "import", "file_sr": int(file_sr), "generation": 0},
    )


def asset_from_array(data: np.ndarray, name: str, sr: int = SAMPLE_RATE, origin: str = "generated", **meta) -> AudioAsset:
    a = AudioAsset(id=new_id("a"), name=name, data=dsp.ensure_2d(data), sr=sr)
    a.meta = dict(meta, origin=origin)
    a.meta.setdefault("generation", 0)
    a.sha1 = a.content_hash()
    return a


def save_audio(path: str, data: np.ndarray, sr: int = SAMPLE_RATE, subtype: str = "PCM_24") -> None:
    """Write wav/flac/aiff. Falls back to 16-bit / float32 wav when soundfile is unavailable."""
    d = dsp.ensure_2d(data)
    d = np.clip(d, -1.0, 1.0) if subtype != "FLOAT" else d
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if _sf is not None:
        _sf.write(path, d, sr, subtype=subtype)
        return
    from scipy.io import wavfile

    if subtype in ("FLOAT", "DOUBLE"):
        wavfile.write(path, sr, d.astype(np.float32))
    else:
        wavfile.write(path, sr, (d * 32767.0).astype(np.int16))
