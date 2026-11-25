"""Stem separation. Built in: a DSP split (harmonic/percussive masks, then a low-band bass split) that needs only
NumPy/SciPy and is deterministic. Optional: Demucs (neural, much better, needs `pip install demucs` and a lot of CPU/RAM)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import istft, stft

from . import dsp
from . import asset as assetmod

F32 = np.float32


def hpss(x: np.ndarray, sr: int, harm_width: int = 51, perc_width: int = 13, power: float = 4.0) -> Dict[str, np.ndarray]:
    """Soft-mask harmonic/percussive separation. harmonic + percussive reconstructs the input (complementary masks)."""
    x = dsp.ensure_2d(x)
    n = len(x)
    nper, hop = 2048, 512
    f, t, Z = stft(x.T, fs=sr, nperseg=nper, noverlap=nper - hop, boundary="zeros", padded=True)     # (ch, bins, frames)
    mag = np.abs(Z).mean(axis=0)
    H = median_filter(mag, size=(1, harm_width))
    P = median_filter(mag, size=(perc_width, 1))
    Hp, Pp = H ** power, P ** power
    mh = Hp / (Hp + Pp + 1e-12)
    mp = 1.0 - mh
    out = {}
    for name, m in (("harmonic", mh), ("percussive", mp)):
        _, y = istft(Z * m[None], fs=sr, nperseg=nper, noverlap=nper - hop, boundary=True)
        y = y.T[:n]
        if len(y) < n:
            y = np.pad(y, ((0, n - len(y)), (0, 0)))
        out[name] = y.astype(F32)
    return out


def separate_dsp(x: np.ndarray, sr: int, bass_hz: float = 150.0) -> Dict[str, np.ndarray]:
    """drums (percussive), bass (low harmonic), other (the rest of the harmonic part). drums + bass + other == input."""
    parts = hpss(x, sr)
    harm = parts["harmonic"]
    bass = dsp.apply_filter(dsp.apply_filter(harm, "lp", bass_hz, 0.707, sr), "lp", bass_hz, 0.707, sr)
    return {"drums": parts["percussive"], "bass": bass.astype(F32), "other": (harm - bass).astype(F32)}


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except Exception:
        return shutil.which("demucs") is not None


def separate_demucs(path: str, sr: int, model: str = "htdemucs") -> Dict[str, np.ndarray]:
    """Run the Demucs command line on a file and load drums/bass/other/vocals. Slow; downloads model weights on first use."""
    if not demucs_available():
        raise RuntimeError("Demucs is not installed (pip install demucs). The built-in DSP split needs nothing: method='dsp'.")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [shutil.which("demucs") or sys.executable, "-n", model, "-o", tmp, path] if shutil.which("demucs") \
            else [sys.executable, "-m", "demucs", "-n", model, "-o", tmp, path]
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
        stem_dir = os.path.join(tmp, model, os.path.splitext(os.path.basename(path))[0])
        out = {}
        for stem in ("drums", "bass", "other", "vocals"):
            f = os.path.join(stem_dir, stem + ".wav")
            if os.path.exists(f):
                out[stem] = assetmod.load_audio(f, sr).data
        if not out:
            raise RuntimeError("Demucs produced no stems in %s" % stem_dir)
        return out


def separate(x: np.ndarray, sr: int, method: str = "dsp", path: str = "") -> Dict[str, np.ndarray]:
    if method == "dsp":
        return separate_dsp(x, sr)
    if method == "demucs":
        if not path:
            raise ValueError("demucs needs the source file path")
        return separate_demucs(path, sr)
    raise ValueError("method must be 'dsp' or 'demucs'")
