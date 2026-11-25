"""Synthetic test material: a drum break, single hits, tones and click tracks. No audio files needed."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from . import dsp
from .config import SAMPLE_RATE

SR = SAMPLE_RATE


def _t(dur: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(dur * sr)) / float(sr)


def kick(sr: int = SR) -> np.ndarray:
    t = _t(0.35, sr)
    f = 45.0 + 110.0 * np.exp(-t * 28.0)
    ph = 2 * np.pi * np.cumsum(f) / sr
    y = np.sin(ph) * np.exp(-t * 11.0)
    y += 0.35 * np.exp(-t * 400.0) * np.random.default_rng(1).normal(0, 1, len(t))
    return (y * 0.95).astype(np.float32)


def snare(sr: int = SR) -> np.ndarray:
    t = _t(0.28, sr)
    rng = np.random.default_rng(2)
    n = lfilter(*dsp.biquad_coeffs("bp", 2200.0, 0.6, sr), rng.normal(0, 1, len(t)))
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 28.0)
    y = n * np.exp(-t * 17.0) * 0.9 + body * 0.5
    return (y * 0.8).astype(np.float32)


def hat(open_: bool = False, sr: int = SR) -> np.ndarray:
    t = _t(0.35 if open_ else 0.06, sr)
    rng = np.random.default_rng(3 if open_ else 4)
    n = lfilter(*dsp.biquad_coeffs("hp", 7000.0, 0.7, sr), rng.normal(0, 1, len(t)))
    y = n * np.exp(-t * (9.0 if open_ else 70.0))
    return (y * 0.45).astype(np.float32)


def clap(sr: int = SR) -> np.ndarray:
    t = _t(0.22, sr)
    rng = np.random.default_rng(5)
    n = lfilter(*dsp.biquad_coeffs("bp", 1500.0, 0.8, sr), rng.normal(0, 1, len(t)))
    env = np.exp(-t * 22.0)
    for d in (0.0, 0.012, 0.024):
        env += 0.9 * np.exp(-np.clip(t - d, 0, None) * 120.0) * (t >= d)
    return (n * env * 0.35).astype(np.float32)


def bass(freq: float = 55.0, dur: float = 0.45, sr: int = SR) -> np.ndarray:
    t = _t(dur, sr)
    y = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    return (y * np.exp(-t * 3.0) * 0.6).astype(np.float32)


def tone(freq: float = 440.0, dur: float = 1.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = _t(dur, sr)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def chord(freqs=(261.63, 329.63, 392.0), dur: float = 1.2, sr: int = SR) -> np.ndarray:
    t = _t(dur, sr)
    y = sum(np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t) for f in freqs)
    return (y * np.exp(-t * 1.6) * 0.18).astype(np.float32)


def _mix(buf: np.ndarray, hit: np.ndarray, at: int, gain: float = 1.0) -> None:
    end = min(len(buf), at + len(hit))
    if end > at:
        buf[at:end] += hit[: end - at] * gain


def drum_break(bpm: float = 92.0, bars: int = 2, sr: int = SR, swing: float = 0.0, with_melody: bool = False) -> np.ndarray:
    """A funky two-bar break built from synthetic hits. Truth positions are in `drum_break_truth`."""
    step = 60.0 / bpm / 4.0                      # 16th note seconds
    n = int(bars * 16 * step * sr) + sr // 2
    buf = np.zeros(n, dtype=np.float32)
    k, s, hc, ho, cl = kick(sr), snare(sr), hat(False, sr), hat(True, sr), clap(sr)
    for pos, kind, g in drum_break_truth(bpm, bars, swing):
        at = int(pos * sr)
        _mix(buf, {"kick": k, "snare": s, "hat": hc, "open_hat": ho, "clap": cl}[kind], at, g)
    if with_melody:
        _mix(buf, chord(sr=sr), 0, 1.0)
        _mix(buf, bass(65.41, sr=sr), int(8 * step * sr), 1.0)
    return dsp.normalize(dsp.ensure_2d(buf), -1.0)


def drum_break_truth(bpm: float = 92.0, bars: int = 2, swing: float = 0.0):
    """List of (time_seconds, kind, gain) for every hit in drum_break()."""
    step = 60.0 / bpm / 4.0
    kicks = {0, 7, 10}                           # 1, the "and" of 2, and a late push
    snares = {4, 12}
    hats = {0, 2, 4, 6, 8, 10, 12, 14}
    out = []
    for bar in range(bars):
        for i in range(16):
            t = (bar * 16 + i) * step
            if i % 2 == 1:
                t += swing * step * 0.5
            if i in kicks:
                out.append((t, "kick", 1.0))
            if i in snares:
                out.append((t, "snare", 0.9))
            if i in hats:
                out.append((t, "open_hat" if (i == 14 and bar % 2 == 1) else "hat", 0.8))
        if bar % 2 == 1:
            out.append(((bar * 16 + 15) * step, "clap", 0.7))
    return sorted(out)


def click_track(bpm: float = 120.0, seconds: float = 8.0, sr: int = SR) -> np.ndarray:
    n = int(seconds * sr)
    buf = np.zeros(n, dtype=np.float32)
    h = kick(sr)
    period = 60.0 / bpm
    t = 0.0
    while int(t * sr) < n:
        _mix(buf, h, int(t * sr))
        t += period
    return dsp.ensure_2d(buf)
