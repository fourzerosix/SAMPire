"""Loop-point detection and seamless loop creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import analysis, dsp
from .slicer import snap_zero_crossing


@dataclass
class LoopCandidate:
    start: int
    end: int
    bars: float
    score: float          # 0..1, how well the end joins back to the start

    @property
    def length(self) -> int:
        return self.end - self.start


def _spec(seg: np.ndarray) -> np.ndarray:
    if len(seg) < 256:
        return np.zeros(8)
    w = np.hanning(len(seg))
    s = np.abs(np.fft.rfft(seg * w))
    bands = np.array_split(np.log10(s[1:] + 1e-6), 24)
    return np.array([b.mean() for b in bands])


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


def _join_score(m: np.ndarray, s: int, e: int, w: int) -> float:
    """How alike are the audio before and after the start, and before and after the end? (a periodic signal matches)"""
    parts = []
    after_s, after_e = m[s:s + w], m[e:e + w] if e + w <= len(m) else None
    before_s = m[s - w:s] if s >= w else None
    before_e = m[e - w:e]
    if before_s is not None:
        parts.append(_ncc(before_s, before_e))
        parts.append(1.0 - min(float(np.abs(_spec(before_s) - _spec(before_e)).mean()) / 3.0, 1.0))
    if after_e is not None and len(after_e) == w:
        parts.append(_ncc(after_s, after_e))
        parts.append(1.0 - min(float(np.abs(_spec(after_s) - _spec(after_e)).mean()) / 3.0, 1.0))
    # the tail must flow into the head: compare the last window to the window before the start (or just spectral)
    parts.append(1.0 - min(float(np.abs(_spec(before_e) - _spec(after_s)).mean()) / 3.0, 1.0))
    return float(np.clip(np.mean([max(p, 0.0) for p in parts]), 0.0, 1.0))


def find_loops(x: np.ndarray, sr: int, bpm: Optional[float] = None, beats: Optional[np.ndarray] = None, bars=(1, 2, 4, 8),
               beats_per_bar: int = 4, top: int = 5, window_ms: float = 60.0) -> List[LoopCandidate]:
    """Rank loop regions. Starts are tried on the beat grid, lengths are whole bars; each candidate is scored by how
    well its end joins its start. Refines each boundary to a nearby zero crossing."""
    x = dsp.ensure_2d(x)
    m = dsp.mono(x)
    if bpm is None or beats is None:
        t = analysis.estimate_tempo(x, sr)
        bpm = bpm or float(t["bpm"])
        beats = np.asarray(t["beats"]) if beats is None else beats
    if not bpm or bpm <= 0 or len(beats) == 0:
        return []
    beat = sr * 60.0 / bpm
    w = int(sr * window_ms / 1000.0)
    starts = [int(b * sr) for b in beats[: min(len(beats), 64)]]
    out: List[LoopCandidate] = []
    for s in starts:
        for nb in bars:
            length = int(round(nb * beats_per_bar * beat))
            e = s + length
            if e > len(m) - 1:
                continue
            sc = _join_score(m, s, e, w)
            out.append(LoopCandidate(snap_zero_crossing(m, s), snap_zero_crossing(m, e), float(nb), sc))
    if not out:
        return []
    best = max(c.score for c in out)
    # prefer the shortest loop that is nearly as good as the best (a 4 bar loop of a 1 bar groove should be reported as 1 bar)
    out.sort(key=lambda c: (-(c.score >= best - 0.02), c.bars, -c.score, c.start))
    seen, uniq = set(), []
    for c in out:
        key = (c.start // (sr // 20), c.bars)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq[:top]


def make_seamless(x: np.ndarray, xfade_ms: float = 30.0, sr: int = 48000) -> np.ndarray:
    """Equal-power crossfade of the tail into the head so the loop point cannot click. Output is xfade shorter."""
    x = dsp.ensure_2d(x)
    xf = min(int(sr * xfade_ms / 1000.0), len(x) // 2)
    if xf < 8:
        return x.copy()
    out = x[: len(x) - xf].copy()
    t = np.linspace(0.0, np.pi / 2.0, xf, dtype=np.float32)[:, None]
    out[:xf] = x[:xf] * np.sin(t) + x[len(x) - xf:] * np.cos(t)
    return out.astype(np.float32)
