"""Slicing: transient, grid, equal, silence and hybrid. All return a sorted list of Slice (frames)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import analysis, dsp


@dataclass
class Slice:
    start: int
    end: int
    strength: float = 1.0
    label: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"start": int(self.start), "end": int(self.end), "strength": float(self.strength), "label": self.label}

    @staticmethod
    def from_dict(d: dict) -> "Slice":
        return Slice(int(d["start"]), int(d["end"]), float(d.get("strength", 1.0)), d.get("label", ""))


def snap_zero_crossing(m: np.ndarray, idx: int, radius: int = 192) -> int:
    """Move idx to the nearest rising zero crossing so a cut does not click."""
    lo, hi = max(idx - radius, 1), min(idx + radius, len(m) - 1)
    if hi <= lo:
        return int(idx)
    seg = m[lo - 1:hi]
    rising = np.where((seg[:-1] <= 0) & (seg[1:] > 0))[0] + lo
    if len(rising) == 0:
        cross = np.where(np.signbit(seg[:-1]) != np.signbit(seg[1:]))[0] + lo
        if len(cross) == 0:
            return int(idx)
        return int(cross[np.argmin(np.abs(cross - idx))])
    return int(rising[np.argmin(np.abs(rising - idx))])


def _finish(starts: np.ndarray, strengths: np.ndarray, x: np.ndarray, sr: int, min_len_ms: float, max_len_ms: Optional[float],
            pre_ms: float, snap: bool, tail_ms: Optional[float]) -> List[Slice]:
    """Turn onset positions into slices: pre-roll, zero-cross snap, min/max length, end at next onset."""
    n = len(x)
    m = dsp.mono(x)
    pre = int(sr * pre_ms / 1000.0)
    st = [max(int(s) - pre, 0) for s in starts]
    if snap:
        st = [snap_zero_crossing(m, s) for s in st]
    order = np.argsort(st)
    st = [st[i] for i in order]
    sg = [float(strengths[i]) if len(strengths) else 1.0 for i in order]
    out: List[Slice] = []
    min_len = int(sr * min_len_ms / 1000.0)
    for i, s in enumerate(st):
        e = st[i + 1] if i + 1 < len(st) else n
        if tail_ms is not None:
            e = min(e, s + int(sr * tail_ms / 1000.0))
        if max_len_ms is not None:
            e = min(e, s + int(sr * max_len_ms / 1000.0))
        if e - s >= min_len:
            out.append(Slice(int(s), int(e), sg[i]))
    return out


def slice_transient(x: np.ndarray, sr: int, sensitivity: float = 0.5, min_len_ms: float = 30.0, max_len_ms: Optional[float] = None,
                    pre_ms: float = 1.5, snap: bool = True, tail_ms: Optional[float] = None, min_gap_ms: float = 35.0) -> List[Slice]:
    pos, strength = analysis.detect_onsets(x, sr, sensitivity, min_gap_ms)
    if len(pos) == 0:
        return []
    return _finish(pos, strength, x, sr, min_len_ms, max_len_ms, pre_ms, snap, tail_ms)


def slice_grid(n_frames: int, sr: int, bpm: float, subdivision: int = 4, offset_s: float = 0.0) -> List[Slice]:
    """Cut on a musical grid. subdivision = slices per beat (1 = beats, 2 = 8ths, 4 = 16ths)."""
    step = int(round(sr * 60.0 / bpm / max(subdivision, 1)))
    start = int(offset_s * sr)
    out, s = [], start
    while s + step // 2 < n_frames:
        out.append(Slice(s, min(s + step, n_frames), 1.0))
        s += step
    return out


def slice_equal(n_frames: int, parts: int) -> List[Slice]:
    parts = max(int(parts), 1)
    edges = np.linspace(0, n_frames, parts + 1).astype(int)
    return [Slice(int(a), int(b), 1.0) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def slice_silence(x: np.ndarray, sr: int, thresh_db: float = -45.0, min_silence_ms: float = 40.0, min_len_ms: float = 30.0) -> List[Slice]:
    m = np.abs(dsp.mono(x))
    env = uniform_filter1d(m, max(int(sr * 0.005), 4))
    loud = env > dsp.db_to_lin(thresh_db)
    out, i, n = [], 0, len(loud)
    gap = int(sr * min_silence_ms / 1000.0)
    while i < n:
        if not loud[i]:
            i += 1
            continue
        j = i
        quiet = 0
        while j < n and quiet < gap:
            quiet = 0 if loud[j] else quiet + 1
            j += 1
        end = j - quiet
        if end - i >= int(sr * min_len_ms / 1000.0):
            out.append(Slice(i, end, float(env[i:end].max())))
        i = j
    return out


def slice_hybrid(x: np.ndarray, sr: int, sensitivity: float = 0.5, bpm: Optional[float] = None, quantize: float = 0.0,
                 fill_gaps: bool = True, max_len_ms: Optional[float] = 2000.0, **kw) -> List[Slice]:
    """Transient slicing nudged by the musical grid, with extra cuts where a long stretch has no attack."""
    pos, strength = analysis.detect_onsets(x, sr, sensitivity)
    pos = pos.astype(np.int64)
    strength = strength.astype(np.float32)
    if bpm and bpm > 0 and len(pos):
        t = analysis.estimate_tempo(x, sr) if False else None  # tempo is passed in; avoid re-running it
        step = sr * 60.0 / bpm / 4.0
        first = float(pos[0])
        grid_pos = first + np.round((pos - first) / step) * step
        if quantize > 0:
            pos = np.round(pos + (grid_pos - pos) * float(np.clip(quantize, 0, 1))).astype(np.int64)
        if fill_gaps:
            beat = sr * 60.0 / bpm
            extra, extra_s = [], []
            for a, b in zip(pos[:-1], pos[1:]):
                k = int((b - a) // (beat * 2.0))
                for j in range(1, k + 1):
                    extra.append(int(a + j * beat * 2.0))
                    extra_s.append(0.15)
            if extra:
                pos = np.concatenate([pos, np.array(extra, dtype=np.int64)])
                strength = np.concatenate([strength, np.array(extra_s, dtype=np.float32)])
    if len(pos) == 0:
        return slice_equal(len(x), 16)
    return _finish(pos, strength, x, sr, kw.get("min_len_ms", 30.0), max_len_ms, kw.get("pre_ms", 1.5), kw.get("snap", True), kw.get("tail_ms"))


def make_slices(mode: str, x: np.ndarray, sr: int, count: int = 16, sensitivity: float = 0.5, bpm: Optional[float] = None, **kw) -> List[Slice]:
    mode = mode.lower()
    if mode == "transient":
        return slice_transient(x, sr, sensitivity, **kw)
    if mode == "grid":
        return slice_grid(len(x), sr, bpm or 120.0, kw.get("subdivision", 4), kw.get("offset_s", 0.0))
    if mode == "equal":
        return slice_equal(len(x), count)
    if mode == "silence":
        return slice_silence(x, sr)
    if mode == "hybrid":
        return slice_hybrid(x, sr, sensitivity, bpm, **kw)
    raise ValueError("unknown slice mode %r (transient, grid, equal, silence, hybrid)" % mode)
