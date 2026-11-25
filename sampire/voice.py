"""One playing instance of a pad sample. Unmodulated voices take fast paths; voices with a realtime filter or
LFO render in chunks aligned to the voice's own age, so their output never depends on the audio block size."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import lfilter

from . import dsp

F32 = np.float32
_AR = np.arange(8192, dtype=np.float64)
GRID = 64


class Mod:
    """Per-voice realtime modulation settings (copied from the pad at trigger time)."""
    __slots__ = ("kind", "cut", "q", "target", "rate", "depth", "shape")

    def __init__(self, kind="", cut=2000.0, q=0.9, target="", rate=5.0, depth=0.0, shape="sine"):
        self.kind, self.cut, self.q = kind, float(cut), float(q)
        self.target, self.rate, self.depth, self.shape = target, float(rate), float(depth), shape

    def active(self) -> bool:
        return bool(self.kind) or (bool(self.target) and self.depth != 0.0)


def lfo_value(shape: str, phase: float) -> float:
    ph = phase % 1.0
    if shape == "tri":
        return 4.0 * abs(ph - 0.5) - 1.0
    if shape == "square":
        return 1.0 if ph < 0.5 else -1.0
    if shape == "saw":
        return 2.0 * ph - 1.0
    return float(np.sin(2.0 * np.pi * ph))


class Voice:
    __slots__ = ("pad", "data", "rate", "base_rate", "bend", "gains", "gain_lin", "pan", "pos", "loop", "mode", "group", "send_a", "send_b",
                 "a", "d", "sus", "r", "stage", "t", "lvl", "rel_from", "dead", "seq", "started", "tag", "bus", "protect", "mod",
                 "age", "zi", "sr")

    def __init__(self, pad: int, data: np.ndarray, rate: float, gains: np.ndarray, a: int, d: int, sus: float, r: int,
                 loop: bool, mode: str, group: int, send_a: float, send_b: float, seq: int, started: int, tag: int = 0,
                 bus: int = 0, protect: bool = False, mod: Optional[Mod] = None, gain_lin: float = 1.0, pan: float = 0.0, sr: int = 48000):
        self.pad, self.data = pad, data
        self.rate = self.base_rate = float(rate)
        self.bend = 1.0
        self.gains = gains.astype(F32)
        self.gain_lin, self.pan = gain_lin, pan
        self.pos = 0.0
        self.loop, self.mode, self.group = loop, mode, group
        self.send_a, self.send_b = send_a, send_b
        self.a, self.d, self.sus, self.r = int(a), int(d), float(sus), max(int(r), 1)
        self.stage = 0 if self.a > 0 else (1 if (self.d > 0 and self.sus < 1.0) else 2)
        self.t = 0
        self.lvl = 0.0 if self.a > 0 else 1.0
        self.rel_from = 1.0
        self.dead = False
        self.seq, self.started, self.tag = seq, started, tag
        self.bus, self.protect = bus, protect
        self.mod = mod if (mod is not None and mod.active()) else None
        self.age = 0
        self.zi = np.zeros((2, 2))
        self.sr = sr

    # -- control
    @property
    def releasing(self) -> bool:
        return self.stage == 3

    def loudness(self) -> float:
        return float(self.lvl * float(np.mean(self.gains)))

    def release(self, frames: int = 0) -> None:
        if self.stage == 3 or self.dead:
            return
        self.rel_from = self.lvl if self.lvl > 0 else 1.0
        self.stage, self.t = 3, 0
        if frames:
            self.r = max(int(frames), 1)

    def set_bend(self, mult: float) -> None:
        self.bend = mult
        self.rate = self.base_rate * mult

    # -- envelope
    def _envelope(self, n: int):
        """Scalar when the level is steady for the whole block, otherwise a float32 vector."""
        if self.stage == 2:
            self.lvl = self.sus
            return self.sus
        env = np.empty(n, dtype=F32)
        i = 0
        while i < n:
            if self.stage == 0:
                m = min(n - i, self.a - self.t)
                env[i:i + m] = (self.t + 1 + _AR[:m]) / self.a
                self.t += m
                i += m
                if self.t >= self.a:
                    self.stage, self.t = 1, 0
                    if self.d <= 0 or self.sus >= 1.0:
                        self.stage = 2
            elif self.stage == 1:
                m = min(n - i, self.d - self.t)
                env[i:i + m] = 1.0 + (self.sus - 1.0) * (self.t + 1 + _AR[:m]) / self.d
                self.t += m
                i += m
                if self.t >= self.d:
                    self.stage = 2
            elif self.stage == 2:
                env[i:] = self.sus
                i = n
            else:
                m = min(n - i, self.r - self.t)
                if m > 0:
                    env[i:i + m] = self.rel_from * (1.0 - (self.t + 1 + _AR[:m]) / self.r)
                    self.t += m
                    i += m
                if self.t >= self.r:
                    env[i:] = 0.0
                    self.dead = True
                    i = n
        self.lvl = float(env[-1]) if n else self.lvl
        return env

    # -- audio
    def render_into(self, out: np.ndarray) -> bool:
        """Add this voice into `out` (n, 2). Returns False once the voice has finished."""
        if self.mod is None:
            ok = self._core(out)
            self.age += len(out)
            return ok
        n = len(out)
        i, ok = 0, True
        while i < n and ok:
            m = min(n - i, GRID - (self.age % GRID))
            ok = self._chunk(out[i:i + m])
            self.age += m
            i += m
        return ok

    def _chunk(self, seg: np.ndarray) -> bool:
        md = self.mod
        m = len(seg)
        centre = (self.age // GRID) * GRID + GRID / 2.0      # grid-aligned, so block splits cannot change it
        v = lfo_value(md.shape, md.rate * centre / self.sr) if (md.target and md.depth) else 0.0
        base_rate, base_gains = self.rate, self.gains
        cut, amp = md.cut, 1.0
        if md.target == "pitch":
            self.rate = base_rate * 2.0 ** (md.depth * v / 12.0)
        elif md.target == "cutoff":
            cut = md.cut * 2.0 ** (md.depth * v)
        elif md.target == "amp":
            amp = 1.0 - md.depth * (0.5 + 0.5 * v)
        elif md.target == "pan":
            ang = (float(np.clip(self.pan + md.depth * v, -1.0, 1.0)) + 1.0) * np.pi / 4.0
            self.gains = (np.array([np.cos(ang), np.sin(ang)], dtype=F32) * F32(self.gain_lin * np.sqrt(2.0)))
        tmp = np.zeros((m, 2), dtype=F32)
        ok = self._core(tmp)
        self.rate, self.gains = base_rate, base_gains
        if md.kind:
            b, a = dsp.biquad_coeffs(md.kind, cut, md.q, self.sr)
            tmp, self.zi = lfilter(b, a, tmp, axis=0, zi=self.zi)
            tmp = tmp.astype(F32)
        seg += tmp * F32(amp)
        return ok

    def _core(self, out: np.ndarray) -> bool:
        n = len(out)
        if self.dead or n == 0:
            return not self.dead
        data, ln = self.data, len(self.data)
        env = self._envelope(n)
        if self.rate == 1.0 and not self.loop:
            p = int(self.pos)
            m = min(n, ln - p)
            if m <= 0:
                return False
            seg = data[p:p + m]
            if isinstance(env, float):
                out[:m] += seg * (self.gains * env)
            else:
                out[:m] += seg * self.gains * env[:m, None]
            self.pos = p + m
            return not self.dead and self.pos < ln
        idx = self.pos + self.rate * _AR[:n]
        if self.loop:
            idx = np.mod(idx, ln)
            i0 = idx.astype(np.int64)
            fr = (idx - i0).astype(F32)[:, None]
            seg = data[i0] * (1.0 - fr) + data[(i0 + 1) % ln] * fr
            m = n
            self.pos = float((self.pos + self.rate * n) % ln)
        else:
            m = int(np.searchsorted(idx, ln - 1.0))
            if m <= 0:
                return False
            i0 = idx[:m].astype(np.int64)
            fr = (idx[:m] - i0).astype(F32)[:, None]
            seg = data[i0] * (1.0 - fr) + data[i0 + 1] * fr
            self.pos = float(self.pos + self.rate * n)
        if isinstance(env, float):
            out[:m] += seg * (self.gains * env)
        else:
            out[:m] += seg * self.gains * env[:m, None]
        return not self.dead and (self.loop or m == n)
