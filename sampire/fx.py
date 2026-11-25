"""Real-time effects. Every effect is exact for any block size: feedback structures are processed in chunks no
longer than their shortest delay, so output never depends on how the host slices the audio."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.signal import lfilter

from . import dsp

F32 = np.float32


class FX:
    name = "fx"
    tail_s = 0.0

    def __init__(self, sr: int, mix: float = 1.0):
        self.sr = sr
        self.mix = float(mix)

    def wet(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def process(self, x: np.ndarray) -> np.ndarray:
        w = self.wet(x)
        if self.mix >= 0.999:
            return w
        return (x * (1.0 - self.mix) + w * self.mix).astype(F32)

    def params(self) -> dict:
        return {"mix": self.mix}


class DelayFX(FX):
    name = "delay"

    def __init__(self, sr: int, time_s: float = 0.375, feedback: float = 0.45, mix: float = 1.0, tone_hz: float = 5000.0):
        super().__init__(sr, mix)
        self.feedback = feedback
        self.tone_hz = tone_hz
        self._set_time(time_s)
        self.tail_s = 6.0

    def _set_time(self, time_s: float) -> None:
        self.time_s = time_s
        d = max(int(time_s * self.sr), 256)
        self.buf = np.zeros((d, 2), dtype=F32)
        self.idx = 0
        self.b, self.a = dsp.biquad_coeffs("lp", self.tone_hz, 0.707, self.sr)
        self.zi = np.zeros((2, 2))          # (filter order, channels)

    def set_time(self, time_s: float) -> None:
        if abs(time_s - self.time_s) > 1e-6:
            self._set_time(time_s)

    def sync(self, bpm: float, beats: float = 0.75) -> None:
        self.set_time(60.0 / max(bpm, 20.0) * beats)

    def wet(self, x: np.ndarray) -> np.ndarray:
        n, d = len(x), len(self.buf)
        out = np.empty_like(x)
        pos = 0
        while pos < n:
            m = min(n - pos, d)
            i = (self.idx + np.arange(m)) % d
            read = self.buf[i]
            fb, self.zi = lfilter(self.b, self.a, read * self.feedback, axis=0, zi=self.zi)
            self.buf[i] = x[pos:pos + m] + fb
            out[pos:pos + m] = read
            self.idx = (self.idx + m) % d
            pos += m
        return out

    def params(self) -> dict:
        return {"time_s": self.time_s, "feedback": self.feedback, "mix": self.mix}


class _Comb:
    def __init__(self, d: int, g: float):
        self.buf = np.zeros(d, dtype=F32)
        self.d, self.g, self.idx = d, g, 0

    def run(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        out = np.empty(n, dtype=F32)
        pos = 0
        while pos < n:
            m = min(n - pos, self.d)
            i = (self.idx + np.arange(m)) % self.d
            r = self.buf[i]
            self.buf[i] = x[pos:pos + m] + self.g * r
            out[pos:pos + m] = r
            self.idx = (self.idx + m) % self.d
            pos += m
        return out


class _Allpass:
    def __init__(self, d: int, g: float = 0.5):
        self.buf = np.zeros(d, dtype=F32)
        self.d, self.g, self.idx = d, g, 0

    def run(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        out = np.empty(n, dtype=F32)
        pos = 0
        while pos < n:
            m = min(n - pos, self.d)
            i = (self.idx + np.arange(m)) % self.d
            r = self.buf[i]
            v = x[pos:pos + m] + self.g * r
            out[pos:pos + m] = -self.g * v + r
            self.buf[i] = v
            self.idx = (self.idx + m) % self.d
            pos += m
        return out


class ReverbFX(FX):
    """Small Schroeder/Freeverb style room: 4 combs and 2 allpasses per channel, stereo spread."""
    name = "reverb"

    def __init__(self, sr: int, room: float = 0.8, mix: float = 1.0, damp_hz: float = 6000.0):
        super().__init__(sr, mix)
        k = sr / 44100.0
        g = 0.70 + 0.27 * float(np.clip(room, 0.0, 1.0))
        self.room = room
        self.combs = [[_Comb(int(d * k) + s, g) for d in (1116, 1188, 1277, 1356)] for s in (0, 23)]
        self.aps = [[_Allpass(int(d * k) + s) for d in (556, 441)] for s in (0, 23)]
        self.b, self.a = dsp.biquad_coeffs("lp", damp_hz, 0.707, sr)
        self.zi = [np.zeros(2), np.zeros(2)]
        self.tail_s = 5.0

    def wet(self, x: np.ndarray) -> np.ndarray:
        mono_in = (x.mean(axis=1) * 0.25).astype(F32)
        out = np.empty_like(x)
        for ch in (0, 1):
            y = sum(c.run(mono_in) for c in self.combs[ch])
            for ap in self.aps[ch]:
                y = ap.run(y)
            y, self.zi[ch] = lfilter(self.b, self.a, y, zi=self.zi[ch])
            out[:, ch] = y
        return out

    def params(self) -> dict:
        return {"room": self.room, "mix": self.mix}


class FilterFX(FX):
    name = "filter"

    def __init__(self, sr: int, kind: str = "lp", cutoff: float = 3000.0, q: float = 0.9, mix: float = 1.0):
        super().__init__(sr, mix)
        self.kind, self.cutoff, self.q = kind, cutoff, q
        self._build()

    def _build(self) -> None:
        self.b, self.a = dsp.biquad_coeffs(self.kind, self.cutoff, self.q, self.sr)
        self.zi = np.zeros((2, 2))

    def set(self, kind: Optional[str] = None, cutoff: Optional[float] = None, q: Optional[float] = None) -> None:
        self.kind = kind or self.kind
        self.cutoff = cutoff if cutoff is not None else self.cutoff
        self.q = q if q is not None else self.q
        self._build()

    def wet(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y.astype(F32)

    def params(self) -> dict:
        return {"kind": self.kind, "cutoff": self.cutoff, "q": self.q, "mix": self.mix}


class DriveFX(FX):
    name = "drive"

    def __init__(self, sr: int, amount: float = 0.4, mix: float = 1.0):
        super().__init__(sr, mix)
        self.amount = amount

    def wet(self, x: np.ndarray) -> np.ndarray:
        return dsp.drive(x, self.amount)

    def params(self) -> dict:
        return {"amount": self.amount, "mix": self.mix}


class CrushFX(FX):
    name = "crush"

    def __init__(self, sr: int, bits: int = 8, downsample: int = 4, mix: float = 1.0):
        super().__init__(sr, mix)
        self.bits, self.down = bits, max(int(downsample), 1)
        self.phase = 0
        self.last = np.zeros(2, dtype=F32)

    def wet(self, x: np.ndarray) -> np.ndarray:
        n, d = len(x), self.down
        base = ((self.phase + np.arange(n)) // d) * d - self.phase      # sample-and-hold positions
        held = x[np.clip(base, 0, n - 1)].copy()
        held[base < 0] = self.last
        self.last = x[base[-1]].copy() if base[-1] >= 0 else self.last
        self.phase = (self.phase + n) % d
        levels = 2.0 ** (max(self.bits, 1) - 1)
        return (np.round(held * levels) / levels).astype(F32)

    def params(self) -> dict:
        return {"bits": self.bits, "downsample": self.down, "mix": self.mix}


REGISTRY = {"delay": DelayFX, "reverb": ReverbFX, "filter": FilterFX, "drive": DriveFX, "crush": CrushFX}


def make_fx(name: str, sr: int, **kw) -> FX:
    if name not in REGISTRY:
        raise ValueError("unknown effect %r (%s)" % (name, ", ".join(sorted(REGISTRY))))
    return REGISTRY[name](sr, **kw)
