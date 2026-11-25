"""Real-time effects rack. Every built-in effect gives identical output for any audio block size: feedback
structures are processed in chunks no longer than their shortest delay, modulation is derived from an absolute
sample counter, and noise is generated on a fixed grid. Effects are described by small dict specs so they can be
saved in projects: {"type": "compressor", "threshold_db": -20, ...}."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import lfilter

from . import dsp

F32 = np.float32


class FX:
    """Base class. Subclasses declare DEFAULTS (every constructor parameter) and implement wet()."""

    name = "fx"
    DEFAULTS: Dict[str, object] = {"mix": 1.0}
    tail_s = 0.0
    needs_key = False                 # True for sidechain-capable effects

    def __init__(self, sr: int, **kw):
        self.sr = sr
        self.enabled = True
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)
        self.t = 0                    # absolute sample counter
        self._apply(kw)
        self._update(first=True)

    def _apply(self, kw: dict) -> None:
        for k, v in kw.items():
            if k not in self.DEFAULTS:
                raise ValueError("%s has no parameter %r (has: %s)" % (self.name, k, ", ".join(self.DEFAULTS)))
            setattr(self, k, type(self.DEFAULTS[k])(v) if not isinstance(self.DEFAULTS[k], str) else str(v))

    def set(self, **kw) -> None:
        self._apply(kw)
        self._update()

    def _update(self, first: bool = False) -> None:
        pass

    def wet(self, x: np.ndarray, key: Optional[np.ndarray] = None) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def process(self, x: np.ndarray, key: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(x)
        if not self.enabled:
            self.t += n
            return x
        w = self.wet(x, key)
        self.t += n
        if self.mix >= 0.999:
            return w
        return (x * (1.0 - self.mix) + w * self.mix).astype(F32)

    def params(self) -> dict:
        return {k: getattr(self, k) for k in self.DEFAULTS}

    def spec(self) -> dict:
        d = {"type": self.name}
        d.update(self.params())
        return d


def _lfo(t0: int, n: int, rate: float, sr: int, phase: float = 0.0) -> np.ndarray:
    return np.sin(2.0 * np.pi * rate * (t0 + np.arange(n)) / sr + phase)


# ----------------------------------------------------------------------------- feedback / delay lines
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


class _ModDelay:
    """Fractional delay line with a per-sample delay curve (chorus, flanger, tape wobble). Exact for any block size."""

    def __init__(self, max_delay: int):
        self.L = int(max_delay) + 8
        self.buf = np.zeros((self.L, 2), dtype=F32)
        self.t = 0

    def run(self, x: np.ndarray, delay: np.ndarray, feedback: float = 0.0) -> np.ndarray:
        n = len(x)
        out = np.empty_like(x)
        pos = 0
        while pos < n:
            m = int(max(min(n - pos, int(delay[pos:].min()) - 1), 1))
            m = min(m, n - pos)
            idx = self.t + pos + np.arange(m)
            rp = idx - delay[pos:pos + m]
            i0 = np.floor(rp).astype(np.int64)
            fr = (rp - i0).astype(F32)[:, None]
            r = self.buf[i0 % self.L] * (1.0 - fr) + self.buf[(i0 + 1) % self.L] * fr
            self.buf[idx % self.L] = x[pos:pos + m] + feedback * r
            out[pos:pos + m] = r
            pos += m
        self.t += n
        return out


class DelayFX(FX):
    name = "delay"
    DEFAULTS = {"time_s": 0.375, "feedback": 0.42, "tone_hz": 5000.0, "mix": 1.0}
    tail_s = 6.0

    def _update(self, first: bool = False) -> None:
        d = max(int(self.time_s * self.sr), 256)
        if first or getattr(self, "_d", None) != d:
            self._d = d
            self.buf = np.zeros((d, 2), dtype=F32)
            self.idx = 0
            self.zi = np.zeros((2, 2))
        self.b, self.a = dsp.biquad_coeffs("lp", self.tone_hz, 0.707, self.sr)

    def sync(self, bpm: float, beats: float = 0.75) -> None:
        self.set(time_s=60.0 / max(bpm, 20.0) * beats)

    def wet(self, x, key=None):
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


class ReverbFX(FX):
    """Small Schroeder/Freeverb style room: 4 combs and 2 allpasses per channel, stereo spread."""
    name = "reverb"
    DEFAULTS = {"room": 0.8, "damp_hz": 6000.0, "mix": 1.0}
    tail_s = 5.0

    def _update(self, first: bool = False) -> None:
        k = self.sr / 44100.0
        g = 0.70 + 0.27 * float(np.clip(self.room, 0.0, 1.0))
        if first:
            self.combs = [[_Comb(int(d * k) + s, g) for d in (1116, 1188, 1277, 1356)] for s in (0, 23)]
            self.aps = [[_Allpass(int(d * k) + s) for d in (556, 441)] for s in (0, 23)]
            self.zi = [np.zeros(2), np.zeros(2)]
        for row in self.combs:
            for c in row:
                c.g = g
        self.b, self.a = dsp.biquad_coeffs("lp", self.damp_hz, 0.707, self.sr)

    def wet(self, x, key=None):
        mono_in = (x.mean(axis=1) * 0.25).astype(F32)
        out = np.empty_like(x)
        for ch in (0, 1):
            y = sum(c.run(mono_in) for c in self.combs[ch])
            for ap in self.aps[ch]:
                y = ap.run(y)
            y, self.zi[ch] = lfilter(self.b, self.a, y, zi=self.zi[ch])
            out[:, ch] = y
        return out


# ----------------------------------------------------------------------------- tone shaping
class FilterFX(FX):
    name = "filter"
    DEFAULTS = {"kind": "lp", "cutoff": 3000.0, "q": 0.9, "mix": 1.0}

    def _update(self, first: bool = False) -> None:
        self.b, self.a = dsp.biquad_coeffs(self.kind, self.cutoff, self.q, self.sr)
        if first:
            self.zi = np.zeros((2, 2))

    def wet(self, x, key=None):
        y, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y.astype(F32)


class EQFX(FX):
    """Three band EQ: low shelf, one parametric peak, high shelf."""
    name = "eq"
    DEFAULTS = {"low_db": 0.0, "low_hz": 120.0, "mid_db": 0.0, "mid_hz": 1000.0, "mid_q": 1.0, "high_db": 0.0, "high_hz": 8000.0, "mix": 1.0}

    def _update(self, first: bool = False) -> None:
        self.stages = []
        for kind, hz, db, q in (("lowshelf", self.low_hz, self.low_db, 0.707), ("peak", self.mid_hz, self.mid_db, self.mid_q),
                                ("highshelf", self.high_hz, self.high_db, 0.707)):
            if abs(db) > 0.01:
                self.stages.append(dsp.biquad_coeffs(kind, hz, q, self.sr, db))
        old = getattr(self, "zis", [])
        self.zis = [old[i] if i < len(old) else np.zeros((2, 2)) for i in range(len(self.stages))]

    def wet(self, x, key=None):
        y = x
        for i, (b, a) in enumerate(self.stages):
            y, self.zis[i] = lfilter(b, a, y, axis=0, zi=self.zis[i])
        return y.astype(F32)


class DriveFX(FX):
    name = "drive"
    DEFAULTS = {"amount": 0.4, "mix": 1.0}

    def wet(self, x, key=None):
        return dsp.drive(x, self.amount)


class CrushFX(FX):
    name = "crush"
    DEFAULTS = {"bits": 8, "downsample": 4, "mix": 1.0}

    def _update(self, first: bool = False) -> None:
        self.downsample = max(int(self.downsample), 1)
        if first:
            self.phase = 0
            self.last = np.zeros(2, dtype=F32)

    def wet(self, x, key=None):
        n, d = len(x), self.downsample
        base = ((self.phase + np.arange(n)) // d) * d - self.phase      # sample-and-hold positions
        held = x[np.clip(base, 0, n - 1)].copy()
        held[base < 0] = self.last
        self.last = x[base[-1]].copy() if base[-1] >= 0 else self.last
        self.phase = (self.phase + n) % d
        levels = 2.0 ** (max(self.bits, 1) - 1)
        return (np.round(held * levels) / levels).astype(F32)


class RingModFX(FX):
    name = "ringmod"
    DEFAULTS = {"freq": 80.0, "mix": 0.6}

    def wet(self, x, key=None):
        return (x * _lfo(self.t, len(x), self.freq, self.sr)[:, None].astype(F32)).astype(F32)


# ----------------------------------------------------------------------------- dynamics
class CompressorFX(FX):
    """Feed-forward compressor with soft knee. Fast-attack / slow-release detector built from two one-pole
    filters (max of the two), so it is exact for any block size. Pass `key` for sidechain ducking."""
    name = "compressor"
    DEFAULTS = {"threshold_db": -18.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 120.0, "knee_db": 6.0, "makeup_db": 0.0, "mix": 1.0}
    needs_key = True

    def _update(self, first: bool = False) -> None:
        a_fast = np.exp(-1.0 / max(self.sr * self.attack_ms / 1000.0, 1.0))
        a_slow = np.exp(-1.0 / max(self.sr * self.release_ms / 1000.0, 1.0))
        self.cf = ([1.0 - a_fast], [1.0, -a_fast])
        self.cs = ([1.0 - a_slow], [1.0, -a_slow])
        if first:
            self.zf = np.zeros(1)
            self.zs = np.zeros(1)
            self.gr_db = 0.0

    def wet(self, x, key=None):
        src = key if key is not None and len(key) == len(x) else x
        det = np.max(np.abs(src), axis=1).astype(np.float64)
        ef, self.zf = lfilter(self.cf[0], self.cf[1], det, zi=self.zf)
        es, self.zs = lfilter(self.cs[0], self.cs[1], det, zi=self.zs)
        env_db = 20.0 * np.log10(np.maximum(np.maximum(ef, es), 1e-9))
        over = env_db - self.threshold_db
        slope = 1.0 / max(self.ratio, 1.0) - 1.0
        k = max(self.knee_db, 1e-6)
        gr = np.where(2 * over < -k, 0.0, np.where(2 * np.abs(over) <= k, slope * (over + k / 2) ** 2 / (2 * k), slope * over))
        g_db = gr + self.makeup_db
        self.gr_db = float(gr.min()) if len(gr) else 0.0
        return (x * (10.0 ** (g_db / 20.0))[:, None].astype(F32)).astype(F32)


# ----------------------------------------------------------------------------- modulation
class ChorusFX(FX):
    name = "chorus"
    DEFAULTS = {"rate": 0.8, "depth_ms": 4.0, "base_ms": 16.0, "mix": 0.5}

    def _update(self, first: bool = False) -> None:
        if first:
            self.line = _ModDelay(int(self.sr * 0.06))

    def wet(self, x, key=None):
        n = len(x)
        base = self.base_ms * self.sr / 1000.0
        dep = self.depth_ms * self.sr / 1000.0
        d = np.empty((n, 1))
        out_l = self.line.run(x, np.full(n, 1.0) * (base + dep * _lfo(self.t, n, self.rate, self.sr)))
        return out_l.astype(F32)


class FlangerFX(FX):
    name = "flanger"
    DEFAULTS = {"rate": 0.25, "depth_ms": 2.5, "base_ms": 3.5, "feedback": 0.5, "mix": 0.5}

    def _update(self, first: bool = False) -> None:
        if first:
            self.line = _ModDelay(int(self.sr * 0.03))

    def wet(self, x, key=None):
        n = len(x)
        base = self.base_ms * self.sr / 1000.0
        dep = self.depth_ms * self.sr / 1000.0
        return self.line.run(x, base + dep * _lfo(self.t, n, self.rate, self.sr), float(np.clip(self.feedback, -0.95, 0.95))).astype(F32)


class PhaserFX(FX):
    """Four first-order allpass stages swept by an LFO. The sweep updates on a fixed 32-sample grid."""
    name = "phaser"
    DEFAULTS = {"rate": 0.35, "depth": 0.8, "center_hz": 800.0, "stages": 4, "mix": 0.5}
    GRID = 32

    def _update(self, first: bool = False) -> None:
        self.stages = int(np.clip(self.stages, 1, 8))
        if first or len(getattr(self, "zis", [])) != self.stages:
            self.zis = [np.zeros((1, 2)) for _ in range(self.stages)]

    def wet(self, x, key=None):
        n = len(x)
        out = np.empty_like(x)
        pos = 0
        while pos < n:
            m = min(n - pos, self.GRID - ((self.t + pos) % self.GRID))
            mid = (self.t + pos) + m / 2.0
            lfo = np.sin(2.0 * np.pi * self.rate * mid / self.sr)
            fc = self.center_hz * 2.0 ** (self.depth * 1.8 * lfo)
            tn = np.tan(np.pi * np.clip(fc, 20.0, self.sr * 0.45) / self.sr)
            c = (tn - 1.0) / (tn + 1.0)
            y = x[pos:pos + m]
            for i in range(self.stages):
                y, self.zis[i] = lfilter([c, 1.0], [1.0, c], y, axis=0, zi=self.zis[i])
            out[pos:pos + m] = y
            pos += m
        return out.astype(F32)


class TapeFX(FX):
    """Tape wobble: slow wow, faster flutter, gentle saturation and a rolled-off top."""
    name = "tape"
    DEFAULTS = {"wow": 0.5, "flutter": 0.3, "drive": 0.25, "tone_hz": 9000.0, "mix": 1.0}

    def _update(self, first: bool = False) -> None:
        if first:
            self.line = _ModDelay(int(self.sr * 0.03))
            self.zi = np.zeros((2, 2))
        self.b, self.a = dsp.biquad_coeffs("lp", self.tone_hz, 0.707, self.sr)

    def wet(self, x, key=None):
        n = len(x)
        base = 0.008 * self.sr
        mod = (0.0012 * self.wow * _lfo(self.t, n, 0.55, self.sr) + 0.0002 * self.flutter * _lfo(self.t, n, 6.3, self.sr, 1.1)) * self.sr
        y = self.line.run(x, base + mod)
        y = dsp.drive(y, self.drive) if self.drive > 0 else y
        y, self.zi = lfilter(self.b, self.a, y, axis=0, zi=self.zi)
        return y.astype(F32)


class VinylFX(FX):
    """Vinyl simulation: crackle, hiss, a rolled-off top. Noise comes from a fixed 512-sample grid so it never depends on block size."""
    name = "vinyl"
    DEFAULTS = {"crackle": 0.5, "hiss": 0.3, "tone_hz": 10000.0, "seed": 0, "mix": 1.0}
    GRID = 512

    def _update(self, first: bool = False) -> None:
        self.b, self.a = dsp.biquad_coeffs("lp", self.tone_hz, 0.707, self.sr)
        self.hb, self.ha = dsp.biquad_coeffs("hp", 700.0, 0.707, self.sr)
        if first:
            self.zi = np.zeros((2, 2))
            self.zh = np.zeros((2, 2))

    def _grid(self, k: int) -> np.ndarray:
        rng = np.random.default_rng([int(self.seed) & 0x7FFFFFFF, k])
        pops = (rng.random((self.GRID, 2)) < 0.0006 * self.crackle * 4).astype(F32) * rng.uniform(-1, 1, (self.GRID, 2)).astype(F32)
        hiss = rng.normal(0, 1, (self.GRID, 2)).astype(F32) * 0.02 * self.hiss
        return (pops * 0.6 * self.crackle + hiss).astype(F32)

    def wet(self, x, key=None):
        n = len(x)
        k0, k1 = self.t // self.GRID, (self.t + n - 1) // self.GRID
        noise = np.concatenate([self._grid(k) for k in range(k0, k1 + 1)])
        off = self.t - k0 * self.GRID
        noise = noise[off:off + n]
        noise, self.zh = lfilter(self.hb, self.ha, noise, axis=0, zi=self.zh)
        y, self.zi = lfilter(self.b, self.a, x + noise, axis=0, zi=self.zi)
        return y.astype(F32)


# ----------------------------------------------------------------------------- optional: Pedalboard (studio effects and plugin hosting)
class PedalboardFX(FX):
    """Wraps a Spotify Pedalboard effect or a VST3/AU plugin file. Optional: pip install pedalboard.
    Not block-size exact (the library chunks internally) and a hosted plugin can crash the process."""
    name = "pedalboard"
    DEFAULTS = {"plugin": "Reverb", "path": "", "params": "", "mix": 1.0}
    tail_s = 5.0

    def _update(self, first: bool = False) -> None:
        import json

        try:
            import pedalboard as pb
        except ImportError as exc:
            raise RuntimeError("the pedalboard effect needs: pip install pedalboard") from exc
        kw = json.loads(self.params_json) if getattr(self, "params_json", "") else {}
        if self.params and isinstance(self.params, str) and self.params.strip():
            kw = json.loads(self.params)
        if self.path:
            p = pb.load_plugin(self.path)
            for k, v in kw.items():
                setattr(p, k, v)
        else:
            cls = getattr(pb, self.plugin, None)
            if cls is None or not isinstance(cls, type):
                raise ValueError("unknown pedalboard effect %r" % self.plugin)
            p = cls(**kw)
        self.board = pb.Pedalboard([p])

    def wet(self, x, key=None):
        y = self.board(np.ascontiguousarray(x.T, dtype=np.float32), self.sr, reset=False)
        return np.ascontiguousarray(y.T, dtype=F32)[: len(x)]


REGISTRY = {c.name: c for c in (DelayFX, ReverbFX, FilterFX, EQFX, DriveFX, CrushFX, RingModFX, CompressorFX, ChorusFX, FlangerFX,
                                PhaserFX, TapeFX, VinylFX, PedalboardFX)}


def make_fx(name: str, sr: int, **kw) -> FX:
    if name not in REGISTRY:
        raise ValueError("unknown effect %r (%s)" % (name, ", ".join(sorted(REGISTRY))))
    return REGISTRY[name](sr, **kw)


def from_spec(spec: dict, sr: int) -> FX:
    d = dict(spec)
    return make_fx(d.pop("type"), sr, **d)


def list_effects() -> List[str]:
    return sorted(REGISTRY)
