"""Nondestructive edit recipes. A Recipe is an ordered list of (op, params); it is applied to a region of
an untouched source asset to produce a rendered sample. Same recipe + same seed = same audio."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

import numpy as np

from . import dsp

Op = Dict[str, object]


def _reverse(x, sr, rng, p):
    return x[::-1].copy()


def _gain(x, sr, rng, p):
    return (x * dsp.db_to_lin(float(p.get("db", 0.0)))).astype(np.float32)


def _normalize(x, sr, rng, p):
    return dsp.normalize(x, float(p.get("db", -0.3)))


def _fade(x, sr, rng, p):
    return dsp.fade(x, float(p.get("in_ms", 0)), float(p.get("out_ms", 0)), sr)


def _pitch(x, sr, rng, p):
    return dsp.pitch_shift(x, float(p.get("semitones", 0.0)))


def _stretch(x, sr, rng, p):
    return dsp.time_stretch(x, float(p.get("ratio", 1.0)))


def _rate(x, sr, rng, p):
    """Speed and pitch together (varispeed). ratio > 1 plays faster and higher."""
    return dsp.resample_ratio(x, 1.0 / max(float(p.get("ratio", 1.0)), 0.05))


def _stutter(x, sr, rng, p):
    return dsp.stutter(x, float(p.get("ms", 60)), int(p.get("repeats", 4)), sr)


def _truncate(x, sr, rng, p):
    return x[: max(int(sr * float(p.get("ms", 100)) / 1000.0), 1)].copy()


def _loop(x, sr, rng, p):
    k = max(int(p.get("times", 2)), 1)
    return np.tile(x, (k, 1))


def _filter(x, sr, rng, p):
    return dsp.apply_filter(x, str(p.get("kind", "lp")), float(p.get("cutoff", 4000.0)), float(p.get("q", 0.707)), sr)


def _drive(x, sr, rng, p):
    return dsp.drive(x, float(p.get("amount", 0.5)))


def _bitcrush(x, sr, rng, p):
    return dsp.bitcrush(x, int(p.get("bits", 8)), int(p.get("downsample", 1)))


def _gate(x, sr, rng, p):
    return dsp.gate_pattern(x, str(p.get("steps", "1011")), float(p.get("rate_hz", 16.0)), float(p.get("depth", 1.0)), sr)


def _tape_stop(x, sr, rng, p):
    return dsp.tape_stop(x, float(p.get("ms", 400)), sr)


def _granular(x, sr, rng, p):
    return dsp.granular(x, float(p.get("grain_ms", 40)), float(p.get("jitter", 0.5)), float(p.get("density", 1.0)), int(rng.integers(0, 2**31 - 1)), sr)


def _crackle(x, sr, rng, p):
    return (x + dsp.vinyl_crackle(len(x), sr, level=float(p.get("level", 0.03)), seed=int(rng.integers(0, 2**31 - 1)))).astype(np.float32)


def _mono(x, sr, rng, p):
    m = x.mean(axis=1, keepdims=True)
    return np.repeat(m, 2, axis=1).astype(np.float32)


def _shuffle(x, sr, rng, p):
    """Cut the sample into N equal parts and play them in a new (seeded) order."""
    n = max(int(p.get("parts", 4)), 2)
    edges = np.linspace(0, len(x), n + 1).astype(int)
    parts = [x[a:b] for a, b in zip(edges[:-1], edges[1:])]
    order = rng.permutation(n)
    return np.concatenate([dsp.fade(parts[i], 0.5, 0.5, sr) for i in order]).astype(np.float32)


OPS: Dict[str, Callable] = {
    "reverse": _reverse, "gain": _gain, "normalize": _normalize, "fade": _fade, "pitch": _pitch, "stretch": _stretch,
    "rate": _rate, "stutter": _stutter, "truncate": _truncate, "loop": _loop, "filter": _filter, "drive": _drive,
    "bitcrush": _bitcrush, "gate": _gate, "tape_stop": _tape_stop, "granular": _granular, "crackle": _crackle,
    "mono": _mono, "shuffle": _shuffle,
}


class Recipe:
    def __init__(self, ops: Optional[List[Op]] = None, seed: int = 0):
        self.ops: List[Op] = list(ops or [])
        self.seed = int(seed)

    def add(self, name: str, **params) -> "Recipe":
        if name not in OPS:
            raise ValueError("unknown op %r; available: %s" % (name, ", ".join(sorted(OPS))))
        self.ops.append({"op": name, **params})
        return self

    def copy(self) -> "Recipe":
        return Recipe([dict(o) for o in self.ops], self.seed)

    def apply(self, x: np.ndarray, sr: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        y = dsp.ensure_2d(x)
        for o in self.ops:
            p = {k: v for k, v in o.items() if k != "op"}
            y = dsp.ensure_2d(OPS[str(o["op"])](y, sr, rng, p))
            if len(y) == 0:
                y = np.zeros((1, 2), dtype=np.float32)
        return y

    def key(self) -> str:
        return json.dumps([self.ops, self.seed], sort_keys=True)

    def describe(self) -> str:
        if not self.ops:
            return "original"
        bits = []
        for o in self.ops:
            args = ",".join("%s=%s" % (k, v) for k, v in o.items() if k != "op")
            bits.append("%s(%s)" % (o["op"], args) if args else str(o["op"]))
        return " > ".join(bits)

    def to_dict(self) -> dict:
        return {"ops": self.ops, "seed": self.seed}

    @staticmethod
    def from_dict(d: Optional[dict]) -> "Recipe":
        d = d or {}
        return Recipe([dict(o) for o in d.get("ops", [])], d.get("seed", 0))
