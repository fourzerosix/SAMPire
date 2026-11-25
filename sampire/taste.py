"""Learned taste: a tiny online logistic model over human-readable features (hit label, transform used, pattern
style...). Keep or reject what the creative engine makes and its next suggestions drift toward what you keep.
Stored as plain JSON in ~/.config/sampire/taste.json. Deterministic, local, inspectable, and resettable."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import config_dir


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-np.clip(z, -30, 30))))


class Taste:
    def __init__(self, path: Optional[str] = None, lr: float = 0.35, decay: float = 0.002):
        self.path = path or str(config_dir() / "taste.json")
        self.lr, self.decay = lr, decay
        self.w: Dict[str, float] = {}
        self.n = 0
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            self.w, self.n = {k: float(v) for k, v in d.get("w", {}).items()}, int(d.get("n", 0))
        except (OSError, ValueError):
            self.w, self.n = {}, 0

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"w": self.w, "n": self.n}, fh, indent=1, sort_keys=True)
        except OSError:
            pass

    def reset(self) -> None:
        self.w, self.n = {}, 0
        self.save()

    def score(self, feats: Dict[str, float]) -> float:
        return float(sum(self.w.get(k, 0.0) * v for k, v in feats.items()))

    def record(self, feature_sets: Iterable[Dict[str, float]], liked: bool) -> None:
        y = 1.0 if liked else 0.0
        for f in feature_sets:
            p = _sigmoid(self.score(f))
            for k, v in f.items():
                self.w[k] = self.w.get(k, 0.0) * (1.0 - self.decay) + self.lr * (y - p) * v
            self.n += 1
        self.save()

    def preference(self, prefix: str, options: List[str], temperature: float = 1.0) -> np.ndarray:
        """Softmax probabilities over options (feature names are prefix + option). Uniform when nothing was learned."""
        z = np.array([self.w.get(prefix + o, 0.0) for o in options]) / max(temperature, 1e-6)
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def top(self, k: int = 8) -> List[tuple]:
        return sorted(self.w.items(), key=lambda kv: -abs(kv[1]))[:k]
