"""Pads and banks. A pad is an instrument assignment (source region + recipe + playback settings),
not a button wired to a wav file. The rendered audio is derived data and is never saved."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import dsp
from .config import NUM_BANKS, PADS_PER_BANK, SAMPLE_RATE
from .recipe import Recipe

MODES = ("one_shot", "gate", "loop", "toggle")


@dataclass
class Pad:
    index: int = 0                              # global index: bank * 16 + slot
    name: str = ""
    label: str = ""                             # classifier output: kick, snare, hat, ...
    asset_id: Optional[str] = None
    start: int = 0
    end: int = 0
    recipe: Recipe = field(default_factory=Recipe)
    mode: str = "one_shot"
    reverse: bool = False
    gain_db: float = 0.0
    pan: float = 0.0
    pitch: float = 0.0                          # semitones, applied by varispeed at playback
    fine: float = 0.0                           # cents
    attack: float = 0.0                         # seconds
    decay: float = 0.0
    sustain: float = 1.0
    release: float = 0.008
    choke_group: int = 0                        # pads sharing a non-zero group cut each other off
    vel_sens: float = 0.6                       # 0 = fixed level, 1 = fully velocity controlled
    send_a: float = 0.0                         # delay send
    send_b: float = 0.0                         # reverb send
    lp_cut: float = 0.0                         # baked low-pass (Hz), 0 = off
    hp_cut: float = 0.0                         # baked high-pass (Hz), 0 = off
    midi_note: int = 0
    variants: List[dict] = field(default_factory=list)   # extra {"start","end","recipe"} for round robin
    variation: str = "none"                     # none | round_robin | random
    # derived, not saved
    sample: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    variant_samples: List[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    _key: str = field(default="", repr=False, compare=False)
    _rr: int = field(default=0, repr=False, compare=False)

    def __post_init__(self):
        if not self.midi_note:
            self.midi_note = 36 + (self.index % 64)

    @property
    def empty(self) -> bool:
        return self.asset_id is None or self.end <= self.start

    def clear(self) -> None:
        self.asset_id, self.start, self.end, self.sample, self.name, self.label = None, 0, 0, None, "", ""
        self.recipe = Recipe()
        self.variants, self.variant_samples = [], []

    def render_key(self) -> str:
        return "|".join(str(v) for v in (self.asset_id, self.start, self.end, self.recipe.key(), self.reverse,
                                          self.lp_cut, self.hp_cut, str(self.variants)))

    def render(self, assets: Dict[str, object], sr: int = SAMPLE_RATE, force: bool = False) -> Optional[np.ndarray]:
        """Build (or reuse) the rendered sample: region > recipe > baked filters > reverse > declick."""
        if self.empty or self.asset_id not in assets:
            self.sample = None
            return None
        key = self.render_key()
        if self.sample is not None and key == self._key and not force:
            return self.sample
        self.sample = self._build(assets, self.start, self.end, self.recipe, sr)
        self.variant_samples = []
        for v in self.variants:
            self.variant_samples.append(self._build(assets, int(v["start"]), int(v["end"]), Recipe.from_dict(v.get("recipe")), sr))
        self._key = key
        return self.sample

    def _build(self, assets, start: int, end: int, recipe: Recipe, sr: int) -> np.ndarray:
        a = assets[self.asset_id]
        x = a.region(start, end)
        if len(x) == 0:
            return np.zeros((1, 2), dtype=np.float32)
        y = recipe.apply(x, sr)
        if self.hp_cut > 0:
            y = dsp.apply_filter(y, "hp", self.hp_cut, 0.707, sr)
        if self.lp_cut > 0:
            y = dsp.apply_filter(y, "lp", self.lp_cut, 0.707, sr)
        if self.reverse:
            y = y[::-1]
        return dsp.declick(np.ascontiguousarray(y, dtype=np.float32), sr)

    def pick_sample(self, rng: Optional[np.random.Generator] = None) -> Optional[np.ndarray]:
        pool = [self.sample] + list(self.variant_samples) if self.sample is not None else []
        if not pool:
            return None
        if self.variation == "round_robin" and len(pool) > 1:
            self._rr = (self._rr + 1) % len(pool)
            return pool[self._rr]
        if self.variation == "random" and len(pool) > 1 and rng is not None:
            return pool[int(rng.integers(0, len(pool)))]
        return pool[0]

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "index", "name", "label", "asset_id", "start", "end", "mode", "reverse", "gain_db", "pan", "pitch", "fine",
            "attack", "decay", "sustain", "release", "choke_group", "vel_sens", "send_a", "send_b", "lp_cut", "hp_cut",
            "midi_note", "variation")}
        d["recipe"] = self.recipe.to_dict()
        d["variants"] = self.variants
        return d

    @staticmethod
    def from_dict(d: dict) -> "Pad":
        d = dict(d)
        rec = Recipe.from_dict(d.pop("recipe", None))
        variants = d.pop("variants", [])
        known = set(Pad.__dataclass_fields__.keys())
        p = Pad(**{k: v for k, v in d.items() if k in known})
        p.recipe = rec
        p.variants = variants
        return p


class Bank:
    def __init__(self, bank_index: int = 0, name: str = ""):
        self.index = bank_index
        self.name = name or "ABCD"[bank_index % 4]
        self.pads: List[Pad] = [Pad(index=bank_index * PADS_PER_BANK + i, name="") for i in range(PADS_PER_BANK)]

    def to_dict(self) -> dict:
        return {"index": self.index, "name": self.name, "pads": [p.to_dict() for p in self.pads]}

    @staticmethod
    def from_dict(d: dict) -> "Bank":
        b = Bank(int(d.get("index", 0)), d.get("name", ""))
        b.pads = [Pad.from_dict(p) for p in d.get("pads", [])]
        while len(b.pads) < PADS_PER_BANK:
            b.pads.append(Pad(index=b.index * PADS_PER_BANK + len(b.pads)))
        return b


def make_banks(n: int = NUM_BANKS) -> List[Bank]:
    return [Bank(i) for i in range(n)]
