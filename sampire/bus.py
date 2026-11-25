"""Mixer group buses. Pads are assigned to a bus; each bus has gain, mute, an effect chain and an optional
sidechain source (another bus whose pre-effects signal keys any compressor in this bus's chain)."""

from __future__ import annotations

from typing import List, Optional

from .fx import FX, from_spec

NUM_BUSES = 4


class Bus:
    def __init__(self, index: int, name: str = ""):
        self.index = index
        self.name = name or "bus%d" % (index + 1)
        self.gain = 1.0
        self.mute = False
        self.fx: List[FX] = []
        self.sidechain: Optional[int] = None

    def spec(self) -> dict:
        return {"name": self.name, "gain": self.gain, "mute": self.mute, "sidechain": self.sidechain, "fx": [f.spec() for f in self.fx]}

    @staticmethod
    def from_spec(i: int, d: dict, sr: int) -> "Bus":
        b = Bus(i, d.get("name", ""))
        b.gain = float(d.get("gain", 1.0))
        b.mute = bool(d.get("mute", False))
        sc = d.get("sidechain")
        b.sidechain = int(sc) if sc is not None else None
        b.fx = []
        for s in d.get("fx", []):
            try:
                b.fx.append(from_spec(s, sr))
            except Exception:
                pass                                  # a missing optional effect must not stop the project loading
        return b
