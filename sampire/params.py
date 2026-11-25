"""Parameter registry: one place that knows every controllable parameter, its range and how to apply it.
The UI, the command line and MIDI (CC, learn, aftertouch) all go through `apply`, so one control can be driven
by mouse, keyboard or hardware."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# name -> (low, high, log scale)
PAD_RANGES: Dict[str, Tuple[float, float, bool]] = {
    "gain_db": (-24.0, 6.0, False), "pan": (-1.0, 1.0, False), "pitch": (-12.0, 12.0, False), "fine": (-50.0, 50.0, False),
    "attack": (0.0, 1.0, False), "decay": (0.0, 2.0, False), "sustain": (0.0, 1.0, False), "release": (0.002, 2.0, True),
    "send_a": (0.0, 1.0, False), "send_b": (0.0, 1.0, False), "filter_cut": (60.0, 16000.0, True), "filter_q": (0.3, 8.0, True),
    "lfo_rate": (0.1, 20.0, True), "lfo_depth": (0.0, 2.0, False), "vel_sens": (0.0, 1.0, False),
}
GLOBAL_RANGES: Dict[str, Tuple[float, float, bool]] = {
    "master": (0.0, 1.2, False), "return_a": (0.0, 1.0, False), "return_b": (0.0, 1.0, False), "bpm": (60.0, 200.0, False),
    "swing": (0.0, 0.5, False), "bus1.gain": (0.0, 1.5, False), "bus2.gain": (0.0, 1.5, False), "bus3.gain": (0.0, 1.5, False),
    "bus4.gain": (0.0, 1.5, False),
}
CC_PAD_ORDER: List[str] = ["gain_db", "pan", "pitch", "fine", "attack", "decay", "sustain", "release", "send_a", "send_b",
                           "filter_cut", "filter_q", "lfo_rate", "lfo_depth", "vel_sens", "gain_db"]


def targets() -> List[str]:
    return sorted(GLOBAL_RANGES) + ["pad." + k for k in PAD_RANGES] + [
        "transport.play", "transport.stop", "transport.toggle", "bank.next", "bank.prev", "pattern.next", "pattern.prev",
        "record.master", "aftertouch.none"]


def _scale(v: float, lo: float, hi: float, log: bool) -> float:
    v = float(np.clip(v, 0.0, 1.0))
    return float(lo * (hi / lo) ** v) if log else float(lo + (hi - lo) * v)


def apply(engine, target: str, v01: float) -> str:
    """Apply a normalised 0..1 value to a target. Returns a short description for the UI log."""
    pr = engine.project
    if target.startswith("pad."):
        attr = target[4:]
        if attr not in PAD_RANGES:
            return ""
        lo, hi, log = PAD_RANGES[attr]
        val = _scale(v01, lo, hi, log)
        pad = pr.pad(engine.selected_pad)
        setattr(pad, attr, val)
        return "pad %d %s = %.2f" % (pad.index + 1, attr, val)
    if target in GLOBAL_RANGES:
        lo, hi, log = GLOBAL_RANGES[target]
        val = _scale(v01, lo, hi, log)
        if target == "master":
            engine.master_gain = val
        elif target == "return_a":
            engine.return_a = val
        elif target == "return_b":
            engine.return_b = val
        elif target == "bpm":
            engine.set_bpm(val)
        elif target == "swing":
            pr.swing = val
            engine.sched.global_swing = val
        elif target.startswith("bus"):
            engine.buses[int(target[3]) - 1].gain = val
        return "%s = %.2f" % (target, val)
    return ""
