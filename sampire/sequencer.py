"""Sequencer: integer-tick patterns (960 PPQ), swing, groove, probability, ratchets, parameter locks, and a
sample-accurate scheduler driven by the engine's frame counter (so timing never depends on block size)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import PPQ


@dataclass
class Event:
    tick: int
    pad: int
    vel: float = 1.0
    length: int = 0                      # ticks; > 0 releases gate/loop pads at tick + length
    prob: float = 1.0
    ratchet: int = 1
    cond: str = ""                       # "a:b" = play when (loop_iteration % b) == a - 1
    plock: Optional[dict] = None         # per-hit overrides: pitch, gain_db, pan, reverse, send_a, send_b

    def to_dict(self) -> dict:
        d = {"tick": int(self.tick), "pad": int(self.pad), "vel": round(float(self.vel), 4)}
        if self.length:
            d["length"] = int(self.length)
        if self.prob < 1.0:
            d["prob"] = float(self.prob)
        if self.ratchet > 1:
            d["ratchet"] = int(self.ratchet)
        if self.cond:
            d["cond"] = self.cond
        if self.plock:
            d["plock"] = dict(self.plock)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Event":
        return Event(int(d["tick"]), int(d["pad"]), float(d.get("vel", 1.0)), int(d.get("length", 0)), float(d.get("prob", 1.0)),
                     int(d.get("ratchet", 1)), d.get("cond", ""), d.get("plock"))


class Pattern:
    def __init__(self, name: str = "P1", bars: int = 1, beats_per_bar: int = 4, step_div: int = 16):
        self.name = name
        self.bars = int(bars)
        self.beats_per_bar = int(beats_per_bar)
        self.step_div = int(step_div)            # grid resolution for the step editor (16 = sixteenth notes)
        self.events: List[Event] = []
        self.swing = 0.0                         # 0..0.5, delays odd grid steps by this fraction of a step
        self.humanize_ms = 0.0
        self.humanize_vel = 0.0
        self.pad_offset: Dict[int, int] = {}     # per-pad microtiming in ticks
        self.version = 0

    # -- geometry
    @property
    def length_ticks(self) -> int:
        return self.bars * self.beats_per_bar * PPQ

    @property
    def step_ticks(self) -> int:
        return (4 * PPQ) // self.step_div

    @property
    def steps(self) -> int:
        return self.length_ticks // self.step_ticks

    def touch(self) -> None:
        self.version += 1
        self.events.sort(key=lambda e: (e.tick, e.pad))

    # -- editing
    def add(self, tick: int, pad: int, vel: float = 1.0, **kw) -> Event:
        ev = Event(int(tick) % self.length_ticks, int(pad), float(vel), **kw)
        self.events.append(ev)
        self.touch()
        return ev

    def step_event(self, pad: int, step: int) -> Optional[Event]:
        t = step * self.step_ticks
        for e in self.events:
            if e.pad == pad and abs(e.tick - t) < self.step_ticks // 2:
                return e
        return None

    def toggle_step(self, pad: int, step: int, vel: float = 1.0) -> bool:
        """Toggle a grid step. Returns True when the step is now on."""
        e = self.step_event(pad, step)
        if e is not None:
            self.events.remove(e)
            self.touch()
            return False
        self.add(step * self.step_ticks, pad, vel)
        return True

    def pad_steps(self, pad: int) -> List[int]:
        return sorted({int(round(e.tick / self.step_ticks)) % self.steps for e in self.events if e.pad == pad})

    def remove_pad(self, pad: int) -> None:
        self.events = [e for e in self.events if e.pad != pad]
        self.touch()

    def clear(self) -> None:
        self.events = []
        self.touch()

    def quantize(self, div: int = 16, strength: float = 1.0) -> None:
        g = (4 * PPQ) // div
        for e in self.events:
            target = int(round(e.tick / g)) * g
            e.tick = int(round(e.tick + (target - e.tick) * strength)) % self.length_ticks
        self.touch()

    def euclid(self, pad: int, hits: int, steps: Optional[int] = None, rotate: int = 0, vel: float = 1.0) -> None:
        steps = steps or self.steps
        st = self.length_ticks // steps
        for i in range(steps):
            if (i * hits) % steps < hits:
                self.add(((i + rotate) % steps) * st, pad, vel)

    def reverse_order(self) -> None:
        L = self.length_ticks
        for e in self.events:
            e.tick = (L - e.tick - 1) % L
        self.touch()

    def shift(self, ticks: int) -> None:
        for e in self.events:
            e.tick = (e.tick + ticks) % self.length_ticks
        self.touch()

    def set_bars(self, bars: int) -> None:
        old = self.length_ticks
        self.bars = max(int(bars), 1)
        if self.length_ticks > old:                  # extending repeats the existing material
            base = [copy.copy(e) for e in self.events]
            k = 1
            while old * k < self.length_ticks:
                for e in base:
                    ne = copy.copy(e)
                    ne.tick = e.tick + old * k
                    if ne.tick < self.length_ticks:
                        self.events.append(ne)
                k += 1
        else:
            self.events = [e for e in self.events if e.tick < self.length_ticks]
        self.touch()

    def copy(self, name: Optional[str] = None) -> "Pattern":
        p = Pattern.from_dict(self.to_dict())
        p.name = name or self.name
        return p

    # -- serialisation
    def to_dict(self) -> dict:
        return {"name": self.name, "bars": self.bars, "beats_per_bar": self.beats_per_bar, "step_div": self.step_div,
                "swing": self.swing, "humanize_ms": self.humanize_ms, "humanize_vel": self.humanize_vel,
                "pad_offset": {str(k): v for k, v in self.pad_offset.items()}, "events": [e.to_dict() for e in self.events]}

    @staticmethod
    def from_dict(d: dict) -> "Pattern":
        p = Pattern(d.get("name", "P1"), d.get("bars", 1), d.get("beats_per_bar", 4), d.get("step_div", 16))
        p.swing = float(d.get("swing", 0.0))
        p.humanize_ms = float(d.get("humanize_ms", 0.0))
        p.humanize_vel = float(d.get("humanize_vel", 0.0))
        p.pad_offset = {int(k): int(v) for k, v in d.get("pad_offset", {}).items()}
        p.events = [Event.from_dict(e) for e in d.get("events", [])]
        p.touch()
        return p


# ----------------------------------------------------------------------------- scheduler
class Scheduler:
    """Turns patterns into (frame_offset, kind, event) items for each audio block.

    Time is anchored to the integer frame counter: tick(f) = anchor_tick + (f - anchor_frame) * ticks_per_frame.
    An event fires in the block that contains round(frame_of_event), so the result is identical for any block size."""

    def __init__(self, sr: int, seed: int = 0):
        self.sr = sr
        self.seed = seed
        self.bpm = 92.0
        self.global_swing = 0.0
        self.anchor_frame = 0
        self.anchor_tick = 0.0
        self.timeline: List[Tuple[float, str, int]] = []   # (start_abs_tick, pattern name, serial)
        self.mode = "pattern"                              # pattern | song
        self.pattern_name = "P1"
        self.song: List[Tuple[str, int]] = []
        self._song_i = 0
        self._song_rep = 0
        self._serial = 0
        self.patterns: Dict[str, Pattern] = {}
        self._cache: Dict[Tuple[str, int, float], List[Tuple[int, int, Event, int]]] = {}

    # -- transport
    @property
    def ticks_per_frame(self) -> float:
        return self.bpm * PPQ / (60.0 * self.sr)

    def tick_at(self, frame: float) -> float:
        return self.anchor_tick + (frame - self.anchor_frame) * self.ticks_per_frame

    def frame_of(self, abs_tick: float) -> float:
        return self.anchor_frame + (abs_tick - self.anchor_tick) / self.ticks_per_frame

    def set_bpm(self, bpm: float, at_frame: int) -> None:
        self.anchor_tick = self.tick_at(at_frame)
        self.anchor_frame = at_frame
        self.bpm = float(np.clip(bpm, 20.0, 400.0))

    def start(self, at_frame: int) -> None:
        self.anchor_frame, self.anchor_tick = at_frame, 0.0
        self.timeline, self._serial, self._song_i, self._song_rep = [], 0, 0, 0

    def local_tick(self, frame: int) -> Optional[Tuple[str, float]]:
        """(pattern name, tick inside that pattern) at a frame, or None when nothing is scheduled."""
        t = self.tick_at(frame)
        for start, name, _ in reversed(self.timeline):
            if start <= t and name in self.patterns:
                return name, t - start
        return None

    # -- timeline
    def _next_segment(self) -> Optional[Tuple[str, int]]:
        if self.mode == "song" and self.song:
            name, reps = self.song[self._song_i % len(self.song)]
            self._song_rep += 1
            if self._song_rep >= max(int(reps), 1):
                self._song_rep, self._song_i = 0, (self._song_i + 1) % len(self.song)
            return name, 0
        return self.pattern_name, 0

    def _extend(self, upto_tick: float) -> None:
        end = 0.0
        if self.timeline:
            s, n, _ = self.timeline[-1]
            end = s + self.patterns[n].length_ticks if n in self.patterns else s
        guard = 0
        while end <= upto_tick and guard < 4096:
            guard += 1
            seg = self._next_segment()
            if seg is None or seg[0] not in self.patterns:
                break
            self.timeline.append((end, seg[0], self._serial))
            self._serial += 1
            end += self.patterns[seg[0]].length_ticks
        if len(self.timeline) > 64:
            self.timeline = self.timeline[-32:]

    # -- per-pattern event expansion (swing, offsets, ratchets, note-off)
    def _expanded(self, pat: Pattern) -> List[Tuple[int, int, Event, int]]:
        sw = pat.swing if pat.swing else self.global_swing
        key = (pat.name, pat.version, round(sw, 4))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = []
        L = pat.length_ticks
        st = pat.step_ticks if pat.step_ticks else 240
        for idx, e in enumerate(pat.events):
            base = e.tick + pat.pad_offset.get(e.pad, 0)
            if sw > 0 and st > 0 and e.tick % st == 0 and (e.tick // st) % 2 == 1:
                base += int(round(sw * st))
            n = max(int(e.ratchet), 1)
            span = e.length if e.length else st
            for k in range(n):
                t = base + (span * k) // n
                out.append((int(np.clip(t, 0, L - 1)), idx, e, k))
        out.sort(key=lambda r: (r[0], r[1], r[3]))
        self._cache[key] = out
        if len(self._cache) > 256:
            self._cache.clear()
        return out

    def advance(self, f0: int, n: int) -> List[Tuple[int, str, Event, float]]:
        """Items firing in frames [f0, f0 + n): (offset, 'on'|'off', event, velocity)."""
        tpf = self.ticks_per_frame
        lo, hi = self.tick_at(f0 - 2), self.tick_at(f0 + n + 2)
        self._extend(hi + 1)
        out = []
        for start, name, serial in self.timeline:
            pat = self.patterns.get(name)
            if pat is None:
                continue
            L = pat.length_ticks
            if start + L < lo or start > hi:
                continue
            for t, idx, e, k in self._expanded(pat):
                abs_t = start + t
                if abs_t < lo or abs_t >= hi:
                    continue
                f = int(np.floor(self.frame_of(abs_t) + 0.5))
                if not (f0 <= f < f0 + n):
                    continue
                rng = None
                if e.prob < 1.0 or pat.humanize_ms > 0 or pat.humanize_vel > 0:
                    rng = np.random.default_rng([self.seed & 0x7FFFFFFF, serial, idx, k])
                if e.cond:
                    try:
                        a, b = [int(v) for v in e.cond.split(":")]
                        if serial % max(b, 1) != a - 1:
                            continue
                    except ValueError:
                        pass
                if e.prob < 1.0 and rng.random() > e.prob:
                    continue
                vel = e.vel * (0.85 ** k if e.ratchet > 1 else 1.0)
                if pat.humanize_vel > 0:
                    vel = float(np.clip(vel + rng.normal(0, pat.humanize_vel), 0.05, 1.0))
                if pat.humanize_ms > 0:
                    jit = rng.normal(0, pat.humanize_ms / 1000.0 * self.sr)
                    f = int(np.clip(f + round(jit), f0, f0 + n - 1))
                out.append((f - f0, "on", e, float(np.clip(vel, 0.0, 1.0))))
                if e.length and e.ratchet == 1:
                    fo = int(np.floor(self.frame_of(abs_t + e.length) + 0.5))
                    if f0 <= fo < f0 + n:
                        out.append((fo - f0, "off", e, 0.0))
        out.sort(key=lambda r: (r[0], 0 if r[1] == "off" else 1))
        return out


def ticks_to_bar_beat(tick: float, beats_per_bar: int = 4) -> Tuple[int, int, int]:
    """1-based (bar, beat, sixteenth) for display."""
    beat = int(tick // PPQ)
    return beat // beats_per_bar + 1, beat % beats_per_bar + 1, int((tick % PPQ) // (PPQ // 4)) + 1
