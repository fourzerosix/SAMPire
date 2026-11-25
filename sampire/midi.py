"""MIDI controller framework (optional: pip install sampire[midi]).

- MidiMap: note and CC bindings, defaults + global file + per-project overrides, MIDI learn
- MidiRouter: turns messages into engine commands (pads, velocity, CC, aftertouch, pitch bend, program change)
- ClockFollower: follows MIDI clock, start/stop/continue/song position
- MidiManager: opens every input, hot-plugs devices, survives unplugging, sends MIDI clock out

Everything here only enqueues engine commands, so sound-triggering never waits on the UI. The mido-like backend
is injectable, which is how the test suite exercises hot-plug and clock without hardware."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

from . import params
from .config import PPQ, config_dir

DEFAULT_CC: Dict[int, str] = {1: "pad.lfo_depth", 7: "master", 10: "pad.pan"}
for _i, _name in enumerate(params.CC_PAD_ORDER):
    DEFAULT_CC[16 + _i] = "pad." + _name
AFTERTOUCH = 128


def available() -> bool:
    try:
        import mido  # noqa: F401
        import rtmidi  # noqa: F401
        return True
    except Exception:
        return False


def _mido():
    import mido

    return mido


def list_ports() -> Dict[str, List[str]]:
    try:
        m = _mido()
        return {"inputs": list(m.get_input_names()), "outputs": list(m.get_output_names())}
    except Exception:
        return {"inputs": [], "outputs": []}


class MidiMap:
    """cc and note bindings. Lookup order: project map, global map file, defaults."""

    def __init__(self, project_map: Optional[dict] = None, path: Optional[str] = None):
        self.path = path or str(config_dir() / "midimap.json")
        self.glob: Dict[str, str] = {}
        self.project: Dict[str, str] = dict(project_map or {})
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self.glob = {str(k): v for k, v in json.load(fh).items()}
        except (OSError, ValueError):
            self.glob = {}

    def save_global(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.glob, fh, indent=1)
        except OSError:
            pass

    def cc(self, number: int) -> Optional[str]:
        k = "cc%d" % number
        return self.project.get(k) or self.glob.get(k) or DEFAULT_CC.get(number)

    def note(self, number: int) -> Optional[str]:
        k = "note%d" % number
        return self.project.get(k) or self.glob.get(k)

    def bind(self, key: str, target: str, scope: str = "global") -> None:
        (self.glob if scope == "global" else self.project)[key] = target
        if scope == "global":
            self.save_global()


class MidiRouter:
    def __init__(self, engine, mmap: Optional[MidiMap] = None, on_event: Optional[Callable[[str], None]] = None):
        self.engine = engine
        self.map = mmap or MidiMap(engine.project.midi_map)
        self.on_event = on_event or (lambda s: None)
        self.learning: Optional[str] = None
        self.learn_scope = "global"
        self.follower: Optional["ClockFollower"] = None
        self.last = ""

    def learn(self, target: str, scope: str = "global") -> None:
        self.learning, self.learn_scope = target, scope
        self.on_event("MIDI learn: move a control for %s" % target)

    def _pad_for_note(self, note: int):
        for p in self.engine.project.all_pads():
            if p.midi_note == note:
                return p
        return None

    def _chromatic_pad(self):
        for p in self.engine.project.all_pads():
            if p.chromatic and not p.empty:
                return p
        return None

    def handle(self, msg) -> str:
        e, t = self.engine, getattr(msg, "type", "")
        try:
            if t in ("clock", "start", "stop", "continue", "songpos"):
                if self.follower is not None:
                    return self.follower.handle(msg)
                if t == "start":
                    e.play()
                    return "start"
                if t == "stop":
                    e.stop()
                    return "stop"
                if t == "continue":
                    e.play(False)
                    return "continue"
                return ""
            if t == "note_on" and msg.velocity > 0:
                if self.learning and self.learning.split(".")[0] in ("transport", "bank", "pattern", "record"):
                    self.map.bind("note%d" % msg.note, self.learning, self.learn_scope)
                    r, self.learning = "learned note %d -> %s" % (msg.note, self.learning), None
                    self.on_event(r)
                    return r
                tgt = self.map.note(msg.note)
                if tgt:
                    return self._transport(tgt)
                cp = self._chromatic_pad()
                if cp is not None:
                    e.trigger(cp.index, msg.velocity / 127.0, {"pitch": float(msg.note - 60)})
                    return "chromatic %d" % msg.note
                p = self._pad_for_note(msg.note)
                if p is not None:
                    e.selected_pad = p.index
                    e.trigger(p.index, msg.velocity / 127.0)
                    return "pad %d" % (p.index + 1)
                return ""
            if t in ("note_off", "note_on"):
                cp = self._chromatic_pad()
                p = cp or self._pad_for_note(msg.note)
                if p is not None:
                    e.release(p.index)
                return ""
            if t == "control_change":
                if self.learning:
                    key = "cc%d" % msg.control
                    self.map.bind(key, self.learning, self.learn_scope)
                    r, self.learning = "learned CC%d -> %s" % (msg.control, self.learning), None
                    self.on_event(r)
                    return r
                tgt = self.map.cc(msg.control)
                if tgt:
                    r = params.apply(e, tgt, msg.value / 127.0)
                    self.last = r
                    return r
                return ""
            if t == "pitchwheel":
                e.set_pitch_bend(msg.pitch / 8192.0 * e.pitch_bend_range)
                return "bend"
            if t in ("aftertouch", "polytouch"):
                tgt = self.map.cc(AFTERTOUCH)
                if tgt:
                    return params.apply(e, tgt, msg.value / 127.0)
                return ""
            if t == "program_change":
                e.active_bank = msg.program % max(len(e.project.banks), 1)
                self.on_event("bank %s" % e.project.banks[e.active_bank].name)
                return "program"
        except Exception as exc:  # noqa: BLE001 - a bad message must never take the router down
            return "error: %s" % exc
        return ""

    def _transport(self, tgt: str) -> str:
        e = self.engine
        if tgt == "transport.play":
            e.play()
        elif tgt == "transport.stop":
            e.stop()
        elif tgt == "transport.toggle":
            e.stop() if e.playing else e.play()
        elif tgt == "bank.next":
            e.active_bank = (e.active_bank + 1) % len(e.project.banks)
        elif tgt == "bank.prev":
            e.active_bank = (e.active_bank - 1) % len(e.project.banks)
        elif tgt in ("pattern.next", "pattern.prev"):
            names = list(e.project.patterns)
            i = (names.index(e.project.current_pattern) + (1 if tgt.endswith("next") else -1)) % len(names)
            e.set_pattern(names[i])
        elif tgt == "record.master":
            e.stop_master_recording() if e.rec.active else e.start_master_recording()
        return tgt


class ClockFollower:
    """Follows MIDI clock (24 pulses per quarter). Tempo is the median pulse interval; phase is nudged each beat."""

    def __init__(self, engine, now: Callable[[], float] = time.perf_counter, on_event: Optional[Callable[[str], None]] = None):
        self.engine = engine
        self.now = now
        self.on_event = on_event or (lambda s: None)
        self.intervals: Deque[float] = deque(maxlen=48)
        self.last_t: Optional[float] = None
        self.pulses = 0
        self.running = False
        self.bpm = 0.0

    def handle(self, msg, t: Optional[float] = None) -> str:
        e = self.engine
        t = self.now() if t is None else t
        k = msg.type
        if k == "start":
            self.pulses, self.running, self.last_t = 0, True, None
            self.intervals.clear()
            e.external_sync = True
            e.play(True)
            return "sync start"
        if k == "continue":
            self.running = True
            e.external_sync = True
            e.play(False)
            return "sync continue"
        if k == "stop":
            self.running = False
            e.stop()
            return "sync stop"
        if k == "songpos":
            self.pulses = int(msg.pos) * 6                       # song position counts 16ths = 6 pulses
            e.resync(self.pulses * PPQ / 24.0)
            return "songpos"
        if k == "clock":
            if self.last_t is not None:
                dt = t - self.last_t
                if 0.001 < dt < 0.2:
                    self.intervals.append(dt)
            self.last_t = t
            if len(self.intervals) >= 12:
                bpm = 60.0 / (float(np.median(self.intervals)) * 24.0)
                self.bpm = bpm
                if abs(bpm - e.sched.bpm) > 0.3:
                    e.set_bpm(round(bpm, 1))
            if self.running:
                self.pulses += 1
                if self.pulses % 24 == 0:
                    e.resync(self.pulses * PPQ / 24.0)
            return "clock"
        return ""


class MidiManager:
    """Opens every MIDI input, hot-plugs, and drives MIDI clock out. `backend` defaults to mido."""

    def __init__(self, engine, backend=None, on_event: Optional[Callable[[str], None]] = None, poll_s: float = 2.0,
                 primary: Optional[str] = None):
        self.engine = engine
        self.backend = backend
        self.on_event = on_event or (lambda s: None)
        self.router = MidiRouter(engine, on_event=self.on_event)
        self.follower = ClockFollower(engine, on_event=self.on_event)
        self.follow_clock = False
        self.poll_s = poll_s
        self.primary = primary
        self.inputs: Dict[str, object] = {}
        self.out = None
        self.out_name = ""
        self._run = False
        self._th: Optional[threading.Thread] = None
        self._tc: Optional[threading.Thread] = None
        self.state = "idle"

    def _be(self):
        if self.backend is None:
            self.backend = _mido()
        return self.backend

    def set_follow_clock(self, on: bool) -> None:
        self.follow_clock = on
        self.router.follower = self.follower if on else None
        self.engine.external_sync = on

    def start(self) -> str:
        self._run = True
        try:
            self._be()
        except Exception as exc:
            self.state = "unavailable"
            return "MIDI unavailable: %s" % exc
        self.scan()
        self._th = threading.Thread(target=self._poll, name="sampire-midi", daemon=True)
        self._th.start()
        self._tc = threading.Thread(target=self._clock_out, name="sampire-midiclock", daemon=True)
        self._tc.start()
        return self.status_text()

    def stop(self) -> None:
        self._run = False
        for name in list(self.inputs):
            self._close(name)
        if self.out is not None:
            try:
                self.out.close()
            except Exception:
                pass

    def scan(self) -> None:
        be = self._be()
        try:
            names = list(be.get_input_names())
        except Exception:
            names = []
        for name in list(self.inputs):
            if name not in names:                      # unplugged: close quietly and keep running
                self._close(name)
                self.on_event("MIDI disconnected: %s" % name)
        for name in names:
            if name not in self.inputs and (self.primary is None or self.primary in name or not any(self.primary in n for n in names)):
                try:
                    self.inputs[name] = be.open_input(name, callback=self.router.handle)
                    self.on_event("MIDI connected: %s" % name)
                except Exception as exc:
                    self.on_event("MIDI open failed (%s): %s" % (name, exc))
        self.state = "connected" if self.inputs else "no devices"

    def _close(self, name: str) -> None:
        port = self.inputs.pop(name, None)
        try:
            if port is not None:
                port.close()
        except Exception:
            pass

    def _poll(self) -> None:
        while self._run:
            time.sleep(self.poll_s)
            try:
                self.scan()
            except Exception:
                pass

    def open_output(self, name: Optional[str] = None) -> str:
        be = self._be()
        names = list(be.get_output_names())
        if not names:
            return "no MIDI outputs"
        pick = name if name in names else names[0]
        self.out = be.open_output(pick)
        self.out_name = pick
        return "MIDI clock out: %s" % pick

    def _clock_out(self) -> None:
        be = self.backend
        while self._run:
            try:
                item = self.engine.clock_q.get(timeout=0.2)
            except Exception:
                continue
            if self.out is None or not self.engine.clock_out:
                continue
            try:
                self.out.send(be.Message(item))
            except Exception:
                self.out = None
                self.on_event("MIDI output lost")

    def status(self) -> dict:
        return {"state": self.state, "inputs": list(self.inputs), "output": self.out_name, "follow_clock": self.follow_clock,
                "clock_out": self.engine.clock_out, "learning": self.router.learning, "bpm_in": self.follower.bpm}

    def status_text(self) -> str:
        s = self.status()
        return "MIDI: %s" % (", ".join(s["inputs"]) if s["inputs"] else s["state"])
