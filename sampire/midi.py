"""MIDI controller input (optional: pip install sampire[midi]). Notes trigger pads, CC and MIDI learn map to
engine parameters. Runs in mido's own callback thread and only ever enqueues engine commands."""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

from .config import config_dir


def available() -> bool:
    try:
        import mido  # noqa: F401
        import rtmidi  # noqa: F401
        return True
    except Exception:
        return False


def list_inputs() -> List[str]:
    try:
        import mido

        return list(mido.get_input_names())
    except Exception:
        return []


class MidiMap:
    """note -> pad lookups are by pad.midi_note; CC bindings are learned and saved in the config dir."""

    def __init__(self):
        self.cc: Dict[int, str] = {}
        self.learning: Optional[str] = None
        self._load()

    def _file(self) -> str:
        return str(config_dir() / "midimap.json")

    def _load(self) -> None:
        try:
            with open(self._file(), "r", encoding="utf-8") as fh:
                self.cc = {int(k): v for k, v in json.load(fh).items()}
        except (OSError, ValueError):
            self.cc = {}

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._file()), exist_ok=True)
            with open(self._file(), "w", encoding="utf-8") as fh:
                json.dump({str(k): v for k, v in self.cc.items()}, fh)
        except OSError:
            pass


class MidiInput:
    def __init__(self, engine, port: Optional[str] = None, on_event: Optional[Callable[[str], None]] = None):
        self.engine = engine
        self.port_name = port
        self.map = MidiMap()
        self.on_event = on_event or (lambda s: None)
        self._port = None

    def start(self) -> str:
        import mido

        names = mido.get_input_names()
        if not names:
            return "no MIDI inputs found"
        name = self.port_name if self.port_name in names else names[0]
        self._port = mido.open_input(name, callback=self._handle)
        return "MIDI: " + name

    def stop(self) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None

    def _handle(self, msg) -> None:
        e = self.engine
        try:
            if msg.type == "note_on" and msg.velocity > 0:
                for pad in e.project.all_pads():
                    if pad.midi_note == msg.note:
                        e.trigger(pad.index, msg.velocity / 127.0)
                        break
            elif msg.type in ("note_off", "note_on"):
                for pad in e.project.all_pads():
                    if pad.midi_note == msg.note:
                        e.release(pad.index)
                        break
            elif msg.type == "control_change":
                if self.map.learning:
                    self.map.cc[msg.control] = self.map.learning
                    self.on_event("learned CC%d -> %s" % (msg.control, self.map.learning))
                    self.map.learning = None
                    self.map.save()
                else:
                    self._apply_cc(self.map.cc.get(msg.control), msg.value / 127.0)
            elif msg.type == "start":
                e.play()
            elif msg.type == "stop":
                e.stop()
        except Exception:
            pass

    def _apply_cc(self, target: Optional[str], v: float) -> None:
        e = self.engine
        if target == "master":
            e.master_gain = v
        elif target == "bpm":
            e.set_bpm(60.0 + v * 140.0)
        elif target == "reverb":
            e.return_b = v
        elif target == "delay":
            e.return_a = v
