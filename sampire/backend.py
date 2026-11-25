"""Audio device backends. sounddevice (PortAudio) is optional: without it SAMPire still runs silent so the
interface, analysis, offline rendering and exports all work on machines with no audio stack (servers, CI, Docker)."""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import numpy as np


def _sd():
    import sounddevice as sd  # raises ImportError/OSError when PortAudio is missing

    return sd


def list_devices() -> List[str]:
    try:
        sd = _sd()
        rows = []
        for i, d in enumerate(sd.query_devices()):
            rows.append("%2d  %-44s in:%d out:%d  %.0f Hz" % (i, d["name"][:44], d["max_input_channels"], d["max_output_channels"],
                                                          d["default_samplerate"]))
        return rows
    except Exception as exc:
        return ["audio devices unavailable: %s" % exc]


class NullBackend:
    """Runs the engine at real-time pace with no sound output (meters, sequencer and recording still work)."""

    name = "silent"

    def __init__(self, engine, reason: str = ""):
        self.engine = engine
        self.reason = reason
        self._run = False
        self._th: Optional[threading.Thread] = None

    def start(self) -> None:
        self._run = True
        self._th = threading.Thread(target=self._loop, name="sampire-null", daemon=True)
        self._th.start()

    def _loop(self) -> None:
        e = self.engine
        period = e.block / float(e.sr)
        nxt = time.perf_counter()
        while self._run:
            try:
                e.render_block(e.block)
            except Exception:
                e.underruns += 1
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()

    def stop(self) -> None:
        self._run = False
        if self._th:
            self._th.join(timeout=1.0)

    def info(self) -> dict:
        return {"name": "silent (%s)" % self.reason if self.reason else "silent", "latency_ms": 0.0, "cpu": self.engine.cpu,
                "underruns": self.engine.underruns, "input": False}


class SoundDeviceBackend:
    def __init__(self, engine, device=None, input_device=None, want_input: bool = False):
        self.engine = engine
        self.device = device
        self.input_device = input_device
        self.want_input = want_input or input_device is not None
        self.stream = None
        self.name = "portaudio"

    def start(self) -> None:
        sd = _sd()
        e = self.engine
        dev = (self.input_device, self.device) if self.want_input else self.device
        if self.want_input:
            self.stream = sd.Stream(samplerate=e.sr, blocksize=e.block, device=dev, channels=2, dtype="float32",
                                    latency="low", callback=self._duplex)
        else:
            self.stream = sd.OutputStream(samplerate=e.sr, blocksize=e.block, device=self.device, channels=2, dtype="float32",
                                          latency="low", callback=self._out)
        self.stream.start()

    def _out(self, outdata, frames, time_info, status):
        if status:
            self.engine.underruns += 1
        try:
            outdata[:] = self.engine.render_block(frames)
        except Exception:
            outdata.fill(0.0)
            self.engine.underruns += 1

    def _duplex(self, indata, outdata, frames, time_info, status):
        if status:
            self.engine.underruns += 1
        try:
            self.engine.feed_input(indata)
            outdata[:] = self.engine.render_block(frames)
        except Exception:
            outdata.fill(0.0)
            self.engine.underruns += 1

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

    def info(self) -> dict:
        lat = 0.0
        try:
            lat = float(self.stream.latency if not isinstance(self.stream.latency, tuple) else self.stream.latency[1]) * 1000.0
        except Exception:
            pass
        return {"name": "portaudio", "latency_ms": lat, "cpu": float(getattr(self.stream, "cpu_load", 0.0)),
                "underruns": self.engine.underruns, "input": self.want_input}


def open_backend(engine, device=None, input_device=None, no_output: bool = False, want_input: bool = False):
    """Return (backend, message). Falls back to the silent backend when no audio stack is available."""
    if no_output:
        nb = NullBackend(engine, "no-output")
        nb.start()
        return nb, "running silent (--no-output)"
    try:
        be = SoundDeviceBackend(engine, device, input_device, want_input)
        be.start()
        return be, "audio ok"
    except Exception as exc:
        msg = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        nb = NullBackend(engine, "no audio device")
        nb.start()
        return nb, "audio unavailable (%s); running silent" % msg[:70]
