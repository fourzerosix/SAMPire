"""The real-time engine: voices, sequencer, send effects, master bus, and the capture taps.

`render_block(n)` is the single entry point. The audio backend calls it from the device callback; tests and
exports call it offline. Everything the UI wants done arrives through a command queue and is applied at the
top of the next block, so the audio path never waits on the interface."""

from __future__ import annotations

import queue
import time
from typing import Callable, List, Optional

import numpy as np

from . import dsp
from .config import BLOCK, MAX_VOICES, PADS_PER_BANK, SAMPLE_RATE, SKIPBACK_SECONDS
from .fx import DelayFX, ReverbFX
from .project import Project
from .recorder import MasterRecorder, RingBuffer, ThresholdRecorder
from .sequencer import Scheduler
from .voice import Voice

F32 = np.float32
FAST_RELEASE_S = 0.004


class Engine:
    def __init__(self, project: Optional[Project] = None, sr: int = SAMPLE_RATE, block: int = BLOCK,
                 max_voices: int = MAX_VOICES, skipback_s: float = SKIPBACK_SECONDS, seed: int = 1):
        self.project = project or Project(sr=sr)
        self.sr, self.block, self.max_voices = sr, block, max_voices
        self.rng = np.random.default_rng(seed)
        self.cmds: "queue.SimpleQueue" = queue.SimpleQueue()
        self.voices: List[Voice] = []
        self.frame = 0
        self.playing = False
        self.sched = Scheduler(sr, seed)
        self.master_gain = 0.85
        self.fx_a = DelayFX(sr, 0.375, 0.42, mix=1.0)
        self.fx_b = ReverbFX(sr, 0.8, mix=1.0)
        self.return_a, self.return_b = 0.6, 0.5
        self.master_fx: list = []
        self._send_idle = 10 ** 9
        self.skip = RingBuffer(int(sr * skipback_s))
        self.in_ring = RingBuffer(int(sr * skipback_s))
        self.rec = MasterRecorder(sr=sr)
        self.resample_rec = MasterRecorder(sr=sr)
        self.in_rec = MasterRecorder(sr=sr)
        self.thresh: Optional[ThresholdRecorder] = None
        self.monitor_input = False
        self._in_block: Optional[np.ndarray] = None
        self.rec_arm = False                 # record pad hits into the current pattern while playing
        self.rec_quantize = 16               # grid for live recording, 0 = off
        self.latency_comp_ms = 0.0
        self.flash = {}                      # pad -> frame of last trigger (for the UI)
        self.peak = [0.0, 0.0]
        self.cpu = 0.0
        self.underruns = 0
        self.dropped_cmds = 0
        self._seq = 0
        self._ar_cache = {}
        self._out = np.zeros((16384, 2), dtype=F32)
        self._sa = np.zeros((16384, 2), dtype=F32)
        self._sb = np.zeros((16384, 2), dtype=F32)
        self.sync_from_project()

    # -------------------------------------------------------------- control API (thread safe)
    def sync_from_project(self) -> None:
        pr = self.project
        self.sched.patterns = pr.patterns
        self.sched.pattern_name = pr.current_pattern
        self.sched.song = pr.song
        self.sched.mode = pr.mode
        self.sched.global_swing = pr.swing
        self.sched.seed = pr.seed
        self.sched.bpm = pr.bpm

    def set_project(self, project: Project) -> None:
        self.cmds.put(("project", project))

    def trigger(self, pad: int, vel: float = 1.0, plock: Optional[dict] = None) -> None:
        self.cmds.put(("on", int(pad), float(vel), plock))

    def release(self, pad: int) -> None:
        self.cmds.put(("off", int(pad)))

    def stop_all(self) -> None:
        self.cmds.put(("panic",))

    def play(self, from_start: bool = True) -> None:
        self.cmds.put(("play", from_start))

    def stop(self) -> None:
        self.cmds.put(("stop",))

    def set_bpm(self, bpm: float) -> None:
        self.cmds.put(("bpm", float(bpm)))

    def set_pattern(self, name: str) -> None:
        self.cmds.put(("pattern", name))

    def start_master_recording(self) -> None:
        self.rec.start()

    def stop_master_recording(self) -> np.ndarray:
        return self.rec.stop()

    def start_resample(self) -> None:
        self.resample_rec.start()

    def stop_resample(self) -> np.ndarray:
        return self.resample_rec.stop()

    def skipback(self, seconds: float = 30.0) -> np.ndarray:
        """The last `seconds` of master output, from the rolling buffer."""
        return self.skip.read_last(int(seconds * self.sr))

    def input_skipback(self, seconds: float = 30.0) -> np.ndarray:
        return self.in_ring.read_last(int(seconds * self.sr))

    def feed_input(self, block: np.ndarray) -> None:
        """Called from the input side of the audio callback with captured audio."""
        x = dsp.ensure_2d(block)
        self.in_ring.write(x)
        self.in_rec.write(x)
        if self.thresh is not None and self.thresh.state != "done":
            self.thresh.push(x)
        if self.monitor_input:
            self._in_block = x

    # -------------------------------------------------------------- internals
    def _drain(self) -> None:
        for _ in range(256):
            try:
                c = self.cmds.get_nowait()
            except queue.Empty:
                return
            k = c[0]
            if k == "on":
                self._live_trigger(c[1], c[2], c[3])
            elif k == "off":
                for v in self.voices:
                    if v.pad == c[1] and v.mode in ("gate", "loop") and not v.releasing:
                        v.release()
            elif k == "panic":
                for v in self.voices:
                    v.release(int(self.sr * FAST_RELEASE_S))
            elif k == "play":
                self.sync_from_project()
                if c[1] or not self.playing:
                    self.sched.start(self.frame)
                self.playing = True
            elif k == "stop":
                self.playing = False
                for v in self.voices:
                    if v.mode in ("gate", "loop"):
                        v.release()
            elif k == "bpm":
                self.project.bpm = float(c[1])
                self.sched.set_bpm(c[1], self.frame)
            elif k == "pattern":
                self.project.current_pattern = c[1]
                self.sched.pattern_name = c[1]
            elif k == "project":
                self.project = c[1]
                self.voices = []
                self.sync_from_project()

    def _live_trigger(self, pad_index: int, vel: float, plock: Optional[dict]) -> None:
        self._start_voice(pad_index, vel, plock)
        if self.rec_arm and self.playing:
            lt = self.sched.local_tick(self.frame)
            if lt is not None:
                name, tick = lt
                tick += self.latency_comp_ms / 1000.0 * self.sr * self.sched.ticks_per_frame
                pat = self.project.patterns.get(name)
                if pat is not None:
                    if self.rec_quantize:
                        g = (4 * 960) // self.rec_quantize
                        tick = round(tick / g) * g
                    pat.add(int(tick) % pat.length_ticks, pad_index, vel)

    def _start_voice(self, pad_index: int, vel: float, plock: Optional[dict] = None, tag: int = 0) -> None:
        pad = self.project.pad(pad_index)
        data = pad.pick_sample(self.rng)
        if data is None:
            return
        pl = plock or {}
        if pad.mode == "toggle":
            running = [v for v in self.voices if v.pad == pad_index and not v.releasing]
            if running:
                for v in running:
                    v.release()
                return
        if pad.choke_group:
            fr = int(self.sr * FAST_RELEASE_S)
            for v in self.voices:
                if v.group == pad.choke_group and not v.releasing:
                    v.release(fr)
        if pl.get("reverse") and not pad.reverse:
            data = np.ascontiguousarray(data[::-1])
        semis = pad.pitch + float(pl.get("pitch", 0.0)) + pad.fine / 100.0
        rate = 2.0 ** (semis / 12.0)
        sens = float(np.clip(pad.vel_sens, 0.0, 1.0))
        vgain = (1.0 - sens) + sens * float(np.clip(vel, 0.0, 1.0)) ** 1.4
        g = dsp.db_to_lin(pad.gain_db + float(pl.get("gain_db", 0.0))) * vgain
        pan = float(np.clip(pad.pan + float(pl.get("pan", 0.0)), -1.0, 1.0))
        ang = (pan + 1.0) * np.pi / 4.0
        gains = np.array([np.cos(ang), np.sin(ang)], dtype=F32) * F32(g) * F32(np.sqrt(2.0))
        self._seq += 1
        loop = pad.mode == "loop"
        v = Voice(pad_index, data, rate, gains, int(pad.attack * self.sr), int(pad.decay * self.sr), pad.sustain,
                  int(pad.release * self.sr), loop, pad.mode, pad.choke_group, float(pl.get("send_a", pad.send_a)),
                  float(pl.get("send_b", pad.send_b)), self._seq, self.frame, tag)
        live = [x for x in self.voices if not x.releasing]
        if len(live) >= self.max_voices:
            oldest = min(live, key=lambda x: x.seq)
            oldest.release(int(self.sr * FAST_RELEASE_S))
        if len(self.voices) > self.max_voices * 2:
            self.voices.pop(0)
        self.voices.append(v)
        self.flash[pad_index] = self.frame

    def _mix(self, a: int, b: int) -> None:
        if b <= a:
            return
        out, sa, sb = self._out[a:b], self._sa[a:b], self._sb[a:b]
        alive = []
        for v in self.voices:
            if v.send_a > 0 or v.send_b > 0:
                tmp = np.zeros_like(out)
                ok = v.render_into(tmp)
                out += tmp
                if v.send_a > 0:
                    sa += tmp * F32(v.send_a)
                if v.send_b > 0:
                    sb += tmp * F32(v.send_b)
                if v.send_a > 0 or v.send_b > 0:
                    self._send_idle = 0
            else:
                ok = v.render_into(out)
            if ok and not v.dead:
                alive.append(v)
        self.voices = alive

    # -------------------------------------------------------------- the audio path
    def render_block(self, n: int) -> np.ndarray:
        t0 = time.perf_counter()
        if n > len(self._out):
            n = len(self._out)
        self._drain()
        out = self._out[:n]
        out[:] = 0.0
        self._sa[:n] = 0.0
        self._sb[:n] = 0.0
        cursor = 0
        if self.playing:
            for off, kind, ev, vel in self.sched.advance(self.frame, n):
                if off > cursor:
                    self._mix(cursor, off)
                    cursor = off
                if kind == "on":
                    self._start_voice(ev.pad, vel, ev.plock, tag=1)
                else:
                    for v in self.voices:
                        if v.pad == ev.pad and v.tag == 1 and v.mode in ("gate", "loop") and not v.releasing:
                            v.release()
        self._mix(cursor, n)
        mix = out
        if self._send_idle < self.sr * 6:
            wa = self.fx_a.process(self._sa[:n])
            wb = self.fx_b.process(self._sb[:n])
            mix = out + wa * F32(self.return_a) + wb * F32(self.return_b)
            self._send_idle += n
        if self._in_block is not None and self.monitor_input and len(self._in_block) == n:
            mix = mix + self._in_block
        mix = mix * F32(self.master_gain)
        for fx in self.master_fx:
            mix = fx.process(mix)
        mix = dsp.soft_limit(mix).astype(F32, copy=False)
        self.peak[0] = max(float(np.abs(mix[:, 0]).max()), self.peak[0] * 0.92)
        self.peak[1] = max(float(np.abs(mix[:, 1]).max()), self.peak[1] * 0.92)
        self.skip.write(mix)
        self.rec.write(mix)
        self.resample_rec.write(mix)
        self.frame += n
        dt = time.perf_counter() - t0
        self.cpu = 0.9 * self.cpu + 0.1 * (dt / (n / float(self.sr)))
        return mix

    def render(self, seconds: float, block: Optional[int] = None) -> np.ndarray:
        """Offline render of the engine as it stands (used by exports, tests and the headless CLI)."""
        block = block or self.block
        total = int(round(seconds * self.sr))
        out = np.zeros((total, 2), dtype=F32)
        pos = 0
        while pos < total:
            n = min(block, total - pos)
            out[pos:pos + n] = self.render_block(n)
            pos += n
        return out

    @property
    def voice_count(self) -> int:
        return len(self.voices)
