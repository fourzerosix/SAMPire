"""The real-time engine: voices, sequencer, group buses, send effects, master bus, and the capture taps.

`render_block(n)` is the single entry point. The audio backend calls it from the device callback; tests and
exports call it offline. Everything the UI wants done arrives through a command queue and is applied at the
top of the next block, so the audio path never waits on the interface.

Signal flow and tap points (what each recorder captures):

    voices -> bus[pad.bus] (pre-effects, "bus raw") -> bus fx chain -> bus gain/mute --+
    voices -> send A -> delay  -> * return A -----------------------------------------+-> * master gain
    voices -> send B -> reverb -> * return B -----------------------------------------+      |
    audition / monitor input ------------------------------------------------------------+      v
                                                                               master fx chain -> soft limiter -> OUT
    taps: skip-back ring, master recorder and resample recorder all read OUT (post-limiter)."""

from __future__ import annotations

import queue
import time
from typing import Callable, List, Optional

import numpy as np

from . import dsp
from .bus import NUM_BUSES, Bus
from .config import BLOCK, MAX_VOICES, PADS_PER_BANK, PPQ, SAMPLE_RATE, SKIPBACK_SECONDS
from .fx import DelayFX, ReverbFX, from_spec
from .project import Project
from .recorder import MasterRecorder, RingBuffer, ThresholdRecorder
from .sequencer import Scheduler
from .voice import Mod, Voice

F32 = np.float32
FAST_RELEASE_S = 0.004
MAXN = 16384


class Engine:
    def __init__(self, project: Optional[Project] = None, sr: int = SAMPLE_RATE, block: int = BLOCK,
                 max_voices: int = MAX_VOICES, skipback_s: float = SKIPBACK_SECONDS, seed: int = 1):
        self.project = project or Project(sr=sr)
        self.sr, self.block, self.max_voices = sr, block, max_voices
        self.steal_policy = "oldest"             # oldest | quietest (protected pads are never stolen)
        self.rng = np.random.default_rng(seed)
        self.cmds: "queue.SimpleQueue" = queue.SimpleQueue()
        self.voices: List[Voice] = []
        self.frame = 0
        self.playing = False
        self.sched = Scheduler(sr, seed)
        self.master_gain = 0.85
        self.fx_a = DelayFX(sr, time_s=0.375, feedback=0.42)
        self.fx_b = ReverbFX(sr, room=0.8)
        self.return_a, self.return_b = 0.6, 0.5
        self.master_fx: list = []
        self.buses: List[Bus] = [Bus(i) for i in range(NUM_BUSES)]
        self.only_bus: Optional[int] = None      # stem rendering: output just this bus (and its sends)
        self.stem_mode = False                   # skip master fx and limiter so stems sum back to the mix
        self._send_idle = 10 ** 9
        self.skip = RingBuffer(int(sr * skipback_s))
        self.in_ring = RingBuffer(int(sr * skipback_s))
        self.rec = MasterRecorder(sr=sr)
        self.resample_rec = MasterRecorder(sr=sr)
        self.in_rec = MasterRecorder(sr=sr)
        self.thresh: Optional[ThresholdRecorder] = None
        self.monitor_input = False
        self._in_block: Optional[np.ndarray] = None
        self.rec_arm = False                     # record pad hits into the current pattern while playing
        self.rec_quantize = 16
        self.latency_comp_ms = 0.0
        self.flash = {}
        self.peak = [0.0, 0.0]
        self.bus_peak = [0.0] * NUM_BUSES
        self.cpu = 0.0
        self.underruns = 0
        self.input_overruns = 0
        self.last_callback = time.perf_counter()
        self.device_state = "running"
        self.selected_pad = 0                    # UI keeps this current; MIDI parameter control targets it
        self.active_bank = 0                     # MIDI program change moves this; the UI follows it
        self.bend = 0.0                          # pitch bend in semitones, applies to all playing voices
        self.pitch_bend_range = 2.0
        self.clock_out = False
        self.clock_q: "queue.SimpleQueue" = queue.SimpleQueue()
        self._clock_pos = 0.0
        self.external_sync = False
        self.live: List[tuple] = []              # (frame, pad, vel) triggers scheduled by rolls
        self._seq = 0
        self._out = np.zeros((MAXN, 2), dtype=F32)
        self._sa = np.zeros((MAXN, 2), dtype=F32)
        self._sb = np.zeros((MAXN, 2), dtype=F32)
        self._bus = [np.zeros((MAXN, 2), dtype=F32) for _ in range(NUM_BUSES)]
        self._aud: List[tuple] = []              # (data, pos) audition playbacks
        self.sync_from_project()

    # -------------------------------------------------------------- settings (effects, buses) <-> project
    def apply_settings(self) -> None:
        s = self.project.fx_settings or {}
        if not s:
            return
        self.master_gain = float(s.get("master_gain", self.master_gain))
        self.return_a = float(s.get("return_a", self.return_a))
        self.return_b = float(s.get("return_b", self.return_b))
        try:
            if "send_a" in s:
                self.fx_a = from_spec(s["send_a"], self.sr)
            if "send_b" in s:
                self.fx_b = from_spec(s["send_b"], self.sr)
        except Exception:
            pass
        self.master_fx = []
        for sp in s.get("master", []):
            try:
                self.master_fx.append(from_spec(sp, self.sr))
            except Exception:
                pass
        self.buses = [Bus.from_spec(i, d, self.sr) for i, d in enumerate((s.get("buses") or [])[:NUM_BUSES])]
        while len(self.buses) < NUM_BUSES:
            self.buses.append(Bus(len(self.buses)))

    def commit_settings(self) -> None:
        """Write the live effect and bus state back into the project so it is saved and undoable."""
        self.project.fx_settings = {
            "master_gain": self.master_gain, "return_a": self.return_a, "return_b": self.return_b,
            "send_a": self.fx_a.spec(), "send_b": self.fx_b.spec(),
            "master": [f.spec() for f in self.master_fx], "buses": [b.spec() for b in self.buses]}

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
        self.apply_settings()

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

    def set_pitch_bend(self, semitones: float) -> None:
        self.cmds.put(("bend", float(semitones)))

    def resync(self, abs_tick: float) -> None:
        self.cmds.put(("resync", float(abs_tick)))

    def roll(self, pad: int, division: int = 16, count: int = 8, vel: float = 0.9, decay: float = 0.93) -> None:
        """Note repeat: `count` hits at 1/division notes from now, velocity fading by `decay` each hit."""
        self.cmds.put(("roll", int(pad), int(division), int(count), float(vel), float(decay)))

    def audition(self, data: np.ndarray, gain: float = 0.8) -> None:
        """Play a buffer straight to the master (browser previews). Replaces the previous audition."""
        self.cmds.put(("audition", dsp.ensure_2d(data), float(gain)))

    def start_master_recording(self) -> None:
        self.rec.start()

    def stop_master_recording(self) -> np.ndarray:
        return self.rec.stop()

    def start_resample(self) -> None:
        self.resample_rec.start()

    def stop_resample(self) -> np.ndarray:
        return self.resample_rec.stop()

    def skipback(self, seconds: float = 30.0) -> np.ndarray:
        return self.skip.read_last(int(seconds * self.sr))

    def input_skipback(self, seconds: float = 30.0) -> np.ndarray:
        return self.in_ring.read_last(int(seconds * self.sr))

    def feed_input(self, block: np.ndarray) -> None:
        x = dsp.ensure_2d(block)
        self.in_ring.write(x)
        self.in_rec.write(x)
        if self.thresh is not None and self.thresh.state != "done":
            self.thresh.push(x)
        if self.monitor_input:
            self._in_block = x

    def diagnostics(self) -> dict:
        mem = 0
        for p in self.project.all_pads():
            if p.sample is not None:
                mem += p.sample.nbytes + sum(v.nbytes for v in p.variant_samples)
        for a in self.project.assets.values():
            mem += a.data.nbytes
        stalled = (time.perf_counter() - self.last_callback) > 1.0 and self.device_state == "running" and self.frame > 0
        return {"sample_rate": self.sr, "block": self.block, "latency_ms": 1000.0 * self.block / self.sr,
                "voices": len(self.voices), "max_voices": self.max_voices, "cpu": self.cpu, "underruns": self.underruns,
                "input_overruns": self.input_overruns, "sample_memory_mb": mem / 1e6, "master_recording": self.rec.active,
                "resampling": self.resample_rec.active, "input_recording": self.in_rec.active,
                "device_state": "stalled" if stalled else self.device_state, "frames": self.frame, "playing": self.playing,
                "bpm": self.sched.bpm, "clock_out": self.clock_out, "external_sync": self.external_sync}

    # -------------------------------------------------------------- internals
    def _drain(self) -> None:
        for _ in range(512):
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
                self.live = []
            elif k == "play":
                self.sync_from_project()
                if c[1] or not self.playing:
                    self.sched.start(self.frame)
                    self._clock_pos = 0.0
                    if self.clock_out:
                        self.clock_q.put("start")
                self.playing = True
            elif k == "stop":
                self.playing = False
                if self.clock_out:
                    self.clock_q.put("stop")
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
            elif k == "bend":
                self.bend = c[1]
                mult = 2.0 ** (self.bend / 12.0)
                for v in self.voices:
                    v.set_bend(mult)
            elif k == "resync":
                err = c[1] - self.sched.tick_at(self.frame)
                tol = 0.020 * self.sched.bpm * PPQ / 60.0          # 20 ms: snap, otherwise slew gently
                if abs(err) > tol:
                    self.sched.anchor_tick, self.sched.anchor_frame = c[1], self.frame
                else:
                    self.sched.anchor_tick = self.sched.tick_at(self.frame) + 0.1 * err
                    self.sched.anchor_frame = self.frame
            elif k == "roll":
                _, pad, div, count, vel, decay = c
                step = int(self.sr * 60.0 / max(self.sched.bpm, 20.0) * 4.0 / max(div, 1))
                for i in range(max(count, 1)):
                    self.live.append((self.frame + i * step, pad, float(np.clip(vel * decay ** i, 0.05, 1.0))))
                self.live.sort()
            elif k == "audition":
                self._aud = [(c[1] * F32(c[2]), 0)]

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
                        g = (4 * PPQ) // self.rec_quantize
                        tick = round(tick / g) * g
                    pat.add(int(tick) % pat.length_ticks, pad_index, vel)

    def _steal_one(self) -> None:
        cand = [v for v in self.voices if not v.releasing and not v.protect]
        if not cand:
            return
        victim = min(cand, key=(lambda x: x.seq) if self.steal_policy == "oldest" else (lambda x: x.loudness()))
        victim.release(int(self.sr * FAST_RELEASE_S))

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
        mod = Mod(pad.filter_kind, pad.filter_cut, pad.filter_q, pad.lfo_target, pad.lfo_rate, pad.lfo_depth, pad.lfo_shape) \
            if (pad.filter_kind or pad.lfo_target) else None
        self._seq += 1
        v = Voice(pad_index, data, rate, gains, int(pad.attack * self.sr), int(pad.decay * self.sr), pad.sustain,
                  int(pad.release * self.sr), pad.mode == "loop", pad.mode, pad.choke_group, float(pl.get("send_a", pad.send_a)),
                  float(pl.get("send_b", pad.send_b)), self._seq, self.frame, tag, int(np.clip(pad.bus, 0, NUM_BUSES - 1)),
                  pad.protect, mod, g, pan, self.sr)
        if self.bend:
            v.set_bend(2.0 ** (self.bend / 12.0))
        live = [x for x in self.voices if not x.releasing]
        if len(live) >= self.max_voices:
            self._steal_one()
        if len(self.voices) > self.max_voices * 2:
            for i, x in enumerate(self.voices):
                if not x.protect:
                    self.voices.pop(i)
                    break
        self.voices.append(v)
        self.flash[pad_index] = self.frame

    def _mix(self, a: int, b: int) -> None:
        if b <= a:
            return
        sa, sb = self._sa[a:b], self._sb[a:b]
        alive = []
        for v in self.voices:
            tgt = self._bus[v.bus][a:b]
            counted = self.only_bus is None or v.bus == self.only_bus
            if v.send_a > 0 or v.send_b > 0:
                tmp = np.zeros_like(tgt)
                ok = v.render_into(tmp)
                tgt += tmp
                if counted:
                    if v.send_a > 0:
                        sa += tmp * F32(v.send_a)
                    if v.send_b > 0:
                        sb += tmp * F32(v.send_b)
                    self._send_idle = 0
            else:
                ok = v.render_into(tgt)
            if ok and not v.dead:
                alive.append(v)
        self.voices = alive

    # -------------------------------------------------------------- the audio path
    def render_block(self, n: int) -> np.ndarray:
        t0 = time.perf_counter()
        self.last_callback = t0
        if n > MAXN:
            n = MAXN
        self._drain()
        for b in self._bus:
            b[:n] = 0.0
        self._sa[:n] = 0.0
        self._sb[:n] = 0.0
        events: list = []
        if self.playing:
            events = [(off, 1, ev, vel, kind) for off, kind, ev, vel in self.sched.advance(self.frame, n)]
        if self.live:
            due = [x for x in self.live if x[0] < self.frame + n]
            self.live = [x for x in self.live if x[0] >= self.frame + n]
            for fr, pad, vel in due:
                events.append((max(fr - self.frame, 0), 0, None, vel, "roll:%d" % pad))
        events.sort(key=lambda e: (e[0], e[1]))
        cursor = 0
        for off, _, ev, vel, kind in events:
            if off > cursor:
                self._mix(cursor, off)
                cursor = off
            if kind == "on":
                self._start_voice(ev.pad, vel, ev.plock, tag=1)
            elif kind == "off":
                for v in self.voices:
                    if v.pad == ev.pad and v.tag == 1 and v.mode in ("gate", "loop") and not v.releasing:
                        v.release()
            else:
                pad_i = int(kind.split(":")[1])
                self._start_voice(pad_i, vel, None)
                if self.rec_arm and self.playing:
                    self._record_roll_hit(pad_i, vel)
        self._mix(cursor, n)
        # clock out (24 pulses per quarter note), derived from the sequencer's tick position
        if self.clock_out and self.playing:
            t0c, t1c = self.sched.tick_at(self.frame), self.sched.tick_at(self.frame + n)
            pulses = int(t1c // (PPQ / 24.0)) - int(t0c // (PPQ / 24.0))
            for _ in range(max(pulses, 0)):
                self.clock_q.put("clock")
        # buses: effects, sidechain, gain
        mix = np.zeros((n, 2), dtype=F32)
        for k, bus in enumerate(self.buses):
            if self.only_bus is not None and k != self.only_bus:
                continue
            y = self._bus[k][:n]
            for fx in bus.fx:
                key = self._bus[bus.sidechain][:n] if (fx.needs_key and bus.sidechain is not None and bus.sidechain != k) else None
                y = fx.process(y, key)
            self.bus_peak[k] = max(float(np.abs(y).max()) * bus.gain if len(y) else 0.0, self.bus_peak[k] * 0.9)
            if not bus.mute:
                mix += y * F32(bus.gain) if bus.gain != 1.0 else y
        if self._send_idle < self.sr * 6:
            wa = self.fx_a.process(self._sa[:n])
            wb = self.fx_b.process(self._sb[:n])
            mix = mix + wa * F32(self.return_a) + wb * F32(self.return_b)
            self._send_idle += n
        if self._aud:
            data, pos = self._aud[0]
            m = min(n, len(data) - pos)
            if m > 0:
                mix[:m] += data[pos:pos + m]
                self._aud = [(data, pos + m)]
            else:
                self._aud = []
        if self._in_block is not None and self.monitor_input and len(self._in_block) == n:
            mix = mix + self._in_block
        mix = mix * F32(self.master_gain)
        if not self.stem_mode:
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

    def _record_roll_hit(self, pad_i: int, vel: float) -> None:
        lt = self.sched.local_tick(self.frame)
        if lt is None:
            return
        name, tick = lt
        pat = self.project.patterns.get(name)
        if pat is not None:
            pat.add(int(tick) % pat.length_ticks, pad_i, vel)

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
