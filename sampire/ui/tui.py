"""The terminal interface. `App` holds all state and logic (testable without a terminal); `run_tui` wires up curses."""

from __future__ import annotations

import curses
import locale
import os
import time
from typing import List, Optional

import numpy as np

from .. import __version__, analysis, creative, dsp, export, slicer
from .. import asset as assetmod
from .. import demo
from ..backend import open_backend
from ..config import PADS_PER_BANK, save_config
from ..engine import Engine
from ..project import Project
from ..recipe import OPS
from ..recorder import ThresholdRecorder
from ..sequencer import Pattern, ticks_to_bar_beat
from . import braille
from .themes import NAMES, THEMES

PAD_KEYS = "1234qwerasdfzxcv"
CTRL = lambda c: ord(c) - 96
HELP = [
    "PADS      1234 qwer asdf zxcv  play pads       arrows  move selection     Tab  pads / step grid",
    "TRANSPORT Space play/stop   +/-  bpm   [ ]  swing   < >  bank   , .  prev/next pattern   ^R/^U redo/undo",
    "CREATE    ^G kit from source   ^P new pattern   ^V vary   ^F chop-flip   ^X clear pattern",
    "CAPTURE   ^T record output   ^B skip-back to pad   ^Y resample start/stop   ^W save   ^O open",
    "STEP GRID Enter toggle step   Backspace remove   the bar under the cursor follows the playhead",
    "COMMANDS  :  opens the command line.  Try  :help  :demo  :open file.wav  :kit 12  :rev  :pitch -3",
    "          :prompt chop into 12 pieces, reverse a few, two 4-bar patterns at 95 bpm with swing",
]
COMMAND_HELP = (
    "open PATH | demo | chop MODE [N] | kit [N] | pattern [STYLE] | vary [AMT] | flip [STRATEGY] | prompt TEXT | bpm N | swing N | "
    "rev | pitch ST | stretch R | rate R | crush [BITS] | drive [AMT] | lp HZ | hp HZ | trunc MS | gain DB | norm | undo-edit | "
    "mode M | choke N | send a|b V | pan V | att S | rel S | name TEXT | clear | fx NAME ... | save [DIR] | load DIR | wav PATH | stems DIR | "
    "pads DIR | midi PATH | seed N | theme NAME | quit")


class App:
    def __init__(self, project: Project, engine: Engine, backend=None, status: str = "", cfg: Optional[dict] = None):
        self.pr, self.eng, self.backend = project, engine, backend
        self.cfg = cfg or {}
        self.sel = 0                     # selected global pad index
        self.bank = 0
        self.focus = "pads"
        self.cur_step = 0
        self.log: List[str] = [status or "ready"]
        self.cmd_mode = False
        self.cmd = ""
        self.history: List[str] = []
        self.hist_i = 0
        self.show_help = False
        self.quit = False
        self.theme = self.cfg.get("theme", "default") if self.cfg.get("theme") in THEMES else "default"
        self.source_id: Optional[str] = next(iter(project.assets), None)
        self.master_recording = False
        self.resampling = False
        self.last_status = status
        self._wave_key = None
        self._wave_rows: List[str] = []
        self.seed = int(self.cfg.get("seed", 1))
        self.pad_rect = {}               # pad index -> (y, x, h, w) for mouse hits
        self.thresh_target: Optional[int] = None

    # ------------------------------------------------------------------ helpers
    def say(self, msg: str) -> None:
        self.log.append(msg)
        self.log = self.log[-50:]

    @property
    def pad(self):
        return self.pr.pad(self.sel)

    @property
    def source(self):
        return self.pr.assets.get(self.source_id) if self.source_id else None

    def reload_engine(self) -> None:
        self.pr.render_all()
        self.eng.sync_from_project()

    def free_pad(self) -> int:
        for p in self.pr.banks[self.bank].pads:
            if p.empty:
                return p.index
        return self.sel

    def assign_audio(self, data: np.ndarray, name: str, origin: str, pad_index: Optional[int] = None) -> None:
        if len(data) < 64:
            self.say("nothing captured")
            return
        a = assetmod.asset_from_array(dsp.normalize(data, -1.0) if origin != "resample" else data, name, self.pr.sr, origin,
                                      generation=(self.source.meta.get("generation", 0) + 1) if self.source else 1)
        a.meta["dirty"] = True
        self.pr.checkpoint()
        self.pr.add_asset(a)
        i = self.sel if pad_index is None else pad_index
        p = self.pr.pad(i)
        p.clear()
        p.asset_id, p.start, p.end, p.name = a.id, 0, a.frames, name[:8]
        p.label = analysis.classify_sample(a.data[: self.pr.sr // 2], self.pr.sr) if a.duration < 2.0 else "loop"
        p.render(self.pr.assets, self.pr.sr)
        self.say("%s -> pad %d (%.2fs)" % (name, i + 1, a.duration))

    # ------------------------------------------------------------------ loading
    def load_source(self, path: str) -> None:
        a = assetmod.load_audio(path)
        self.use_source(a)

    def use_source(self, a) -> None:
        self.pr.checkpoint()
        self.pr.add_asset(a)
        self.source_id = a.id
        a.analysis = analysis.analyze(a.data, a.sr)
        if a.analysis["bpm_confidence"] > 0.2:
            self.pr.bpm = a.analysis["bpm"]
            self.eng.set_bpm(self.pr.bpm)
        self.say("%s: %.1fs  %.1f BPM  %s" % (a.name, a.duration, a.analysis["bpm"], a.analysis["key"]))
        self.make_kit()

    def make_kit(self, n: int = 16, mode: str = "hybrid", experimental: float = 0.3) -> None:
        a = self.source
        if a is None:
            self.say("no source loaded (:open FILE or :demo)")
            return
        self.seed += 1
        self.pr.checkpoint()
        rep = creative.generate_kit(self.pr, a, creative.KitParams(pads=n, mode=mode, experimental=experimental, bank=self.bank, seed=self.seed))
        if rep.notes:
            self.say(rep.notes[0])
        else:
            self.say("kit: %d pads from %d slices, %d transformed (seed %d)" % (rep.chosen, rep.slices, rep.transformed, self.seed))
        self.reload_engine()

    def make_pattern(self, style: str = "break") -> None:
        self.seed += 1
        self.pr.checkpoint()
        pt = creative.generate_pattern(self.pr, self.bank, max(self.pr.pattern.bars, 2), style, 0.5, 0.7, self.pr.swing, self.seed)
        self.pr.patterns[pt.name] = pt
        self.pr.current_pattern = pt.name
        self.reload_engine()
        self.say("pattern %s: %d events (%s, seed %d)" % (pt.name, len(pt.events), style, self.seed))

    # ------------------------------------------------------------------ commands
    def command(self, line: str) -> None:
        line = line.strip().lstrip(":")
        if not line:
            return
        self.history.append(line)
        w = line.split(None, 1)
        c, rest = w[0].lower(), (w[1] if len(w) > 1 else "")
        args = rest.split()
        try:
            self._command(c, rest, args)
        except Exception as exc:  # noqa: BLE001 - the UI must survive any bad command
            self.say("error: %s" % exc)

    def _edit_pad(self, op: str, **params) -> None:
        p = self.pad
        if p.empty:
            self.say("pad %d is empty" % (p.index + 1))
            return
        self.pr.checkpoint()
        p.recipe.add(op, **params)
        p.render(self.pr.assets, self.pr.sr, force=True)
        self.say("pad %d: %s" % (p.index + 1, p.recipe.describe()))

    def _command(self, c: str, rest: str, a: List[str]) -> None:
        f = lambda i, d: float(a[i]) if len(a) > i else d
        p = self.pad
        if c in ("q", "quit", "exit"):
            self.quit = True
        elif c == "help":
            self.say(COMMAND_HELP)
            self.show_help = True
        elif c == "demo":
            self.use_source(assetmod.asset_from_array(demo.drum_break(92.0, 2, with_melody=True), "demo-break", origin="demo"))
        elif c in ("open", "o"):
            self.load_source(os.path.expanduser(rest.strip()))
        elif c == "chop":
            self.make_kit(int(f(1, 16)), a[0] if a else "hybrid")
        elif c == "kit":
            self.make_kit(int(f(0, 16)))
        elif c == "pattern":
            self.make_pattern(a[0] if a else "break")
        elif c == "vary":
            self.seed += 1
            self.pr.checkpoint()
            v = creative.vary_pattern(self.pr.pattern, f(0, 0.3), self.seed)
            self.pr.patterns[v.name] = v
            self.pr.current_pattern = v.name
            self.reload_engine()
            self.say("variation %s (%d events)" % (v.name, len(v.events)))
        elif c == "flip":
            self.seed += 1
            self.pr.checkpoint()
            fl = creative.rearrange_chops(self.pr, self.bank, self.pr.pattern.bars, a[0] if a else "shuffle", self.seed)
            self.pr.patterns[fl.name] = fl
            self.pr.current_pattern = fl.name
            self.reload_engine()
            self.say("chop-flip %s" % fl.name)
        elif c == "prompt":
            plan = creative.parse_instruction(rest)
            self.say("plan: %s" % {k: v for k, v in plan.items() if v and k != "notes"})
            for n in plan["notes"]:
                self.say(n)
            if self.source is None:
                self.say("load a source first")
                return
            self.seed += 1
            creative.run_plan(self.pr, self.source, plan, self.seed, self.bank)
            self.eng.set_bpm(self.pr.bpm)
            self.reload_engine()
            self.say("plan applied (:undo-edit is ^U)")
        elif c == "bpm":
            self.pr.bpm = float(np.clip(f(0, self.pr.bpm), 20, 400))
            self.eng.set_bpm(self.pr.bpm)
        elif c == "swing":
            self.pr.swing = float(np.clip(f(0, 0) / 100.0, 0, 0.5))
            self.eng.sched.global_swing = self.pr.swing
            self.pr.pattern.swing = 0.0
        elif c == "rev":
            self._edit_pad("reverse")
        elif c == "pitch":
            self._edit_pad("pitch", semitones=f(0, 0))
        elif c == "stretch":
            self._edit_pad("stretch", ratio=f(0, 1.0))
        elif c == "rate":
            self._edit_pad("rate", ratio=f(0, 1.0))
        elif c == "crush":
            self._edit_pad("bitcrush", bits=int(f(0, 8)), downsample=int(f(1, 2)))
        elif c == "drive":
            self._edit_pad("drive", amount=f(0, 0.5))
        elif c in ("lp", "hp"):
            self._edit_pad("filter", kind=c, cutoff=f(0, 2000))
        elif c == "trunc":
            self._edit_pad("truncate", ms=f(0, 100))
        elif c == "gain":
            self._edit_pad("gain", db=f(0, 0))
        elif c == "norm":
            self._edit_pad("normalize")
        elif c == "stutter":
            self._edit_pad("stutter", ms=f(0, 60), repeats=int(f(1, 4)))
        elif c in ("undo-edit", "undo"):
            self.undo()
        elif c == "redo":
            self.redo()
        elif c == "mode":
            m = a[0] if a else "one_shot"
            if m not in ("one_shot", "gate", "loop", "toggle"):
                raise ValueError("mode: one_shot gate loop toggle")
            p.mode = m
        elif c == "choke":
            p.choke_group = int(f(0, 0))
        elif c == "send":
            if a and a[0] == "a":
                p.send_a = f(1, 0)
            else:
                p.send_b = f(1, 0)
        elif c == "pan":
            p.pan = float(np.clip(f(0, 0), -1, 1))
        elif c in ("att", "attack"):
            p.attack = f(0, 0)
        elif c in ("rel", "release"):
            p.release = f(0, 0.008)
        elif c == "name":
            p.name = rest.strip()[:8]
        elif c == "clear":
            self.pr.checkpoint()
            p.clear()
            self.eng.stop_all()
        elif c == "fx":
            self._fx(a)
        elif c == "save":
            self.say("saved " + self.pr.save(rest.strip() or self.pr.path or os.path.join(os.getcwd(), "sampire-project")))
        elif c == "load":
            self.pr = Project.load(os.path.expanduser(rest.strip()))
            self.eng.set_project(self.pr)
            self.source_id = next(iter(self.pr.assets), None)
            self.say("loaded " + self.pr.name)
        elif c == "wav":
            y = export.render_pattern(self.pr, None, 1)
            self.say("wrote " + export.export_audio(os.path.expanduser(rest.strip() or "sampire_out.wav"), y, self.pr.sr))
        elif c == "stems":
            self.say("%d stems written" % len(export.export_stems(self.pr, os.path.expanduser(rest.strip() or "sampire_stems"))))
        elif c == "pads":
            self.say("%d pad wavs written" % len(export.export_pads(self.pr, os.path.expanduser(rest.strip() or "sampire_pads"))))
        elif c == "midi":
            self.say("wrote " + export.pattern_to_midi(self.pr, self.pr.pattern, os.path.expanduser(rest.strip() or "sampire.mid")))
        elif c == "seed":
            self.seed = int(f(0, 1))
        elif c == "theme":
            if a and a[0] in THEMES:
                self.theme = a[0]
                self.cfg["theme"] = a[0]
                self.apply_theme()
        elif c == "sample":
            secs = f(0, 4.0)
            self.thresh_target = self.sel
            self.eng.thresh = ThresholdRecorder(self.eng.sr, -38.0, 40.0, fixed_frames=int(secs * self.eng.sr))
            self.say("armed: sampling %.1fs into pad %d when input crosses -38 dB" % (secs, self.sel + 1))
        else:
            self.say("unknown command %r  (:help)" % c)

    def _fx(self, a: List[str]) -> None:
        from ..fx import make_fx

        if not a:
            self.say("fx delay|reverb|filter|drive|crush [on|off] | fx clear")
            return
        if a[0] == "clear":
            self.eng.master_fx = []
        elif a[0] in ("delay", "reverb"):
            if len(a) > 1:
                (setattr(self.eng, "return_a", float(a[1])) if a[0] == "delay" else setattr(self.eng, "return_b", float(a[1])))
        else:
            kw = {}
            if a[0] == "filter" and len(a) > 1:
                kw = {"cutoff": float(a[1])}
            self.eng.master_fx = [fx for fx in self.eng.master_fx if fx.name != a[0]] + [make_fx(a[0], self.eng.sr, **kw)]
        self.say("master fx: %s" % ", ".join(x.name for x in self.eng.master_fx) or "none")

    def undo(self) -> None:
        if self.pr.undo():
            self.reload_engine()
            self.say("undo")
        else:
            self.say("nothing to undo")

    def redo(self) -> None:
        if self.pr.redo():
            self.reload_engine()
            self.say("redo")

    # ------------------------------------------------------------------ keys
    def handle_key(self, ch: int) -> None:
        if self.cmd_mode:
            return self._cmd_key(ch)
        if ch in (-1,):
            return
        if ch == curses.KEY_MOUSE:
            return self._mouse()
        if ch in (3, CTRL("q")):
            self.quit = True
        elif ch == ord(":") or ch == CTRL("e"):
            self.cmd_mode, self.cmd = True, ""
        elif ch == ord("/") or ch == ord("?"):
            self.show_help = not self.show_help
        elif ch == 27:
            self.show_help = False
        elif ch == ord(" "):
            if self.eng.playing:
                self.eng.stop()
            else:
                self.eng.play()
        elif ch == 9:
            self.focus = "seq" if self.focus == "pads" else "pads"
        elif ch in (ord("+"), ord("=")):
            self.command("bpm %.1f" % (self.pr.bpm + 1))
        elif ch in (ord("-"), ord("_")):
            self.command("bpm %.1f" % (self.pr.bpm - 1))
        elif ch == ord("["):
            self.command("swing %d" % max(self.pr.swing * 100 - 2, 0))
        elif ch == ord("]"):
            self.command("swing %d" % min(self.pr.swing * 100 + 2, 50))
        elif ch in (ord("<"), ord(">")):
            self.bank = (self.bank + (1 if ch == ord(">") else -1)) % len(self.pr.banks)
            self.sel = self.bank * PADS_PER_BANK + (self.sel % PADS_PER_BANK)
        elif ch in (ord(","), ord(".")):
            names = list(self.pr.patterns)
            i = (names.index(self.pr.current_pattern) + (1 if ch == ord(".") else -1)) % len(names)
            self.pr.current_pattern = names[i]
            self.eng.set_pattern(names[i])
        elif ch == CTRL("g"):
            self.make_kit()
        elif ch == CTRL("p"):
            self.make_pattern()
        elif ch == CTRL("v"):
            self.command("vary")
        elif ch == CTRL("f"):
            self.command("flip")
        elif ch == CTRL("x"):
            self.pr.checkpoint()
            self.pr.pattern.clear()
        elif ch == CTRL("u"):
            self.undo()
        elif ch == CTRL("r"):
            self.redo()
        elif ch == CTRL("w"):
            self.command("save")
        elif ch == CTRL("o"):
            self.cmd_mode, self.cmd = True, "open "
        elif ch == CTRL("l"):
            self.theme = NAMES[(NAMES.index(self.theme) + 1) % len(NAMES)]
            self.apply_theme()
        elif ch == CTRL("t"):
            if self.master_recording:
                y = self.eng.stop_master_recording()
                self.master_recording = False
                os.makedirs("recordings", exist_ok=True)
                path = os.path.join("recordings", time.strftime("take-%Y%m%d-%H%M%S.wav"))
                export.export_audio(path, y, self.pr.sr)
                self.say("saved %s (%.1fs)" % (path, len(y) / self.pr.sr))
            else:
                self.eng.start_master_recording()
                self.master_recording = True
                self.say("recording output...")
        elif ch == CTRL("b"):
            self.assign_audio(self.eng.skipback(30.0), "skipbk", "skipback", self.free_pad())
        elif ch == CTRL("y"):
            if self.resampling:
                self.resampling = False
                self.assign_audio(self.eng.stop_resample(), "resamp", "resample", self.free_pad())
            else:
                self.eng.start_resample()
                self.resampling = True
                self.say("resampling the master bus; press ^Y again to stop")
        elif ch in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT):
            self._arrow(ch)
        elif ch in (10, 13, curses.KEY_ENTER):
            if self.focus == "seq":
                on = self.pr.pattern.toggle_step(self.sel, self.cur_step)
                if on:
                    self.eng.trigger(self.sel, 0.9)
            else:
                self.eng.trigger(self.sel, 1.0)
        elif ch in (curses.KEY_BACKSPACE, 127, 8) and self.focus == "seq":
            e = self.pr.pattern.step_event(self.sel, self.cur_step)
            if e:
                self.pr.pattern.events.remove(e)
                self.pr.pattern.touch()
        elif 0 < ch < 256 and chr(ch).lower() in PAD_KEYS:
            slot = PAD_KEYS.index(chr(ch).lower())
            gi = self.bank * PADS_PER_BANK + slot
            self.sel = gi
            self.eng.trigger(gi, 1.0 if chr(ch).islower() else 0.55)
            if self.focus == "seq":
                pass

    def _arrow(self, ch: int) -> None:
        if self.focus == "pads":
            slot = self.sel % PADS_PER_BANK
            r, c = divmod(slot, 4)
            if ch == curses.KEY_UP:
                r = (r - 1) % 4
            elif ch == curses.KEY_DOWN:
                r = (r + 1) % 4
            elif ch == curses.KEY_LEFT:
                c = (c - 1) % 4
            else:
                c = (c + 1) % 4
            self.sel = self.bank * PADS_PER_BANK + r * 4 + c
        else:
            steps = self.pr.pattern.steps
            if ch == curses.KEY_LEFT:
                self.cur_step = (self.cur_step - 1) % steps
            elif ch == curses.KEY_RIGHT:
                self.cur_step = (self.cur_step + 1) % steps
            else:
                slot = (self.sel % PADS_PER_BANK + (1 if ch == curses.KEY_DOWN else -1)) % PADS_PER_BANK
                self.sel = self.bank * PADS_PER_BANK + slot

    def _cmd_key(self, ch: int) -> None:
        if ch in (27,):
            self.cmd_mode = False
        elif ch in (10, 13, curses.KEY_ENTER):
            self.cmd_mode = False
            self.command(self.cmd)
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            self.cmd = self.cmd[:-1]
        elif ch == curses.KEY_UP and self.history:
            self.hist_i = (self.hist_i + 1) % len(self.history)
            self.cmd = self.history[-self.hist_i]
        elif 32 <= ch < 127:
            self.cmd += chr(ch)

    def _mouse(self) -> None:
        try:
            _, mx, my, _, bstate = curses.getmouse()
        except curses.error:
            return
        if bstate & (curses.BUTTON1_PRESSED | curses.BUTTON1_CLICKED):
            for gi, (y, x, h, w) in self.pad_rect.items():
                if y <= my < y + h and x <= mx < x + w:
                    self.sel = gi
                    self.eng.trigger(gi, 1.0)

    # ------------------------------------------------------------------ periodic
    def tick(self) -> None:
        t = self.eng.thresh
        if t is not None and t.state == "done":
            self.eng.thresh = None
            self.assign_audio(t.result(), "sample", "record", self.thresh_target)

    # ------------------------------------------------------------------ drawing
    def apply_theme(self) -> None:
        fg, acc, hot, ok, dim = THEMES[self.theme]
        try:
            curses.use_default_colors()
            for i, col in enumerate((fg, acc, hot, ok, dim), start=1):
                curses.init_pair(i, col if curses.COLORS > 8 or col < 8 else -1, -1)
        except curses.error:
            pass

    def _put(self, s, y, x, text, attr=0, maxw=None):
        h, w = s.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        n = min(len(text), (maxw if maxw is not None else w - x), w - x - (1 if y == h - 1 else 0))
        if n > 0:
            try:
                s.addnstr(y, x, text, n, attr)
            except curses.error:
                pass

    def draw(self, s) -> None:
        s.erase()
        h, w = s.getmaxyx()
        C = lambda i: curses.color_pair(i)
        ACC, HOT, OK, DIM = C(2), C(3), C(4), C(5)
        if h < 24 or w < 80:
            self._put(s, 0, 0, "terminal too small: need 80x24 (have %dx%d)" % (w, h), HOT)
            s.refresh()
            return
        e, pr = self.eng, self.pr
        info = self.backend.info() if self.backend else {"name": "none", "latency_ms": 0, "cpu": 0, "underruns": 0}
        lt = e.sched.local_tick(e.frame) if e.playing else None
        pos = "%d.%d.%d" % ticks_to_bar_beat(lt[1]) if lt else "-.-.-"
        play = "PLAY" if e.playing else "STOP"
        rec = " REC" if e.rec_arm else ""
        # header
        self._put(s, 0, 0, " SAMPIRE %s " % __version__, ACC | curses.A_BOLD)
        head = " %s %s  %.1f BPM  swing %d%%  pat %s %s  bank %s  src %s" % (
            play, rec, pr.bpm, round(pr.swing * 100), pr.current_pattern, pos, pr.banks[self.bank].name,
            (self.source.name[:14] if self.source else "none"))
        self._put(s, 0, 17, head, (OK if e.playing else DIM) | curses.A_BOLD)
        cpu = max(e.cpu, info.get("cpu", 0))
        right = "%s %3.0f%% vox %2d xr %d " % (info["name"][:18], cpu * 100, e.voice_count, e.underruns)
        self._put(s, 0, max(w - len(right) - 1, 0), right, DIM)
        lw = 36
        # pad grid
        self._put(s, 1, 0, " PADS %s " % pr.banks[self.bank].name, ACC)
        self.pad_rect = {}
        for slot in range(PADS_PER_BANK):
            r, c = divmod(slot, 4)
            gi = self.bank * PADS_PER_BANK + slot
            pad = pr.pad(gi)
            y, x = 2 + r * 4, c * 9
            flash = (e.frame - e.flash.get(gi, -10 ** 9)) < 0.12 * e.sr
            attr = (HOT | curses.A_REVERSE) if flash else (ACC | curses.A_BOLD if gi == self.sel else (0 if not pad.empty else DIM))
            self._put(s, y, x, "+-------+", attr)
            self._put(s, y + 1, x, "|%s %-5s|" % (PAD_KEYS[slot].upper(), (pad.name or "")[:5]), attr)
            self._put(s, y + 2, x, "|%-7s|" % ((pad.label or ("." if pad.empty else "?"))[:7]), attr)
            self.pad_rect[gi] = (y, x, 3, 9)
        # waveform
        wx, ww = lw, w - lw - 1
        pad = self.pad
        title = " pad %d %s  %s  %s " % (pad.index + 1, pad.name or "-", pad.mode, pad.recipe.describe()[:ww - 30])
        self._put(s, 1, wx, title, ACC)
        wh = 7
        key = (pad.render_key(), ww, id(pad.sample))
        if key != self._wave_key:
            self._wave_key = key
            if pad.sample is not None:
                self._wave_rows = braille.waveform(pad.sample, ww - 1, wh).rows()
            elif self.source is not None:
                c = braille.waveform(self.source.data, ww - 1, wh)
                a = self.source.analysis
                braille.markers(c, a.get("onsets", [])[:200], self.source.frames)
                self._wave_rows = c.rows()
            else:
                self._wave_rows = braille.waveform(np.zeros((0, 2)), ww - 1, wh).rows()
        for i, row in enumerate(self._wave_rows):
            self._put(s, 2 + i, wx, row, OK if pad.sample is not None else DIM)
        for v in e.voices:
            if v.pad == self.sel and len(v.data):
                x = wx + int(min(v.pos / len(v.data), 1.0) * (ww - 2))
                for i in range(wh):
                    self._put(s, 2 + i, x, "|", HOT)
        # step grid
        pat = pr.pattern
        gy = 2 + wh + 1
        steps = pat.steps
        page = 16
        start = (self.cur_step // page) * page if self.focus == "seq" else (int(lt[1] // pat.step_ticks) // page * page if lt else 0)
        self._put(s, gy - 1, wx, " STEPS %s %d bars  steps %d-%d/%d " % (pat.name, pat.bars, start + 1, min(start + page, steps), steps), ACC)
        play_step = int(lt[1] // pat.step_ticks) if lt else -1
        rows = min(PADS_PER_BANK, h - gy - 5)
        base = self.bank * PADS_PER_BANK
        for r in range(rows):
            gi = base + r
            on = set(pat.pad_steps(gi))
            lab = "%2d %-5s" % (r + 1, (pr.pad(gi).name or "")[:5])
            self._put(s, gy + r, wx, lab, ACC if gi == self.sel else DIM)
            for k in range(page):
                st = start + k
                if st >= steps:
                    break
                ch = "#" if st in on else ("." if st % 4 else ":")
                attr = 0
                if st == play_step:
                    attr = HOT | curses.A_REVERSE
                elif self.focus == "seq" and gi == self.sel and st == self.cur_step:
                    attr = ACC | curses.A_REVERSE
                elif st in on:
                    attr = OK | curses.A_BOLD
                else:
                    attr = DIM
                self._put(s, gy + r, wx + 9 + k * 2, ch + " ", attr)
        # meters
        for i, pk in enumerate(e.peak):
            n = int(min(pk, 1.0) * 20)
            self._put(s, 2 + wh - 1 + i, 0, "", 0)
        mrow = 2 + 16
        self._put(s, mrow, 0, "L " + "#" * int(min(e.peak[0], 1.0) * 30), OK if e.peak[0] < 0.95 else HOT)
        self._put(s, mrow + 1, 0, "R " + "#" * int(min(e.peak[1], 1.0) * 30), OK if e.peak[1] < 0.95 else HOT)
        # status area
        sa = self.source.analysis if self.source else {}
        det = " source: %s  %.1f BPM  %s  %d onsets | master rec %s | resample %s | undo %d" % (
            (self.source.name if self.source else "-"), sa.get("bpm", 0), sa.get("key", "-"), len(sa.get("onsets", [])),
            "ON" if self.master_recording else "off", "ON" if self.resampling else "off", len(pr._undo))
        self._put(s, h - 5, 0, det, DIM)
        for i, m in enumerate(self.log[-2:]):
            self._put(s, h - 4 + i, 0, " " + m, 0)
        if self.cmd_mode:
            self._put(s, h - 2, 0, ":" + self.cmd + "_", ACC | curses.A_BOLD)
        else:
            self._put(s, h - 2, 0, " 1234 qwer asdf zxcv pads  Space play  Tab steps  ^G kit  ^P pattern  ^T rec  : commands  ? help", DIM)
        if self.show_help:
            for i, line in enumerate(HELP):
                self._put(s, 3 + i, 2, " " + line.ljust(w - 6) + " ", curses.A_REVERSE)
        s.refresh()


def _open_midi(app: App) -> None:
    try:
        from ..midi import MidiInput, available

        if available():
            mi = MidiInput(app.eng, on_event=app.say)
            app.say(mi.start())
            app.midi = mi
    except Exception:
        pass


def build_app(args, cfg) -> App:
    pr = Project.load(args.project) if args.project and os.path.exists(os.path.join(args.project, "project.json")) else Project("untitled")
    if args.project and not pr.path:
        pr.path = os.path.abspath(args.project)
    if args.seed is not None:
        pr.seed = args.seed
    eng = Engine(pr, block=args.block or int(cfg.get("block", 256)), seed=pr.seed)
    status = "welcome. :demo loads a built-in break, :open FILE loads yours, ? shows keys"
    app = App(pr, eng, None, status, cfg)
    if args.seed is not None:
        app.seed = args.seed
    return app


def run_tui(args, cfg) -> int:
    locale.setlocale(locale.LC_ALL, "")
    app = build_app(args, cfg)
    dev = args.device
    if dev is not None and str(dev).isdigit():
        dev = int(dev)
    be, msg = open_backend(app.eng, dev, args.input_device, args.no_output, want_input=args.input_device is not None)
    app.backend = be
    app.say(msg)
    if getattr(be, "name", "") == "silent" and not args.no_output:
        pass
    app.midi = None
    _open_midi(app)

    def loop(stdscr):
        curses.raw()
        curses.curs_set(0)
        stdscr.keypad(True)
        stdscr.timeout(33)
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS)
        except curses.error:
            pass
        app.apply_theme()
        if args.demo:
            app.command("demo")
        for f in args.files:
            try:
                app.load_source(f)
            except Exception as exc:  # noqa: BLE001
                app.say("could not load %s: %s" % (f, exc))
        last_auto = time.time()
        while not app.quit:
            app.draw(stdscr)
            app.handle_key(stdscr.getch())
            app.tick()
            if time.time() - last_auto > 60:
                app.pr.autosave()
                last_auto = time.time()

    try:
        curses.wrapper(loop)
    finally:
        be.stop()
        if getattr(app, "midi", None):
            app.midi.stop()
        app.cfg["seed"] = app.seed
        save_config(app.cfg)
    return 0
