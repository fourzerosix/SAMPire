"""All interface state and behaviour (no curses calls here, so it is fully testable without a terminal).
Drawing lives in views.py and the curses loop in tui.py."""

from __future__ import annotations

import curses
import os
import shlex
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from .. import analysis, creative, dsp, export, loops, providers, slicer, stems
from .. import asset as assetmod
from .. import demo, fx as fxmod, library as libmod
from ..config import PADS_PER_BANK, PPQ, save_config
from ..engine import Engine
from ..project import Project, ProjectError
from ..recorder import ThresholdRecorder
from ..taste import Taste
from .themes import NAMES, THEMES

PAD_KEYS = "1234qwerasdfzxcv"
CTRL = lambda c: ord(c) - 96
VIEWS = ["perf", "edit", "seq", "mix", "create", "browse", "diag"]
VIEW_TITLES = {"perf": "Perform", "edit": "Edit", "seq": "Sequence", "mix": "Mixer", "create": "Create", "browse": "Library", "diag": "Diagnostics"}
VIEW_ALIASES = {"1": "perf", "2": "edit", "3": "seq", "4": "mix", "5": "create", "6": "browse", "7": "diag", "perform": "perf",
                "editor": "edit", "sequencer": "seq", "mixer": "mix", "creative": "create", "library": "browse", "lib": "browse",
                "diagnostics": "diag"}
KIT_FIELDS = ["pads", "mode", "sensitivity", "experimental", "keep_original", "prefer", "length", "style", "taste"]
MODES = ["hybrid", "transient", "grid", "equal", "silence"]
PREFERS = ["any", "percussive", "melodic"]
LENGTHS = ["any", "short", "long"]
STYLES = ["auto", "break", "four_floor", "sparse", "euclid", "chops"]
MIX_ROWS = ["bus1", "bus2", "bus3", "bus4", "send_a", "send_b", "master"]

HELP = [
    "VIEWS     F1-F7 or ^N or :view NAME   perform / edit / sequence / mixer / create / library / diagnostics",
    "PADS      1234 qwer asdf zxcv  play (shift = softer)    arrows move    Tab pads <-> step grid",
    "TRANSPORT Space play/stop   + -  bpm   [ ]  swing   < >  bank   , .  pattern   ^U/^R undo/redo",
    "CREATE    ^G kit   ^P pattern   ^V vary   ^F chop-flip   y keep / n reject (teaches your taste, in Create view)",
    "CAPTURE   ^T record output   ^B skip-back   ^Y resample   ^W save   ^O open   drag a file onto the terminal to load it",
    "EDIT VIEW m add marker  k delete  h/l prev/next marker  i/u zoom  p audition  o put chops on pads  g auto markers",
    "SEQ VIEW  Enter toggle step  u/i velocity  p probability  k ratchet  l length  t grid/roll  Backspace delete",
    "MIXER     up/down select  left/right adjust  m mute  :fx busN add compressor  :fx busN sidechain 1",
    "LIBRARY   / search  Enter preview  o open as source  a assign to pad   :tag :rate :fav :similar :pack :lib scan DIR",
    "COMMANDS  :help lists them all.  :prompt chop into 12 pieces, reverse a few, two 4-bar patterns at 95 bpm with swing",
]
COMMAND_HELP = (
    "open demo chop kit pattern vary flip prompt provider keep reject taste | bpm swing seed | rev pitch stretch rate crush drive lp hp trunc "
    "gain norm stutter mode choke send pan att rel name clear pfilter lfo padbus protect chromatic roll variant variation | "
    "mk (add|del|clear|auto|snap|apply) loop (find|use) stems like | song (add|del|move|clear|on|off|list) | fx (list|master|busN|send_a|send_b|return) | "
    "lib find tag rate fav similar dups pack | midi (ports|learn|follow|out|map) | save load recover wav lufs stems pads busstems midi-file | "
    "view diag steal backend sample monitor theme quit")


def _kv(args: List[str]) -> dict:
    out = {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            try:
                out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                out[k] = v
    return out


class App:
    def __init__(self, project: Project, engine: Engine, backend=None, status: str = "", cfg: Optional[dict] = None,
                 library: Optional[libmod.Library] = None, taste: Optional[Taste] = None):
        self.pr, self.eng, self.backend = project, engine, backend
        self.cfg = cfg or {}
        self._lib = library
        self.taste = taste or Taste()
        self.midi = None
        self.view = "perf"
        self.sel = 0
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
        self.seed = int(self.cfg.get("seed", 1))
        self.pad_rect: dict = {}
        self.thresh_target: Optional[int] = None
        self.provider = "rules"
        self.allow_python = False
        # editor
        self.markers: List[int] = []
        self.ed_cursor = 0
        self.ed_zoom = 1
        self.ed_start = 0
        self.loop_cands: list = []
        # sequencer
        self.seq_roll = False
        # mixer
        self.mix_sel = 0
        # create
        self.cp = creative.KitParams()
        self.cp_sel = 0
        self.pat_style = "break"
        self.last_gen: Optional[dict] = None
        self.last_report = ""
        # browser
        self.results: list = []
        self.br_sel = 0
        self.br_query = ""
        self.engine_sel_sync()

    # ------------------------------------------------------------------ helpers
    @property
    def lib(self) -> libmod.Library:
        if self._lib is None:
            self._lib = libmod.Library()
        return self._lib

    def say(self, msg: str) -> None:
        self.log.append(msg)
        self.log = self.log[-80:]

    @property
    def pad(self):
        return self.pr.pad(self.sel)

    @property
    def source(self):
        return self.pr.assets.get(self.source_id) if self.source_id else None

    def engine_sel_sync(self) -> None:
        self.eng.selected_pad = self.sel

    def reload_engine(self) -> None:
        self.pr.render_all()
        self.eng.sync_from_project()

    def free_pad(self) -> int:
        for p in self.pr.banks[self.bank].pads:
            if p.empty:
                return p.index
        return self.sel

    def free_pads(self, n: int) -> List[int]:
        out = [p.index for p in self.pr.banks[self.bank].pads if p.empty][:n]
        while len(out) < n:
            out.append(self.sel)
        return out

    def set_view(self, name: str) -> None:
        name = VIEW_ALIASES.get(name, name)
        if name in VIEWS:
            self.view = name
            if name == "edit" and not self.markers:
                self.ed_reset()
            if name == "browse" and not self.results:
                self.results = self.lib.search(limit=300)

    def assign_audio(self, data: np.ndarray, name: str, origin: str, pad_index: Optional[int] = None) -> None:
        if len(data) < 64:
            self.say("nothing captured")
            return
        a = assetmod.asset_from_array(dsp.normalize(data, -1.0) if origin not in ("resample",) else data, name, self.pr.sr, origin,
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

    # ------------------------------------------------------------------ loading and generating
    def load_source(self, path: str) -> None:
        self.use_source(assetmod.load_audio(path))

    def use_source(self, a, make_kit: bool = True) -> None:
        self.pr.checkpoint()
        self.pr.add_asset(a)
        self.source_id = a.id
        a.analysis = analysis.analyze(a.data, a.sr)
        if a.analysis["bpm_confidence"] > 0.2:
            self.pr.bpm = a.analysis["bpm"]
            self.eng.set_bpm(self.pr.bpm)
        self.say("%s: %.1fs  %.1f BPM  %s" % (a.name, a.duration, a.analysis["bpm"], a.analysis["key"]))
        self.markers = []
        if make_kit:
            self.make_kit()

    def make_kit(self) -> None:
        a = self.source
        if a is None:
            self.say("no source loaded (:open FILE, :demo, or drag a file here)")
            return
        self.seed += 1
        self.cp.seed, self.cp.bank = self.seed, self.bank
        self.pr.checkpoint()
        rep = creative.generate_kit(self.pr, a, self.cp, taste_model=self.taste)
        self.last_report = "kit: %d pads from %d slices, %d transformed (seed %d)%s" % (
            rep.chosen, rep.slices, rep.transformed, self.seed, " [taste]" if self.cp.taste and self.taste.n else "")
        self.say(rep.notes[0] if rep.notes else self.last_report)
        self.last_gen = {"kind": "kit", "feats": creative.kit_features(self.pr, self.bank)}
        self.reload_engine()

    def make_pattern(self, style: Optional[str] = None) -> None:
        self.seed += 1
        style = style or self.pat_style
        self.pr.checkpoint()
        pt = creative.generate_pattern(self.pr, self.bank, max(self.pr.pattern.bars, 2), style, 0.5, 0.7, self.pr.swing, self.seed,
                                       taste_model=self.taste if self.cp.taste else None)
        self.pr.patterns[pt.name] = pt
        self.pr.current_pattern = pt.name
        self.last_gen = {"kind": "pattern", "feats": [creative.pattern_features(pt.style or style, 0.5)]}
        self.reload_engine()
        self.last_report = "pattern %s: %d events (%s, seed %d)" % (pt.name, len(pt.events), pt.style or style, self.seed)
        self.say(self.last_report)

    def keep(self) -> None:
        if not self.last_gen:
            self.say("nothing to keep yet (generate a kit or pattern first)")
            return
        self.taste.record(self.last_gen["feats"], True)
        self.say("kept: taste updated (%d examples)" % self.taste.n)

    def reject(self) -> None:
        if not self.last_gen:
            self.say("nothing to reject yet")
            return
        self.taste.record(self.last_gen["feats"], False)
        self.undo()
        self.last_gen = None
        self.say("rejected and undone: taste updated (%d examples)" % self.taste.n)

    # ------------------------------------------------------------------ sample editor
    def ed_reset(self) -> None:
        a = self.source
        if a is None:
            self.markers = []
            return
        bpm = a.analysis.get("bpm") if a.analysis.get("bpm_confidence", 0) > 0.25 else None
        sl = slicer.make_slices("hybrid", a.data, a.sr, bpm=bpm)
        self.markers = sorted(int(s.start) for s in sl)
        self.ed_cursor, self.ed_zoom, self.ed_start = 0, 1, 0

    def ed_span(self) -> int:
        a = self.source
        return max((a.frames if a else 1) // max(self.ed_zoom, 1), 1)

    def ed_clamp(self) -> None:
        a = self.source
        if a is None:
            return
        self.ed_cursor = int(np.clip(self.ed_cursor, 0, a.frames - 1))
        span = self.ed_span()
        if self.ed_cursor < self.ed_start:
            self.ed_start = self.ed_cursor
        if self.ed_cursor >= self.ed_start + span:
            self.ed_start = self.ed_cursor - span + 1
        self.ed_start = int(np.clip(self.ed_start, 0, max(a.frames - span, 0)))

    def ed_move(self, direction: int, coarse: bool = False) -> None:
        span = self.ed_span()
        self.ed_cursor += direction * max(span // (20 if coarse else 200), 1)
        self.ed_clamp()

    def ed_add_marker(self) -> None:
        a = self.source
        if a is None:
            return
        pos = slicer.snap_zero_crossing(dsp.mono(a.data), self.ed_cursor)
        if pos not in self.markers:
            self.markers = sorted(self.markers + [pos])
        self.say("marker at %.3fs (%d markers)" % (pos / a.sr, len(self.markers)))

    def ed_delete_marker(self) -> None:
        if not self.markers or self.source is None:
            return
        i = int(np.argmin([abs(m - self.ed_cursor) for m in self.markers]))
        m = self.markers.pop(i)
        self.say("deleted marker at %.3fs" % (m / self.source.sr))

    def ed_jump(self, direction: int) -> None:
        if not self.markers:
            return
        if direction > 0:
            nxt = [m for m in self.markers if m > self.ed_cursor]
            self.ed_cursor = nxt[0] if nxt else self.markers[-1]
        else:
            prv = [m for m in self.markers if m < self.ed_cursor]
            self.ed_cursor = prv[-1] if prv else self.markers[0]
        self.ed_clamp()

    def ed_region(self):
        a = self.source
        if a is None:
            return 0, 0
        starts = [m for m in self.markers if m <= self.ed_cursor]
        s = starts[-1] if starts else 0
        nxt = [m for m in self.markers if m > s]
        return s, (nxt[0] if nxt else a.frames)

    def ed_audition(self) -> None:
        a = self.source
        if a is None:
            return
        s, e = self.ed_region()
        self.eng.audition(dsp.declick(a.data[s:e][: 4 * a.sr], a.sr))

    def ed_apply(self) -> None:
        a = self.source
        if a is None or not self.markers:
            self.say("no markers (press g for automatic ones)")
            return
        edges = sorted(set([0] + self.markers + [a.frames]))
        sl = [slicer.Slice(x, y) for x, y in zip(edges[:-1], edges[1:]) if y - x > 64][:PADS_PER_BANK]
        self.pr.checkpoint()
        rep = creative.assign_slices(self.pr, a, sl, self.bank)
        self.reload_engine()
        self.say("%d chops on bank %s in time order" % (rep.chosen, self.pr.banks[self.bank].name))

    # ------------------------------------------------------------------ sequencer edits
    def seq_event(self):
        return self.pr.pattern.step_event(self.sel, self.cur_step)

    def seq_edit(self, what: str) -> None:
        ev = self.seq_event()
        if ev is None:
            self.say("no step here (Enter adds one)")
            return
        self.pr.checkpoint()
        pat = self.pr.pattern
        if what == "vel_up":
            ev.vel = min(ev.vel + 0.1, 1.0)
        elif what == "vel_down":
            ev.vel = max(ev.vel - 0.1, 0.05)
        elif what == "prob":
            steps = [1.0, 0.75, 0.5, 0.25]
            ev.prob = steps[(steps.index(min(steps, key=lambda v: abs(v - ev.prob))) + 1) % 4]
        elif what == "ratchet":
            ev.ratchet = ev.ratchet % 4 + 1
        elif what == "length":
            st = pat.step_ticks
            cycle = [0, st, 2 * st, 4 * st]
            ev.length = cycle[(cycle.index(min(cycle, key=lambda v: abs(v - ev.length))) + 1) % 4]
        pat.touch()
        self.say("vel %.1f  prob %d%%  ratchet x%d  length %d steps" % (ev.vel, round(ev.prob * 100), ev.ratchet, ev.length // pat.step_ticks))

    # ------------------------------------------------------------------ commands
    def command(self, line: str) -> None:
        line = line.strip().lstrip(":")
        if not line:
            return
        self.history.append(line)
        w = line.split(None, 1)
        c, rest = w[0].lower(), (w[1] if len(w) > 1 else "")
        try:
            args = shlex.split(rest) if rest else []
        except ValueError:
            args = rest.split()
        try:
            self._command(c, rest, args)
        except ProjectError as exc:
            self.say("project: %s" % exc)
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
        ext = getattr(self, "do_" + c.replace("-", "_"), None)
        if ext is not None:
            return ext(rest, a)
        if c in ("q", "quit", "exit"):
            self.quit = True
        elif c == "help":
            self.say(COMMAND_HELP)
            self.show_help = True
        elif c == "demo":
            self.use_source(assetmod.asset_from_array(demo.drum_break(92.0, 2, with_melody=True), "demo-break", origin="demo"))
        elif c in ("open", "o"):
            self.on_paste(rest)
        elif c == "chop":
            self.cp.mode = a[0] if a else "hybrid"
            self.cp.pads = int(f(1, 16))
            self.make_kit()
        elif c == "kit":
            self.cp.pads = int(f(0, self.cp.pads))
            self.make_kit()
        elif c == "pattern":
            self.make_pattern(a[0] if a else None)
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
            self._prompt(rest)
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
        elif c == "seed":
            self.seed = int(f(0, 1))
        elif c == "theme":
            if a and a[0] in THEMES:
                self.theme = a[0]
                self.cfg["theme"] = a[0]
        elif c == "sample":
            secs = f(0, 4.0)
            self.thresh_target = self.sel
            self.eng.thresh = ThresholdRecorder(self.eng.sr, -38.0, 40.0, fixed_frames=int(secs * self.eng.sr))
            self.say("armed: sampling %.1fs into pad %d when input crosses -38 dB" % (secs, self.sel + 1))
        else:
            self.say("unknown command %r  (:help)" % c)

    # -- creative extras
    def _prompt(self, text: str) -> None:
        if self.source is None:
            self.say("load a source first")
            return
        ctx = {"bpm": self.source.analysis.get("bpm"), "key": self.source.analysis.get("key"), "duration": self.source.duration,
               "pads_filled": sum(1 for q in self.pr.banks[self.bank].pads if not q.empty)}
        plan = providers.plan_from_prompt(text, self.provider, ctx, self.allow_python)
        self.say("plan: %s" % {k: v for k, v in plan.items() if v and k != "notes"})
        for n in plan["notes"]:
            self.say(n)
        self.seed += 1
        creative.run_plan(self.pr, self.source, plan, self.seed, self.bank, self.taste if self.cp.taste else None)
        self.eng.set_bpm(self.pr.bpm)
        self.reload_engine()
        self.last_gen = {"kind": "kit", "feats": creative.kit_features(self.pr, self.bank)}
        self.say("plan applied (^U undoes it; y/n in Create view teaches taste)")

    def do_provider(self, rest, a):
        if a:
            if a[0].startswith("python:") and not self.allow_python:
                raise PermissionError("python providers run your own code; start with --allow-python-provider")
            providers.get(a[0], self.allow_python)
            self.provider = a[0]
        self.say("plan provider: %s" % self.provider)

    def do_keep(self, rest, a):
        self.keep()

    def do_reject(self, rest, a):
        self.reject()

    def do_taste(self, rest, a):
        if a and a[0] == "reset":
            self.taste.reset()
            self.say("taste reset")
        elif a and a[0] in ("on", "off"):
            self.cp.taste = a[0] == "on"
            self.say("taste %s" % a[0])
        else:
            self.say("taste: %d examples; strongest: %s" % (self.taste.n, ", ".join("%s %+.2f" % kv for kv in self.taste.top(5)) or "none"))

    def do_like(self, rest, a):
        src, p = self.source, self.pad
        if src is None or p.sample is None:
            self.say("needs a source and a filled pad")
            return
        sl = slicer.make_slices("hybrid", src.data, src.sr, bpm=src.analysis.get("bpm") or None)
        target = analysis.embedding(p.sample, self.pr.sr)
        scored = sorted(((analysis.similarity(target, analysis.embedding(src.data[s.start:s.end], src.sr)), s) for s in sl
                         if (s.start, s.end) != (p.start, p.end)), key=lambda t: -t[0])
        n = int(a[0]) if a else 3
        self.pr.checkpoint()
        for (sc, s), gi in zip(scored[:n], self.free_pads(n)):
            q = self.pr.pad(gi)
            q.clear()
            q.asset_id, q.start, q.end, q.name = src.id, s.start, s.end, "like%d" % (gi % 16 + 1)
            q.label = analysis.classify_sample(src.data[s.start:s.end][: src.sr // 2], src.sr)
        self.reload_engine()
        self.say("%d chops most like pad %d (best %.2f)" % (min(n, len(scored)), p.index + 1, scored[0][0] if scored else 0))

    # -- markers and loops
    def do_mk(self, rest, a):
        sub = a[0] if a else "auto"
        if sub == "add":
            self.ed_cursor = int(float(a[1]) * self.source.sr) if len(a) > 1 else self.ed_cursor
            self.ed_add_marker()
        elif sub == "del":
            self.ed_delete_marker()
        elif sub == "clear":
            self.markers = []
        elif sub == "auto":
            self.ed_reset()
            self.say("%d automatic markers" % len(self.markers))
        elif sub == "snap" and self.source is not None:
            m = dsp.mono(self.source.data)
            self.markers = sorted({slicer.snap_zero_crossing(m, x) for x in self.markers})
        elif sub == "apply":
            self.ed_apply()

    def do_loop(self, rest, a):
        src = self.source
        if src is None:
            self.say("needs a source")
            return
        if not a or a[0] == "find":
            bpm = src.analysis.get("bpm") or None
            self.loop_cands = loops.find_loops(src.data, src.sr, bpm=bpm, beats=np.asarray(src.analysis.get("beats", [])) if bpm else None)
            self.say("loops: " + "  ".join("%d) %.2fs +%.1fs %g bar%s %.2f" % (i + 1, c.start / src.sr, c.length / src.sr, c.bars,
                                                                                "" if c.bars == 1 else "s", c.score)
                                           for i, c in enumerate(self.loop_cands[:3])) if self.loop_cands else "no loop found")
        elif a[0] == "use":
            i = int(a[1]) - 1 if len(a) > 1 else 0
            c = self.loop_cands[i]
            self.pr.checkpoint()
            p = self.pad
            p.clear()
            p.asset_id, p.start, p.end, p.name, p.mode, p.label = src.id, c.start, c.end, "loop%d" % (i + 1), "loop", "loop"
            p.recipe.add("seamless", ms=20)
            self.reload_engine()
            self.say("pad %d is a seamless %g bar loop" % (p.index + 1, c.bars))

    def do_stems(self, rest, a):
        src = self.source
        if src is None:
            self.say("needs a source")
            return
        method = a[0] if a else "dsp"
        self.say("separating (%s)..." % method)
        parts = stems.separate(src.data, src.sr, method, src.path if method == "demucs" else "")
        self.pr.checkpoint()
        for (name, data), gi in zip(parts.items(), self.free_pads(len(parts))):
            asset = assetmod.asset_from_array(data, "%s-%s" % (src.name, name), self.pr.sr, "stem", parent=src.id, stem=name)
            asset.meta["dirty"] = True
            self.pr.add_asset(asset)
            q = self.pr.pad(gi)
            q.clear()
            q.asset_id, q.start, q.end, q.name, q.label, q.mode = asset.id, 0, asset.frames, name[:8], name, "loop"
        self.reload_engine()
        self.say("stems on free pads: %s" % ", ".join(parts))

    # -- pad extras
    def do_pfilter(self, rest, a):
        p = self.pad
        if not a or a[0] == "off":
            p.filter_kind = ""
        else:
            p.filter_kind, p.filter_cut = a[0], float(a[1]) if len(a) > 1 else p.filter_cut
            p.filter_q = float(a[2]) if len(a) > 2 else p.filter_q
        self.say("pad %d filter: %s %s" % (p.index + 1, p.filter_kind or "off", "%.0f Hz" % p.filter_cut if p.filter_kind else ""))

    def do_lfo(self, rest, a):
        p = self.pad
        if not a or a[0] == "off":
            p.lfo_target = ""
        else:
            if a[0] not in ("pitch", "cutoff", "amp", "pan"):
                raise ValueError("lfo target: pitch cutoff amp pan")
            p.lfo_target = a[0]
            p.lfo_rate = float(a[1]) if len(a) > 1 else p.lfo_rate
            p.lfo_depth = float(a[2]) if len(a) > 2 else (p.lfo_depth or 0.5)
            p.lfo_shape = a[3] if len(a) > 3 else p.lfo_shape
        self.say("pad %d lfo: %s" % (p.index + 1, "off" if not p.lfo_target else "%s %.1f Hz depth %.2f %s" % (p.lfo_target, p.lfo_rate, p.lfo_depth, p.lfo_shape)))

    def do_padbus(self, rest, a):
        self.pad.bus = int(np.clip(int(a[0]) - 1, 0, 3)) if a else 0
        self.say("pad %d -> bus %d" % (self.sel + 1, self.pad.bus + 1))

    def do_protect(self, rest, a):
        self.pad.protect = (a[0] != "off") if a else True
        self.say("pad %d %s from voice stealing" % (self.sel + 1, "protected" if self.pad.protect else "not protected"))

    def do_chromatic(self, rest, a):
        on = (a[0] != "off") if a else not self.pad.chromatic
        for q in self.pr.all_pads():
            q.chromatic = False
        self.pad.chromatic = on and not self.pad.empty
        self.say("chromatic keys on pad %d (C4 = original pitch, keys 1..16 = semitones -4..+11)" % (self.sel + 1) if self.pad.chromatic else "chromatic off")

    def do_roll(self, rest, a):
        self.eng.roll(self.sel, int(a[0]) if a else 16, int(a[1]) if len(a) > 1 else 8)

    def do_variant(self, rest, a):
        p = self.pad
        if p.empty:
            return
        if a and a[0] == "clear":
            p.variants, p.variant_samples, p.variation = [], [], "none"
        else:
            rng = np.random.default_rng(self.seed + len(p.variants))
            self.pr.checkpoint()
            r = creative.Recipe(list(p.recipe.ops), p.recipe.seed + len(p.variants) + 1)
            r.add("pitch", semitones=float(rng.choice([-1.0, 1.0, -2.0, 2.0])))
            p.variants.append({"start": p.start, "end": p.end, "recipe": r.to_dict()})
            p.variation = "round_robin"
        p.render(self.pr.assets, self.pr.sr, force=True)
        self.say("pad %d: %d variant(s), %s" % (p.index + 1, len(p.variants), p.variation))

    def do_variation(self, rest, a):
        self.pad.variation = a[0] if a and a[0] in ("none", "round_robin", "random") else "none"

    def do_steal(self, rest, a):
        self.eng.steal_policy = a[0] if a and a[0] in ("oldest", "quietest") else "oldest"
        self.say("voice stealing: %s" % self.eng.steal_policy)

    def do_monitor(self, rest, a):
        self.eng.monitor_input = (a[0] != "off") if a else True
        self.say("input monitoring %s" % ("on" if self.eng.monitor_input else "off"))

    def do_backend(self, rest, a):
        from .. import dsp as d, analysis as an

        if len(a) >= 2 and a[0] == "stretch":
            d.set_stretch_backend(a[1])
        elif len(a) >= 2 and a[0] == "onsets":
            an.set_onset_backend(a[1])
        self.say("backends: stretch=%s onsets=%s (rubberband %s)" % (d.STRETCH_BACKEND, an.ONSET_BACKEND, "found" if d.rubberband_available() else "not found"))

    # -- song
    def do_song(self, rest, a):
        sub = a[0] if a else "list"
        self.pr.checkpoint()
        if sub == "add":
            self.pr.song_add(a[1] if len(a) > 1 else self.pr.current_pattern, int(a[2]) if len(a) > 2 else 1)
        elif sub == "del":
            self.pr.song_remove(int(a[1]) - 1)
        elif sub == "clear":
            self.pr.song_clear()
        elif sub == "move":
            self.pr.song_move(int(a[1]) - 1, int(a[2]) - 1)
        elif sub in ("on", "off"):
            self.pr.mode = "song" if sub == "on" else "pattern"
        self.eng.sync_from_project()
        self.say("song [%s]: %s (%.1fs)" % (self.pr.mode, "  ".join("%d) %s x%d" % (i + 1, n, r) for i, (n, r) in enumerate(self.pr.song)) or "empty", self.pr.song_seconds()))

    # -- effects and buses
    def do_fx(self, rest, a):
        e = self.eng
        if not a or a[0] == "list":
            self.say("effects: %s | targets: master bus1-4 send_a send_b | fx master add compressor threshold_db=-20 ratio=4" % ", ".join(fxmod.list_effects()))
            return
        e.commit_settings()
        self.pr.checkpoint()
        t = a[0]
        if t == "return":
            if a[1] == "a":
                e.return_a = float(a[2])
            else:
                e.return_b = float(a[2])
        elif t in ("send_a", "send_b"):
            new = fxmod.make_fx(a[1], e.sr, **_kv(a[2:]))
            if t == "send_a":
                e.fx_a = new
            else:
                e.fx_b = new
        elif t == "master" or (t.startswith("bus") and t[3:].isdigit()):
            chain = e.master_fx if t == "master" else e.buses[int(t[3:]) - 1].fx
            bus = None if t == "master" else e.buses[int(t[3:]) - 1]
            sub = a[1] if len(a) > 1 else ""
            if sub == "add":
                chain.append(fxmod.make_fx(a[2], e.sr, **_kv(a[3:])))
            elif sub == "set":
                chain[int(a[2]) - 1].set(**_kv(a[3:]))
            elif sub == "rm":
                chain.pop(int(a[2]) - 1)
            elif sub == "clear":
                chain.clear()
            elif bus is not None and sub == "gain":
                bus.gain = float(a[2])
            elif bus is not None and sub == "mute":
                bus.mute = (a[2] != "off") if len(a) > 2 else not bus.mute
            elif bus is not None and sub == "sidechain":
                bus.sidechain = None if a[2] == "off" else int(a[2]) - 1
            elif bus is not None and sub == "name":
                bus.name = a[2]
            elif sub in fxmod.REGISTRY:                   # shorthand: fx master reverb ...
                chain.append(fxmod.make_fx(sub, e.sr, **_kv(a[2:])))
            else:
                raise ValueError("fx %s: add TYPE k=v | set N k=v | rm N | clear%s" % (t, " | gain V | mute | sidechain N|off" if bus else ""))
        else:
            raise ValueError("fx target: master bus1-4 send_a send_b return")
        e.commit_settings()
        self.say("fx %s: %s" % (t, ", ".join(x.name for x in (e.master_fx if t == "master" else e.buses[int(t[3:]) - 1].fx)) if t == "master" or t.startswith("bus") else "ok"))

    # -- library
    def do_lib(self, rest, a):
        sub = a[0] if a else "info"
        if sub == "scan":
            dirs = [os.path.expanduser(x) for x in a[1:]] or [os.getcwd()]
            self.say("scanning %s in the background..." % ", ".join(dirs))
            threading.Thread(target=self._scan_thread, args=(dirs,), daemon=True).start()
        elif sub == "clean":
            self.say("removed %d missing entries" % self.lib.remove_missing())
        elif sub == "dups":
            groups = self.lib.duplicates(near="near" in a)
            self.say("%d duplicate groups: %s" % (len(groups), "; ".join("/".join(r.name for r in g[:3]) for g in groups[:4])))
        else:
            self.say("library: %d samples (%s)" % (self.lib.count(), self.lib.path))

    def _scan_thread(self, dirs):
        try:
            lib = libmod.Library(self.lib.path)
            added, skipped = lib.scan(dirs, self.say)
            lib.close()
            self.say("scan done: %d added, %d unchanged" % (added, skipped))
            self.results = self.lib.search(limit=300)
        except Exception as exc:  # noqa: BLE001
            self.say("scan failed: %s" % exc)

    def do_find(self, rest, a):
        self.br_query = rest
        self.results = self.lib.find(rest, 300)
        self.br_sel = 0
        self.view = "browse"
        self.say("%d matches for %r" % (len(self.results), rest))

    def do_dups(self, rest, a):
        self.do_lib("", ["dups"] + a)

    def _cur_row(self):
        return self.results[self.br_sel] if self.results and 0 <= self.br_sel < len(self.results) else None

    def do_tag(self, rest, a):
        r = self._cur_row()
        if r is None or not a:
            return
        (self.lib.remove_tag if a[0] == "rm" else self.lib.add_tag)(r.id, a[-1])
        self.results = self.lib.find(self.br_query, 300)

    def do_rate(self, rest, a):
        r = self._cur_row()
        if r is not None and a:
            self.lib.set_rating(r.id, int(a[0]))
            self.results = self.lib.find(self.br_query, 300)

    def do_fav(self, rest, a):
        r = self._cur_row()
        if r is not None:
            self.lib.set_favorite(r.id, not r.fav)
            self.results = self.lib.find(self.br_query, 300)

    def do_similar(self, rest, a):
        r = self._cur_row()
        if self.view != "browse" or r is None:
            p = self.pad
            if p.sample is None:
                self.say("select a filled pad or a library sample")
                return
            self.results = self.lib.similar((p.sample, self.pr.sr), int(a[0]) if a else 20)
        else:
            self.results = self.lib.similar(r.id, int(a[0]) if a else 20)
        self.br_sel = 0
        self.view = "browse"
        self.say("%d similar samples" % len(self.results))

    def do_pack(self, rest, a):
        dest, n = self.lib.import_pack(rest.strip())
        self.results = self.lib.search(limit=300)
        self.say("imported %d samples from pack into %s" % (n, dest))

    def browse_audition(self) -> None:
        r = self._cur_row()
        if r is None:
            return
        a = assetmod.load_audio(r.path)
        self.eng.audition(a.data[: 8 * a.sr])
        self.lib.touch(r.id)

    def browse_open(self) -> None:
        r = self._cur_row()
        if r is not None:
            self.use_source(assetmod.load_audio(r.path))
            self.lib.touch(r.id)

    def browse_assign(self) -> None:
        r = self._cur_row()
        if r is not None:
            a = assetmod.load_audio(r.path)
            self.assign_audio(a.data[: 8 * a.sr], r.name[:8], "import")
            self.lib.touch(r.id)

    # -- midi
    def _midi(self):
        if self.midi is None:
            from ..midi import MidiManager, available

            if not available():
                raise RuntimeError("MIDI needs: pip install sampire[midi]")
            self.midi = MidiManager(self.eng, on_event=self.say)
            self.say(self.midi.start())
        return self.midi

    def do_midi(self, rest, a):
        sub = a[0] if a else "status"
        if sub == "ports":
            from ..midi import list_ports

            pr = list_ports()
            self.say("MIDI in: %s | out: %s" % (", ".join(pr["inputs"]) or "none", ", ".join(pr["outputs"]) or "none"))
        elif sub == "learn":
            m = self._midi()
            m.router.learn(a[1], "project" if "project" in a else "global")
        elif sub == "follow":
            self._midi().set_follow_clock(a[1] != "off" if len(a) > 1 else True)
            self.say("following MIDI clock" if self.midi.follow_clock else "internal clock")
        elif sub == "out":
            m = self._midi()
            on = (a[1] != "off") if len(a) > 1 else True
            if on:
                self.say(m.open_output(a[2] if len(a) > 2 else None))
            self.eng.clock_out = on
        elif sub == "map":
            m = self._midi()
            self.say("map: " + "  ".join("%s>%s" % kv for kv in list(m.router.map.glob.items())[:8]) + "  (defaults: CC7 master, CC10 pan, CC16-31 pad params)")
        else:
            self.say(self._midi().status_text() if self.midi else "MIDI not started (:midi ports)")

    # -- files, export, recovery
    def do_save(self, rest, a):
        self.eng.commit_settings()
        self.say("saved " + self.pr.save(rest.strip() or self.pr.path or os.path.join(os.getcwd(), "sampire-project")))

    def do_load(self, rest, a):
        self.pr = Project.load(os.path.expanduser(rest.strip()))
        self.eng.set_project(self.pr)
        self.source_id = next(iter(self.pr.assets), None)
        self.markers = []
        self.say("loaded " + self.pr.name)

    def do_recover(self, rest, a):
        path = rest.strip() or self.pr.path
        self.pr = Project.recover(os.path.expanduser(path))
        self.eng.set_project(self.pr)
        self.source_id = next(iter(self.pr.assets), None)
        self.say("recovered %s from autosave" % self.pr.name)

    def do_wav(self, rest, a):
        y = export.render_pattern(self.pr, None, 1)
        self.say("wrote " + export.export_audio(os.path.expanduser(rest.strip() or "sampire_out.wav"), y, self.pr.sr))

    def do_lufs(self, rest, a):
        target = float(a[0]) if a else -14.0
        y = export.render_pattern(self.pr, None, 1, lufs=target)
        info = getattr(export.render_pattern, "last_info", {})
        path = os.path.expanduser(a[1] if len(a) > 1 else "sampire_out.wav")
        export.export_audio(path, y, self.pr.sr)
        self.say("wrote %s at %.1f LUFS%s (true peak %.1f dB)" % (path, info.get("after", 0), " (peak limited)" if info.get("limited") else "", info.get("true_peak_db", 0)))

    def do_stemfiles(self, rest, a):
        self.say("%d stems written" % len(export.export_stems(self.pr, os.path.expanduser(rest.strip() or "sampire_stems"))))

    def do_busstems(self, rest, a):
        self.eng.commit_settings()
        self.say("%d bus stems written" % len(export.export_bus_stems(self.pr, os.path.expanduser(rest.strip() or "sampire_busstems"))))

    def do_pads(self, rest, a):
        self.say("%d pad wavs written" % len(export.export_pads(self.pr, os.path.expanduser(rest.strip() or "sampire_pads"))))

    def do_midi_file(self, rest, a):
        self.say("wrote " + export.pattern_to_midi(self.pr, self.pr.pattern, os.path.expanduser(rest.strip() or "sampire.mid")))

    def do_view(self, rest, a):
        self.set_view(a[0] if a else "perf")

    def do_diag(self, rest, a):
        self.view = "diag"

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

    # ------------------------------------------------------------------ drag and drop / paste
    def on_paste(self, text: str) -> None:
        """A terminal pastes the file path when you drop a file on it. Handles quotes, spaces and ~."""
        if self.cmd_mode:
            self.cmd += text.strip().replace("\n", " ")
            return
        try:
            tokens = shlex.split(text.strip())
        except ValueError:
            tokens = [text.strip()]
        loaded = 0
        for tok in tokens:
            path = os.path.expanduser(tok)
            if os.path.isdir(path):
                self.do_lib("", ["scan", path])
            elif os.path.isfile(path) and path.lower().endswith(libmod.AUDIO_EXT):
                if loaded == 0:
                    self.load_source(path)
                else:
                    a = assetmod.load_audio(path)
                    self.assign_audio(a.data[: 8 * a.sr], a.name[:8], "import", self.free_pad())
                loaded += 1
            elif os.path.isfile(path) and path.lower().endswith(".zip"):
                self.do_pack(path, [path])
            else:
                self.say("not an audio file or folder: %s" % tok[:60])
        if loaded:
            self.say("loaded %d file(s)" % loaded)

    # ------------------------------------------------------------------ keys
    def handle_key(self, ch: int) -> None:
        if self.cmd_mode:
            return self._cmd_key(ch)
        if ch == -1:
            return
        if ch == curses.KEY_MOUSE:
            return self._mouse()
        if ch in (3, CTRL("q")):
            self.quit = True
            return
        if ch == ord(":") or ch == CTRL("e"):
            self.cmd_mode, self.cmd = True, ""
            return
        if ch in (ord("/"), ord("?")) and self.view != "browse":
            self.show_help = not self.show_help
            return
        if ch == 27:
            self.show_help = False
            return
        for i in range(7):                                   # F1..F7 switch workspace
            if ch == curses.KEY_F0 + 1 + i:
                self.set_view(VIEWS[i])
                return
        if ch == CTRL("n"):
            self.set_view(VIEWS[(VIEWS.index(self.view) + 1) % len(VIEWS)])
            return
        if self._global_key(ch):
            return
        if self._view_key(ch):
            return
        self._pad_key(ch)

    def _global_key(self, ch: int) -> bool:
        if ch == ord(" "):
            self.eng.stop() if self.eng.playing else self.eng.play()
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
            self.engine_sel_sync()
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
        else:
            return False
        return True

    def _view_key(self, ch: int) -> bool:
        v = self.view
        c = chr(ch) if 0 < ch < 256 else ""
        arrows = (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT)
        if v == "perf":
            if ch in arrows:
                self._arrow_perf(ch)
                return True
            if ch == 9:
                self.focus = "seq" if self.focus == "pads" else "pads"
                return True
            if ch in (10, 13, curses.KEY_ENTER):
                if self.focus == "seq":
                    if self.pr.pattern.toggle_step(self.sel, self.cur_step):
                        self.eng.trigger(self.sel, 0.9)
                else:
                    self.eng.trigger(self.sel, 1.0)
                return True
            if ch in (curses.KEY_BACKSPACE, 127, 8) and self.focus == "seq":
                e = self.seq_event()
                if e:
                    self.pr.pattern.events.remove(e)
                    self.pr.pattern.touch()
                return True
        elif v == "edit":
            if ch == curses.KEY_LEFT:
                self.ed_move(-1)
            elif ch == curses.KEY_RIGHT:
                self.ed_move(1)
            elif ch == curses.KEY_UP:
                self.ed_move(1, True)
            elif ch == curses.KEY_DOWN:
                self.ed_move(-1, True)
            elif c == "m":
                self.ed_add_marker()
            elif c == "k":
                self.ed_delete_marker()
            elif c == "h":
                self.ed_jump(-1)
            elif c == "l":
                self.ed_jump(1)
            elif c == "i":
                self.ed_zoom = min(self.ed_zoom * 2, 4096)
                self.ed_clamp()
            elif c == "u":
                self.ed_zoom = max(self.ed_zoom // 2, 1)
                self.ed_clamp()
            elif c == "p":
                self.ed_audition()
            elif c == "o":
                self.ed_apply()
            elif c == "g":
                self.ed_reset()
                self.say("%d automatic markers" % len(self.markers))
            else:
                return False
            return True
        elif v == "seq":
            steps = self.pr.pattern.steps
            if ch == curses.KEY_LEFT:
                self.cur_step = (self.cur_step - 1) % steps
            elif ch == curses.KEY_RIGHT:
                self.cur_step = (self.cur_step + 1) % steps
            elif ch in (curses.KEY_UP, curses.KEY_DOWN):
                slot = (self.sel % PADS_PER_BANK + (1 if ch == curses.KEY_DOWN else -1)) % PADS_PER_BANK
                self.sel = self.bank * PADS_PER_BANK + slot
                self.engine_sel_sync()
            elif ch in (10, 13, curses.KEY_ENTER):
                if self.pr.pattern.toggle_step(self.sel, self.cur_step):
                    self.eng.trigger(self.sel, 0.9)
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                e = self.seq_event()
                if e:
                    self.pr.pattern.events.remove(e)
                    self.pr.pattern.touch()
            elif c == "i":
                self.seq_edit("vel_up")
            elif c == "u":
                self.seq_edit("vel_down")
            elif c == "p":
                self.seq_edit("prob")
            elif c == "k":
                self.seq_edit("ratchet")
            elif c == "l":
                self.seq_edit("length")
            elif c == "t":
                self.seq_roll = not self.seq_roll
            else:
                return False
            return True
        elif v == "mix":
            e = self.eng
            if ch == curses.KEY_UP:
                self.mix_sel = (self.mix_sel - 1) % len(MIX_ROWS)
            elif ch == curses.KEY_DOWN:
                self.mix_sel = (self.mix_sel + 1) % len(MIX_ROWS)
            elif ch in (curses.KEY_LEFT, curses.KEY_RIGHT):
                d = 0.05 if ch == curses.KEY_RIGHT else -0.05
                row = MIX_ROWS[self.mix_sel]
                if row.startswith("bus"):
                    b = e.buses[int(row[3]) - 1]
                    b.gain = float(np.clip(b.gain + d, 0.0, 1.5))
                elif row == "send_a":
                    e.return_a = float(np.clip(e.return_a + d, 0.0, 1.0))
                elif row == "send_b":
                    e.return_b = float(np.clip(e.return_b + d, 0.0, 1.0))
                else:
                    e.master_gain = float(np.clip(e.master_gain + d, 0.0, 1.2))
                e.commit_settings()
            elif c == "m" and MIX_ROWS[self.mix_sel].startswith("bus"):
                b = e.buses[int(MIX_ROWS[self.mix_sel][3]) - 1]
                b.mute = not b.mute
                e.commit_settings()
            elif c == "c" and MIX_ROWS[self.mix_sel].startswith("bus"):
                e.buses[int(MIX_ROWS[self.mix_sel][3]) - 1].fx.clear()
                e.commit_settings()
            else:
                return False
            return True
        elif v == "create":
            if ch == curses.KEY_UP:
                self.cp_sel = (self.cp_sel - 1) % len(KIT_FIELDS)
            elif ch == curses.KEY_DOWN:
                self.cp_sel = (self.cp_sel + 1) % len(KIT_FIELDS)
            elif ch in (curses.KEY_LEFT, curses.KEY_RIGHT):
                self._cp_adjust(1 if ch == curses.KEY_RIGHT else -1)
            elif c == "g":
                self.make_kit()
            elif c == "p":
                self.make_pattern()
            elif c == "y":
                self.keep()
            elif c == "n":
                self.reject()
            elif c == "t":
                self.cp.taste = not self.cp.taste
            else:
                return False
            return True
        elif v == "browse":
            if ch == curses.KEY_UP:
                self.br_sel = max(self.br_sel - 1, 0)
            elif ch == curses.KEY_DOWN:
                self.br_sel = min(self.br_sel + 1, max(len(self.results) - 1, 0))
            elif ch in (10, 13, curses.KEY_ENTER):
                self.browse_audition()
            elif c == "o":
                self.browse_open()
            elif c == "a":
                self.browse_assign()
            elif c == "/":
                self.cmd_mode, self.cmd = True, "find "
            else:
                return False
            return True
        return False

    def _cp_adjust(self, d: int) -> None:
        f, cp = KIT_FIELDS[self.cp_sel], self.cp
        if f == "pads":
            cp.pads = int(np.clip(cp.pads + d, 2, 16))
        elif f == "mode":
            cp.mode = MODES[(MODES.index(cp.mode) + d) % len(MODES)]
        elif f == "prefer":
            cp.prefer = PREFERS[(PREFERS.index(cp.prefer) + d) % len(PREFERS)]
        elif f == "length":
            cp.length = LENGTHS[(LENGTHS.index(cp.length) + d) % len(LENGTHS)]
        elif f == "style":
            self.pat_style = STYLES[(STYLES.index(self.pat_style) + d) % len(STYLES)]
        elif f == "taste":
            cp.taste = not cp.taste
        else:
            setattr(cp, f, float(np.clip(getattr(cp, f) + 0.1 * d, 0.0, 1.0)))

    def _arrow_perf(self, ch: int) -> None:
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
            self.engine_sel_sync()
        else:
            steps = self.pr.pattern.steps
            if ch == curses.KEY_LEFT:
                self.cur_step = (self.cur_step - 1) % steps
            elif ch == curses.KEY_RIGHT:
                self.cur_step = (self.cur_step + 1) % steps
            else:
                slot = (self.sel % PADS_PER_BANK + (1 if ch == curses.KEY_DOWN else -1)) % PADS_PER_BANK
                self.sel = self.bank * PADS_PER_BANK + slot
                self.engine_sel_sync()

    def _pad_key(self, ch: int) -> None:
        if not (0 < ch < 256 and chr(ch).lower() in PAD_KEYS):
            return
        slot = PAD_KEYS.index(chr(ch).lower())
        vel = 1.0 if chr(ch).islower() else 0.55
        cp = next((p for p in self.pr.all_pads() if p.chromatic and not p.empty), None)
        if cp is not None:
            self.eng.trigger(cp.index, vel, {"pitch": float(slot - 4)})
            return
        gi = self.bank * PADS_PER_BANK + slot
        self.sel = gi
        self.engine_sel_sync()
        self.eng.trigger(gi, vel)

    def _cmd_key(self, ch: int) -> None:
        if ch == 27:
            self.cmd_mode = False
        elif ch in (10, 13, curses.KEY_ENTER):
            self.cmd_mode = False
            self.command(self.cmd)
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            self.cmd = self.cmd[:-1]
        elif ch == curses.KEY_UP and self.history:
            self.hist_i = (self.hist_i + 1) % len(self.history)
            self.cmd = self.history[-self.hist_i]
        elif 32 <= ch < 127 or ch > 160:
            try:
                self.cmd += chr(ch)
            except ValueError:
                pass

    def _mouse(self) -> None:
        try:
            _, mx, my, _, bstate = curses.getmouse()
        except curses.error:
            return
        if self.view == "perf" and bstate & (curses.BUTTON1_PRESSED | curses.BUTTON1_CLICKED):
            for gi, (y, x, h, w) in self.pad_rect.items():
                if y <= my < y + h and x <= mx < x + w:
                    self.sel = gi
                    self.engine_sel_sync()
                    self.eng.trigger(gi, 1.0)

    # ------------------------------------------------------------------ periodic
    def tick(self) -> None:
        t = self.eng.thresh
        if t is not None and t.state == "done":
            self.eng.thresh = None
            self.assign_audio(t.result(), "sample", "record", self.thresh_target)
        if self.eng.active_bank != self.bank and self.eng.active_bank < len(self.pr.banks):
            self.bank = self.eng.active_bank
            self.sel = self.bank * PADS_PER_BANK + (self.sel % PADS_PER_BANK)
            self.engine_sel_sync()
