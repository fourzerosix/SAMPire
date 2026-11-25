"""Drawing for every workspace. Pure functions of App state; the only side effect is writing to the curses screen."""

from __future__ import annotations

import curses
import sys
from typing import List

import numpy as np

from .. import __version__
from ..config import PADS_PER_BANK
from ..sequencer import ticks_to_bar_beat
from . import braille
from .app import HELP, KIT_FIELDS, MIX_ROWS, PAD_KEYS, VIEWS, VIEW_TITLES, App

GLYPHS = " ▁▂▃▄▅▆▇█"


class Ctx:
    def __init__(self, s, app: App):
        self.s, self.app = s, app
        self.h, self.w = s.getmaxyx()
        C = curses.color_pair
        self.ACC, self.HOT, self.OK, self.DIM = C(2), C(3), C(4), C(5)

    def put(self, y: int, x: int, text: str, attr: int = 0, maxw=None) -> None:
        if y < 0 or y >= self.h or x >= self.w or x < 0:
            return
        n = min(len(text), (maxw if maxw is not None else self.w - x), self.w - x - (1 if y == self.h - 1 else 0))
        if n > 0:
            try:
                self.s.addnstr(y, x, text, n, attr)
            except curses.error:
                pass


def bar(v: float, width: int, hi: float = 1.0) -> str:
    n = int(np.clip(v / hi, 0.0, 1.0) * width)
    return "#" * n + "." * (width - n)


def draw(s, app: App) -> None:
    s.erase()
    c = Ctx(s, app)
    if c.h < 24 or c.w < 80:
        c.put(0, 0, "terminal too small: need 80x24 (have %dx%d)" % (c.w, c.h), c.HOT)
        s.refresh()
        return
    header(c)
    tabs(c)
    {"perf": perf, "edit": edit, "seq": seq, "mix": mix, "create": create, "browse": browse, "diag": diag}[app.view](c)
    footer(c)
    if app.show_help:
        for i, line in enumerate(HELP):
            c.put(3 + i, 2, " " + line.ljust(c.w - 6) + " ", curses.A_REVERSE)
    s.refresh()


def header(c: Ctx) -> None:
    app, e, pr = c.app, c.app.eng, c.app.pr
    info = app.backend.info() if app.backend else {"name": "none", "cpu": 0, "underruns": 0}
    lt = e.sched.local_tick(e.frame) if e.playing else None
    pos = "%d.%d.%d" % ticks_to_bar_beat(lt[1]) if lt else "-.-.-"
    c.put(0, 0, " SAMPIRE %s " % __version__, c.ACC | curses.A_BOLD)
    sync = " EXT" if e.external_sync else ""
    head = " %s%s %s  %.1f BPM  swing %d%%  pat %s %s  bank %s  src %s" % (
        "PLAY" if e.playing else "STOP", " REC" if e.rec_arm else "", sync, pr.bpm, round(pr.swing * 100), pr.current_pattern, pos,
        pr.banks[app.bank].name, app.source.name[:12] if app.source else "none")
    c.put(0, 17, head, (c.OK if e.playing else c.DIM) | curses.A_BOLD)
    cpu = max(e.cpu, info.get("cpu", 0))
    right = "%s %3.0f%% vox %2d xr %d " % (info["name"][:16], cpu * 100, e.voice_count, e.underruns)
    c.put(0, max(c.w - len(right) - 1, 0), right, c.HOT if cpu > 0.7 or e.underruns else c.DIM)


def tabs(c: Ctx) -> None:
    x = 1
    for i, v in enumerate(VIEWS):
        label = " F%d %s " % (i + 1, VIEW_TITLES[v])
        c.put(1, x, label, (c.ACC | curses.A_REVERSE) if v == c.app.view else c.DIM)
        x += len(label) + 1
    e = c.app.eng
    c.put(1, c.w - 24, "L %-8s R %-8s" % (bar(e.peak[0], 8), bar(e.peak[1], 8)), c.HOT if max(e.peak) > 0.95 else c.OK)


def footer(c: Ctx) -> None:
    app, h = c.app, c.h
    sa = app.source.analysis if app.source else {}
    det = " src %s  %.1f BPM  %s  %d onsets | out-rec %s | resample %s | undo %d | taste %d" % (
        app.source.name if app.source else "-", sa.get("bpm", 0), sa.get("key", "-"), len(sa.get("onsets", [])),
        "ON" if app.master_recording else "off", "ON" if app.resampling else "off", len(app.pr._undo), app.taste.n)
    c.put(h - 5, 0, det, c.DIM)
    for i, m in enumerate(app.log[-2:]):
        c.put(h - 4 + i, 0, " " + m)
    if app.cmd_mode:
        c.put(h - 2, 0, ":" + app.cmd + "_", c.ACC | curses.A_BOLD)
    else:
        hints = {
            "perf": "pads 1234 qwer asdf zxcv | Space play | Tab steps | ^G kit ^P pattern | ^T rec | : commands | ? help | F1-F7 views",
            "edit": "m marker  k delete  h/l prev/next  i/u zoom  arrows move  p audition  o chops to pads  g auto  :loop find  :stems",
            "seq": "Enter step  u/i velocity  p probability  k ratchet  l length  t grid/roll  arrows move  Space play",
            "mix": "up/down select  left/right level  m mute  c clear fx  :fx bus1 add compressor  :fx bus2 sidechain 1  :padbus 2",
            "create": "up/down param  left/right change  g kit  p pattern  y keep  n reject  t taste  :prompt TEXT",
            "browse": "/ search  Enter preview  o open as source  a assign to pad  :tag :rate :fav :similar :pack :lib scan DIR",
            "diag": ":midi ports  :lib info  :backend  :steal quietest  :view perf",
        }[app.view]
        c.put(h - 2, 0, " " + hints, c.DIM)


# ------------------------------------------------------------------------------- perform
def perf(c: Ctx) -> None:
    app, e, pr = c.app, c.app.eng, c.app.pr
    top = 2
    lt = e.sched.local_tick(e.frame) if e.playing else None
    c.put(top, 0, " PADS %s " % pr.banks[app.bank].name, c.ACC)
    app.pad_rect = {}
    for slot in range(PADS_PER_BANK):
        r, col = divmod(slot, 4)
        gi = app.bank * PADS_PER_BANK + slot
        pad = pr.pad(gi)
        y, x = top + 1 + r * 4, col * 9
        flash = (e.frame - e.flash.get(gi, -10 ** 9)) < 0.12 * e.sr
        attr = (c.HOT | curses.A_REVERSE) if flash else (c.ACC | curses.A_BOLD if gi == app.sel else (0 if not pad.empty else c.DIM))
        c.put(y, x, "+-------+", attr)
        flags = ("C" if pad.chromatic else "") + ("L" if pad.mode == "loop" else "") + ("f" if pad.filter_kind else "") + ("~" if pad.lfo_target else "") + ("%d" % (pad.bus + 1) if pad.bus else "")
        c.put(y + 1, x, "|%s %-5s|" % (PAD_KEYS[slot].upper(), (pad.name or "")[:5]), attr)
        c.put(y + 2, x, "|%-4s%-3s|" % ((pad.label or ("." if pad.empty else "?"))[:4], flags[:3]), attr)
        app.pad_rect[gi] = (y, x, 3, 9)
    lw = 36
    pad = app.pad
    ww = c.w - lw - 1
    c.put(top, lw, " pad %d %s  %s  bus %d  %s " % (pad.index + 1, pad.name or "-", pad.mode, pad.bus + 1, pad.recipe.describe()[: max(ww - 34, 8)]), c.ACC)
    wh = 7
    key = (pad.render_key(), ww, id(pad.sample), app.source_id)
    if key != getattr(app, "_wave_key", None):
        app._wave_key = key
        if pad.sample is not None:
            app._wave_rows = braille.waveform(pad.sample, ww - 1, wh).rows()
        elif app.source is not None:
            cv = braille.waveform(app.source.data, ww - 1, wh)
            braille.markers(cv, app.source.analysis.get("onsets", [])[:200], app.source.frames)
            app._wave_rows = cv.rows()
        else:
            app._wave_rows = braille.waveform(np.zeros((0, 2)), ww - 1, wh).rows()
    for i, row in enumerate(app._wave_rows):
        c.put(top + 1 + i, lw, row, c.OK if pad.sample is not None else c.DIM)
    for v in e.voices:
        if v.pad == app.sel and len(v.data):
            x = lw + int(min(v.pos / len(v.data), 1.0) * (ww - 2))
            for i in range(wh):
                c.put(top + 1 + i, x, "|", c.HOT)
    seq_block(c, lw, top + 1 + wh + 1, lt, compact=True)


def seq_block(c: Ctx, x0: int, y0: int, lt, compact: bool, rows=None) -> None:
    app, pr = c.app, c.app.pr
    pat = pr.pattern
    steps = pat.steps
    page = 32 if (c.w - x0 - 10) >= 64 else 16
    play_step = int(lt[1] // pat.step_ticks) if lt else -1
    ref = app.cur_step if (app.focus == "seq" or app.view == "seq") else (play_step if play_step >= 0 else 0)
    start = (ref // page) * page
    c.put(y0 - 1, x0, " STEPS %s %d bars  steps %d-%d/%d  swing %d%% %s" % (pat.name, pat.bars, start + 1, min(start + page, steps), steps,
                                                                           round((pat.swing or pr.swing) * 100), "ROLL" if (app.seq_roll and app.view == "seq") else ""), c.ACC)
    nrows = rows or min(PADS_PER_BANK, c.h - y0 - 5)
    base = app.bank * PADS_PER_BANK
    for r in range(nrows):
        gi = base + r
        evs = {}
        for e in pat.events:
            if e.pad == gi:
                evs[int(round(e.tick / pat.step_ticks)) % steps] = e
        c.put(y0 + r, x0, "%2d %-5s" % (r + 1, (pr.pad(gi).name or "")[:5]), c.ACC if gi == app.sel else c.DIM)
        for k in range(page):
            st = start + k
            if st >= steps:
                break
            ev = evs.get(st)
            if ev is not None:
                ch = GLYPHS[int(np.clip(ev.vel, 0, 1) * 7) + 1] if (app.seq_roll and app.view == "seq") else "#"
                if ev.prob < 1.0:
                    ch = "?"
                if ev.ratchet > 1:
                    ch = str(min(ev.ratchet, 9))
            else:
                ch = "." if st % 4 else ":"
            attr = c.DIM
            if st == play_step:
                attr = c.HOT | curses.A_REVERSE
            elif (app.focus == "seq" or app.view == "seq") and gi == app.sel and st == app.cur_step:
                attr = c.ACC | curses.A_REVERSE
            elif ev is not None:
                attr = c.OK | curses.A_BOLD
            c.put(y0 + r, x0 + 9 + k * 2, ch + " ", attr)


# ------------------------------------------------------------------------------- editor
def edit(c: Ctx) -> None:
    app = c.app
    a = app.source
    if a is None:
        c.put(3, 2, "No source loaded. :demo, :open FILE, or drop a file onto the terminal.", c.DIM)
        return
    span = app.ed_span()
    app.ed_clamp()
    s0 = app.ed_start
    c.put(2, 1, " %s  %.2fs  zoom x%d  window %.2f-%.2fs  cursor %.3fs  %d markers" % (
        a.name, a.duration, app.ed_zoom, s0 / a.sr, (s0 + span) / a.sr, app.ed_cursor / a.sr, len(app.markers)), c.ACC)
    wh = max(c.h - 14, 6)
    ww = c.w - 3
    key = (app.source_id, s0, span, ww, wh, len(app.markers), tuple(app.markers[:64]), app.ed_cursor)
    if key != getattr(app, "_ed_key", None):
        app._ed_key = key
        cv = braille.waveform(a.data[s0:s0 + span], ww, wh)
        inside = [m - s0 for m in app.markers if s0 <= m < s0 + span]
        braille.markers(cv, inside, span)
        cx = int((app.ed_cursor - s0) / max(span, 1) * (cv.w - 1))
        cv.vline(cx, 0, cv.h - 1)
        app._ed_rows = cv.rows()
    for i, row in enumerate(app._ed_rows):
        c.put(3 + i, 1, row, c.OK)
    y = 3 + wh + 1
    s, e = app.ed_region()
    c.put(y, 1, "region under cursor: %.3f - %.3f s (%.0f ms)   p auditions it, o puts all regions on pads in time order" % (s / a.sr, e / a.sr, (e - s) * 1000.0 / a.sr), c.ACC)
    c.put(y + 1, 1, "markers: " + "  ".join("%.2f" % (m / a.sr) for m in app.markers[:14]) + (" ..." if len(app.markers) > 14 else ""), c.DIM)
    if app.loop_cands:
        c.put(y + 2, 1, "loops: " + "  ".join("%d) %.2fs %gbar %.2f" % (i + 1, l.start / a.sr, l.bars, l.score) for i, l in enumerate(app.loop_cands[:4])) + "   :loop use N", c.DIM)


# ------------------------------------------------------------------------------- sequencer
def seq(c: Ctx) -> None:
    app, e, pr = c.app, c.app.eng, c.app.pr
    lt = e.sched.local_tick(e.frame) if e.playing else None
    seq_block(c, 1, 3, lt, compact=False, rows=PADS_PER_BANK)
    pat = pr.pattern
    y = 3 + PADS_PER_BANK + 1
    ev = app.seq_event()
    info = "no step here" if ev is None else "vel %.2f  prob %d%%  ratchet x%d  length %d step(s)%s" % (
        ev.vel, round(ev.prob * 100), ev.ratchet, ev.length // pat.step_ticks, "  lock %s" % ev.plock if ev.plock else "")
    c.put(y, 1, "pad %d step %d  %s" % (app.sel + 1, app.cur_step + 1, info), c.ACC)
    # velocity lane for the selected pad (4 rows)
    if c.h - y - 6 >= 4:
        steps = pat.steps
        page = 32 if (c.w - 10) >= 64 else 16
        start = (app.cur_step // page) * page
        evs = {int(round(x.tick / pat.step_ticks)) % steps: x for x in pat.events if x.pad == app.sel}
        for row in range(4):
            line = ""
            for k in range(page):
                st = start + k
                x = evs.get(st)
                eighths = int(round((x.vel if x else 0.0) * 32))
                lvl = int(np.clip(eighths - (3 - row) * 8, 0, 8))
                line += GLYPHS[lvl] + " "
            c.put(y + 1 + row, 10, line, c.OK)
        c.put(y + 2, 1, "velocity", c.DIM)


# ------------------------------------------------------------------------------- mixer
def mix(c: Ctx) -> None:
    app, e = c.app, c.app.eng
    c.put(2, 1, " MIXER   pads feed buses (:padbus N); buses have effects, gain, mute and sidechain; sends feed delay and reverb", c.ACC)
    y = 4
    for i, row in enumerate(MIX_ROWS):
        sel = i == app.mix_sel
        a = (c.ACC | curses.A_BOLD) if sel else 0
        if row.startswith("bus"):
            b = e.buses[int(row[3]) - 1]
            fxn = " > ".join("%s%s" % (f.name, (" %.0fdB" % f.gr_db) if hasattr(f, "gr_db") and f.gr_db < -0.5 else "") for f in b.fx) or "-"
            used = sum(1 for p in app.pr.all_pads() if p.bus == b.index and not p.empty)
            c.put(y, 1, "%-6s %-10s %s %4.2f  %s %s  pads %-2d  sc:%s  fx: %s" % (row, b.name[:10], bar(b.gain, 14, 1.5), b.gain, bar(e.bus_peak[b.index], 10),
                                                                                  "MUTE" if b.mute else "    ", used, "-" if b.sidechain is None else b.sidechain + 1, fxn), a)
        elif row in ("send_a", "send_b"):
            f = e.fx_a if row == "send_a" else e.fx_b
            lvl = e.return_a if row == "send_a" else e.return_b
            c.put(y, 1, "%-6s %-10s %s %4.2f  %s" % (row, f.name, bar(lvl, 14), lvl, " ".join("%s=%s" % (k, ("%.2f" % v) if isinstance(v, float) else v) for k, v in f.params().items())), a)
        else:
            fxn = " > ".join(f.name for f in e.master_fx) or "-"
            c.put(y, 1, "%-6s %-10s %s %4.2f  L %s R %s  fx: %s" % (row, "", bar(e.master_gain, 14, 1.2), e.master_gain, bar(e.peak[0], 6), bar(e.peak[1], 6), fxn), a)
        y += 2
    c.put(y, 1, "effects: " + ", ".join(__import__("sampire.fx", fromlist=["x"]).list_effects()), c.DIM)


# ------------------------------------------------------------------------------- create
def create(c: Ctx) -> None:
    app = c.app
    a = app.source
    c.put(2, 1, " CREATIVE ENGINE", c.ACC)
    if a is not None:
        an = a.analysis
        c.put(3, 1, "source %s: %.1fs  %.1f BPM (conf %.2f)  key %s  %.1f dB peak  %.1f LUFS  %d onsets  %d%% percussive" % (
            a.name, a.duration, an.get("bpm", 0), an.get("bpm_confidence", 0), an.get("key", "?"), an.get("peak_db", 0), an.get("lufs", 0),
            len(an.get("onsets", [])), round(100 * an.get("percussive_ratio", 0))), c.DIM)
    else:
        c.put(3, 1, "no source: :demo, :open FILE, or drop a file here", c.DIM)
    cp = app.cp
    vals = {"pads": cp.pads, "mode": cp.mode, "sensitivity": "%.1f" % cp.sensitivity, "experimental": "%.1f" % cp.experimental,
            "keep_original": "%.1f" % cp.keep_original, "prefer": cp.prefer, "length": cp.length, "style": app.pat_style,
            "taste": "on" if cp.taste else "off"}
    for i, f in enumerate(KIT_FIELDS):
        sel = i == app.cp_sel
        c.put(5 + i, 2, "%s %-14s %s" % (">" if sel else " ", f, vals[f]), (c.ACC | curses.A_BOLD) if sel else 0)
    x2 = 44
    c.put(5, x2, "g  generate kit      p  generate pattern", c.DIM)
    c.put(6, x2, "y  keep (teach taste) n  reject (undo + teach)", c.DIM)
    c.put(7, x2, "provider: %s   (:provider command:PROGRAM)" % app.provider, c.DIM)
    c.put(9, x2, "last: %s" % (app.last_report or "-"), c.OK)
    c.put(11, x2, "taste: %d examples" % app.taste.n, c.ACC)
    for i, (k, w) in enumerate(app.taste.top(6)):
        c.put(12 + i, x2, "%-22s %+.2f %s" % (k, w, "#" * int(min(abs(w) * 4, 12))), c.OK if w > 0 else c.HOT)
    c.put(5 + len(KIT_FIELDS) + 1, 2, "try:  :prompt chop it into 12 pieces, pitch some down, reverse a few, two 4-bar patterns at 95 bpm with swing", c.DIM)


# ------------------------------------------------------------------------------- library browser
def browse(c: Ctx) -> None:
    app = c.app
    c.put(2, 1, " LIBRARY  %d results  query: %s  (/ to search: text bpm:85-95 key:C tag:kick rating:3 fav recent)" % (len(app.results), app.br_query or "-"), c.ACC)
    rows = c.h - 10
    start = max(min(app.br_sel - rows // 2, len(app.results) - rows), 0)
    c.put(3, 1, "%-30s %5s %5s %-9s %-4s %s" % ("name", "sec", "bpm", "key", "rate", "tags"), c.DIM)
    for i in range(rows):
        j = start + i
        if j >= len(app.results):
            break
        r = app.results[j]
        sel = j == app.br_sel
        sc = (" %.2f" % r.score) if r.score else ""
        c.put(4 + i, 1, "%-30s %5.1f %5.0f %-9s %-4s %s%s" % (("F " if r.fav else "  ") + r.name[:27], r.dur, r.bpm, r.key[:9], "*" * r.rating, ",".join(r.tags)[:40], sc),
              (c.ACC | curses.A_REVERSE) if sel else 0)
    r = app._cur_row()
    if r is not None:
        c.put(c.h - 6, 1, r.path, c.DIM)


# ------------------------------------------------------------------------------- diagnostics
def diag(c: Ctx) -> None:
    app, e = c.app, c.app.eng
    d = e.diagnostics()
    be = app.backend.info() if app.backend else {}
    from .. import analysis, dsp

    lines = [
        ("audio", "%s  %d Hz  block %d (%.1f ms)  device %s" % (be.get("name", "?"), d["sample_rate"], d["block"], d["latency_ms"], d["device_state"])),
        ("load", "cpu %.0f%%  voices %d/%d  underruns %d  input overruns %d" % (d["cpu"] * 100, d["voices"], d["max_voices"], d["underruns"], d["input_overruns"])),
        ("memory", "%.1f MB of samples held" % d["sample_memory_mb"]),
        ("transport", "%s  %.1f BPM  frames %d  external sync %s  clock out %s" % ("playing" if d["playing"] else "stopped", d["bpm"], d["frames"], d["external_sync"], d["clock_out"])),
        ("capture", "master %s  resample %s  input %s  monitor %s" % (d["master_recording"], d["resampling"], d["input_recording"], e.monitor_input)),
        ("stealing", "policy %s" % e.steal_policy),
        ("backends", "stretch %s  onsets %s  rubberband %s" % (dsp.STRETCH_BACKEND, analysis.ONSET_BACKEND, "found" if dsp.rubberband_available() else "not found")),
        ("midi", app.midi.status_text() if app.midi else "not started (:midi ports)"),
        ("library", "%d samples" % app.lib.count() if app._lib is not None else "not opened yet"),
        ("python", "%s  numpy %s" % (sys.version.split()[0], np.__version__)),
    ]
    c.put(2, 1, " DIAGNOSTICS", c.ACC)
    for i, (k, v) in enumerate(lines):
        bad = (k == "load" and (d["underruns"] or d["cpu"] > 0.7)) or (k == "audio" and d["device_state"] != "running")
        c.put(4 + i, 2, "%-10s %s" % (k, v), c.HOT if bad else 0)
    c.put(5 + len(lines), 2, "buses: " + "  ".join("%s %s" % (b.name, "peak %.2f" % e.bus_peak[b.index]) for b in e.buses), c.DIM)
