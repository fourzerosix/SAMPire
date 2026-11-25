"""Curses runner. App (app.py) holds the logic, views.py draws; this file owns the terminal."""

from __future__ import annotations

import curses
import locale
import os
import sys
import time

from ..backend import open_backend
from ..config import save_config
from ..engine import Engine
from ..project import Project, ProjectError
from .app import App
from .themes import THEMES
from . import views

__all__ = ["App", "build_app", "run_tui"]


def build_app(args, cfg) -> App:
    pr = Project("untitled")
    if args.project and os.path.exists(os.path.join(args.project, "project.json")):
        try:
            pr = Project.load(args.project)
        except ProjectError as exc:
            print("could not open project: %s" % exc, file=sys.stderr)
            try:
                pr = Project.recover(args.project)
                print("recovered from autosave", file=sys.stderr)
            except ProjectError:
                pass
    if args.project and not pr.path:
        pr.path = os.path.abspath(args.project)
    if args.seed is not None:
        pr.seed = args.seed
    eng = Engine(pr, block=args.block or int(cfg.get("block", 256)), seed=pr.seed)
    status = "welcome. :demo loads a built-in break, :open FILE or drop a file, ? shows keys, F1-F7 switch views"
    app = App(pr, eng, None, status, cfg)
    if args.seed is not None:
        app.seed = args.seed
    app.allow_python = bool(getattr(args, "allow_python_provider", False))
    if getattr(args, "provider", None):
        app.provider = args.provider
    return app


def _apply_theme(app: App) -> None:
    fg, acc, hot, ok, dim = THEMES[app.theme]
    try:
        curses.use_default_colors()
        for i, col in enumerate((fg, acc, hot, ok, dim), start=1):
            curses.init_pair(i, col if curses.COLORS > 8 or col < 8 else -1, -1)
    except curses.error:
        pass


def _read_escape(stdscr) -> str:
    """After an ESC: collect what follows within a few ms (bracketed paste markers, unknown sequences)."""
    stdscr.nodelay(True)
    buf = ""
    t0 = time.time()
    while time.time() - t0 < 0.02 and len(buf) < 8:
        ch = stdscr.getch()
        if ch == -1:
            time.sleep(0.002)
            continue
        buf += chr(ch) if ch < 256 else ""
    stdscr.nodelay(False)
    stdscr.timeout(33)
    return buf


def _read_paste(stdscr) -> str:
    stdscr.nodelay(True)
    text, t0 = "", time.time()
    while time.time() - t0 < 2.0 and len(text) < 8192:
        ch = stdscr.getch()
        if ch == -1:
            time.sleep(0.002)
            continue
        text += chr(ch) if ch < 0x110000 else ""
        if text.endswith("\x1b[201~"):
            text = text[:-6]
            break
    stdscr.nodelay(False)
    stdscr.timeout(33)
    return text


def run_tui(args, cfg) -> int:
    locale.setlocale(locale.LC_ALL, "")
    os.environ.setdefault("ESCDELAY", "25")
    app = build_app(args, cfg)
    dev = args.device
    if dev is not None and str(dev).isdigit():
        dev = int(dev)
    be, msg = open_backend(app.eng, dev, args.input_device, args.no_output, want_input=args.input_device is not None)
    app.backend = be
    app.say(msg)
    try:
        from ..midi import available

        if available() and not getattr(args, "no_midi", False):
            app._midi()
    except Exception:
        pass

    def loop(stdscr):
        curses.raw()
        curses.curs_set(0)
        stdscr.keypad(True)
        stdscr.timeout(33)
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS)
        except curses.error:
            pass
        sys.stdout.write("\x1b[?2004h")                      # bracketed paste: dropped files arrive as one paste
        sys.stdout.flush()
        _apply_theme(app)
        if args.demo:
            app.command("demo")
        for f in args.files:
            try:
                app.load_source(f)
            except Exception as exc:  # noqa: BLE001
                app.say("could not load %s: %s" % (f, exc))
        last_auto = time.time()
        last_theme = app.theme
        while not app.quit:
            if app.theme != last_theme:
                _apply_theme(app)
                last_theme = app.theme
            views.draw(stdscr, app)
            ch = stdscr.getch()
            if ch == 27 and not app.cmd_mode:
                seq = _read_escape(stdscr)
                if seq.startswith("[200~"):
                    app.on_paste(seq[5:] + _read_paste(stdscr))
                    ch = -1
                elif seq == "":
                    pass
                else:
                    ch = -1
            elif ch == 27 and app.cmd_mode:
                seq = _read_escape(stdscr)
                if seq.startswith("[200~"):
                    app.on_paste(seq[5:] + _read_paste(stdscr))
                    ch = -1
            try:
                app.handle_key(ch)
                app.tick()
            except Exception as exc:  # noqa: BLE001 - never let a bug drop the user out of the instrument
                app.say("internal error: %s" % exc)
            if time.time() - last_auto > 60:
                app.pr.autosave()
                last_auto = time.time()

    try:
        curses.wrapper(loop)
    finally:
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
        be.stop()
        if app.midi is not None:
            app.midi.stop()
        app.pr.autosave()
        app.cfg["seed"] = app.seed
        app.cfg["theme"] = app.theme
        save_config(app.cfg)
    return 0
