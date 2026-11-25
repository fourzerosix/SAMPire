import os
import select
import struct
import sys
import time

import pytest

from sampire import cli
from sampire.ui.tui import App, build_app


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SAMPIRE_LIBRARY", str(tmp_path / "lib.db"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))


def _app():
    args = cli.build_parser().parse_args(["--demo", "--no-output"])
    return build_app(args, {})


def test_app_commands_without_a_terminal(tmp_path):
    app = _app()
    app.command("demo")
    assert app.source is not None and not app.pad.empty
    cmds = ["pattern four_floor", "rev", "pitch -3", "kit 12", "flip", "vary 0.4", "mode gate", "bogus", "fx list",
            "fx bus1 add compressor threshold_db=-20", "fx bus2 sidechain 1", "fx master add eq low_db=2", "fx send_a delay time_s=0.25",
            "padbus 2", "mk auto", "mk apply", "loop find", "loop use 1", "stems", "like 2", "chromatic on", "chromatic off",
            "roll 16 4", "lfo pitch 5 0.3", "pfilter lp 1200", "variant add", "steal quietest", "taste", "diag", "view mix", "view 1",
            "backend", "midi ports", "song add P1 2", "song on", "song off", "lib info", "find nothing"]
    for c in cmds:
        app.command(c)
    assert not any(m.startswith("error") for m in app.log), [m for m in app.log if m.startswith("error")]
    app.command("prompt chop into 8 pieces, two 2-bar patterns at 100 bpm")
    assert app.pr.bpm == 100.0
    app.command("keep")
    app.command("kit")
    app.command("reject")
    assert app.taste.n > 0
    app.command("save %s" % (tmp_path / "proj"))
    assert any(m.startswith("saved") for m in app.log)
    app.command("recover")
    assert any("recovered" in m for m in app.log)
    assert any("unknown command" in m for m in app.log)


def test_paste_loads_files_and_folders(tmp_path):
    from sampire import asset, demo

    f = tmp_path / "my loop.wav"
    asset.save_audio(str(f), demo.drum_break(100, 2), 48000)
    app = _app()
    app.on_paste("'%s'\n" % f)
    assert app.source is not None and app.source.name == "my loop"
    app.on_paste("not-a-real-file.txt")
    assert any("not an audio file" in m for m in app.log)


def _session(steps, cols=130, rows=42):
    pytest.importorskip("pyte")
    import fcntl
    import pty
    import termios

    import pyte

    pid, fd = pty.fork()
    if pid == 0:
        os.environ.update(TERM="xterm-256color", LANG="C.UTF-8")
        os.execvp(sys.executable, [sys.executable, "-m", "sampire", "--demo", "--no-output", "--no-midi"])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    screen = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(screen)

    def pump(t):
        end = time.time() + t
        while time.time() < end:
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    return
                if not data:
                    return
                stream.feed(data)

    pump(3.0)
    shots = {}
    for label, keys, wait in steps:
        os.write(fd, keys if isinstance(keys, bytes) else keys.encode())
        pump(wait)
        shots[label] = "\n".join(screen.display)
    os.write(fd, b"\x03")
    pump(1.5)
    _, status = os.waitpid(pid, 0)
    return shots, os.WEXITSTATUS(status)


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
def test_real_curses_ui_every_workspace_and_drag_drop(tmp_path):
    from sampire import asset, demo

    drop = tmp_path / "dropped.wav"
    asset.save_audio(str(drop), demo.drum_break(100, 2), 48000)
    steps = [("perf", " ", 1.0), ("edit", b"\x1bOQ", 0.8), ("seq", b"\x1bOR", 0.8), ("mix", ":view 4\r", 0.8),
             ("fx", ":fx bus1 add compressor\r", 0.8), ("create", ":view create\r", 0.8), ("browse", ":view browse\r", 0.8),
             ("diag", ":view diag\r", 0.8), ("back", ":view 1\r", 0.5), ("paste", b"\x1b[200~" + str(drop).encode() + b"\x1b[201~", 2.0)]
    shots, code = _session(steps)
    want = {"perf": "PLAY", "edit": "markers", "seq": "STEPS", "mix": "MIXER", "fx": "compressor", "create": "CREATIVE ENGINE",
            "browse": "LIBRARY", "diag": "DIAGNOSTICS", "paste": "loaded 1 file"}
    for label, needle in want.items():
        assert needle in shots[label], "%s view missing %r" % (label, needle)
    assert code == 0
