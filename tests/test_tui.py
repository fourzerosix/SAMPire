import os
import select
import struct
import sys
import time

import pytest

from sampire import cli
from sampire.ui.tui import App, build_app


def _app():
    args = cli.build_parser().parse_args(["--demo", "--no-output"])
    return build_app(args, {})


def test_app_commands_without_a_terminal():
    app = _app()
    app.command("demo")
    assert app.source is not None and not app.pad.empty
    for c in ("pattern four_floor", "rev", "pitch -3", "kit 12", "flip", "vary 0.4", "mode gate", "bogus", "fx reverb 0.3"):
        app.command(c)
    app.command("prompt chop into 8 pieces, two 2-bar patterns at 100 bpm")
    assert app.pr.bpm == 100.0
    app.undo()
    assert any("unknown command" in m for m in app.log)


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
def test_real_curses_ui_starts_draws_and_quits():
    pyte = pytest.importorskip("pyte")
    import fcntl
    import pty
    import termios

    pid, fd = pty.fork()
    if pid == 0:
        os.environ.update(TERM="xterm-256color", LANG="C.UTF-8")
        os.execvp(sys.executable, [sys.executable, "-m", "sampire", "--demo", "--no-output"])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    screen = pyte.Screen(120, 40)
    stream = pyte.ByteStream(screen)

    def pump(t):
        end = time.time() + t
        while time.time() < end:
            r, _, _ = select.select([fd], [], [], 0.1)
            if r:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    return
                if not data:
                    return
                stream.feed(data)

    pump(3.0)
    text = "\n".join(screen.display)
    assert "SAMPIRE" in text and "PADS" in text and "kick" in text
    os.write(fd, b" ")
    pump(1.0)
    assert "PLAY" in "\n".join(screen.display)
    os.write(fd, b"\x03")
    pump(1.5)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
