import os

import pytest

from sampire import asset, cli, demo


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SAMPIRE_LIBRARY", str(tmp_path / "lib.db"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))


@pytest.fixture
def loop(tmp_path):
    f = tmp_path / "loop.wav"
    asset.save_audio(str(f), demo.drum_break(100, 2, with_melody=True), 48000)
    return str(f)


def test_analyze_chop_render(loop, tmp_path, capsys):
    assert cli.main(["--analyze", loop]) == 0
    assert "tempo" in capsys.readouterr().out
    out = str(tmp_path / "slices")
    assert cli.main(["--chop", loop, "--out", out]) == 0
    assert len(os.listdir(out)) >= 8
    beat = str(tmp_path / "beat.wav")
    assert cli.main(["--render", loop, "--seed", "3", "--out", beat, "--stems", "--bus-stems", "--lufs", "-18"]) == 0
    assert os.path.getsize(beat) > 10000
    assert os.path.isdir(beat[:-4] + "_stems") and os.path.isdir(beat[:-4] + "_busstems")


def test_render_is_deterministic(loop, tmp_path):
    a, b = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    cli.main(["--render", loop, "--seed", "9", "--out", a])
    cli.main(["--render", loop, "--seed", "9", "--out", b])
    assert open(a, "rb").read() == open(b, "rb").read()


def test_split_loops_library(loop, tmp_path, capsys):
    out = str(tmp_path / "st")
    assert cli.main(["--split", loop, "--out", out]) == 0
    assert len(os.listdir(out)) == 3
    assert cli.main(["--find-loops", loop]) in (0, 1)
    assert cli.main(["--lib-scan", str(tmp_path)]) == 0
    capsys.readouterr()
    assert cli.main(["--find", "loop"]) == 0
    assert "loop" in capsys.readouterr().out
    assert cli.main(["--similar", loop]) == 0


def test_diagnostics_soak_ports(capsys):
    assert cli.main(["--diagnostics"]) == 0
    assert '"voices"' in capsys.readouterr().out
    assert cli.main(["--soak", "3"]) == 0
    assert cli.main(["--midi-ports"]) == 0
    assert cli.main(["--list-devices"]) == 0
