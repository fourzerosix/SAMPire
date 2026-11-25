import numpy as np

from sampire import asset, creative, demo, dsp, export, slicer
from sampire.project import Project
from sampire.recipe import OPS, Recipe
from sampire.sequencer import Pattern


def test_every_recipe_op_runs_and_stays_finite():
    x = demo.drum_break(92, 1)[:30000]
    for name in OPS:
        params = {"ms": 50} if name == "truncate" else {}
        y = Recipe().add(name, **params).apply(x, 48000)
        assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all(), name


def test_recipe_is_deterministic_per_seed():
    x = demo.drum_break(92, 1)[:30000]
    a = Recipe([{"op": "granular"}], seed=5).apply(x, 48000)
    b = Recipe([{"op": "granular"}], seed=5).apply(x, 48000)
    c = Recipe([{"op": "granular"}], seed=6).apply(x, 48000)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_euclid_and_quantize():
    p = Pattern("E", 1)
    p.euclid(0, 5, 16)
    assert len(p.events) == 5
    p.events[0].tick += 37
    p.quantize(16)
    assert all(e.tick % p.step_ticks == 0 for e in p.events)


def test_empty_and_silent_inputs_do_not_crash():
    pr = Project("e")
    a = pr.add_asset(asset.asset_from_array(np.zeros((48000, 2), np.float32), "silence"))
    rep = creative.generate_kit(pr, a)
    assert rep.chosen == 0 and rep.notes
    assert slicer.make_slices("transient", a.data, 48000) == []


def test_midi_file_header():
    pr = Project("m")
    pr.pad(0).midi_note = 36
    pat = Pattern("P", 1)
    pat.add(0, 0, 1.0)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        f = export.pattern_to_midi(pr, pat, os.path.join(d, "a.mid"))
        raw = open(f, "rb").read()
    assert raw[:4] == b"MThd" and raw[12:14] == (960).to_bytes(2, "big")
