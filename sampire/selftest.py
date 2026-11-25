"""`sampire --selftest`: headless proof that the instrument is accurate. Needs no audio device or files."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Callable, List, Tuple

import numpy as np

from . import analysis, asset as assetmod, creative, demo, dsp, export, slicer
from .config import PPQ, SAMPLE_RATE
from .engine import Engine
from .pad import Pad
from .project import Project
from .recorder import RingBuffer, ThresholdRecorder
from .sequencer import Pattern, Scheduler

SR = SAMPLE_RATE


def _f0(y: np.ndarray) -> float:
    m = dsp.mono(y)
    sp = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    return float(np.argmax(sp)) * SR / len(m)


def _project_with_kit(bpm: float = 92.0, seed: int = 3) -> Project:
    pr = Project("selftest")
    pr.bpm, pr.seed = bpm, seed
    a = pr.add_asset(assetmod.asset_from_array(demo.drum_break(bpm, 2, with_melody=True), "break"))
    creative.generate_kit(pr, a, creative.KitParams(seed=seed, experimental=0.4))
    pt = creative.generate_pattern(pr, 0, 2, "break", 0.6, 0.7, 0.12, seed)
    pr.patterns[pt.name] = pt
    pr.current_pattern = pt.name
    return pr


# ------------------------------------------------------------------ checks (each returns a short detail string or raises)
def t_io_resample() -> str:
    x = demo.tone(1000.0, 1.0, 44100)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.wav")
        assetmod.save_audio(p, x, 44100, "FLOAT")
        a = assetmod.load_audio(p)
    assert a.sr == SR and abs(a.frames - SR) <= 2, "resampled length %d" % a.frames
    f = _f0(a.data[:, :])
    assert abs(f - 1000.0) < 2.0, "f0 %.1f" % f
    return "44.1k -> 48k, 1 kHz tone at %.1f Hz" % f


def t_onsets() -> str:
    tp = fp = fn = 0
    worst = 0.0
    for bpm in (80.0, 92.0, 120.0, 140.0):
        b = demo.drum_break(bpm, 2)
        tt = sorted({int(round(t * SR)) for t, _, _ in demo.drum_break_truth(bpm, 2)})
        pos, _ = analysis.detect_onsets(b, SR, 0.5)
        hit = set()
        for p in pos:
            d = [abs(p - q) for q in tt]
            j = int(np.argmin(d))
            if d[j] <= 0.006 * SR and j not in hit:
                hit.add(j)
                tp += 1
                worst = max(worst, d[j] * 1000.0 / SR)
            else:
                fp += 1
        fn += len(tt) - len(hit)
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    assert f1 > 0.85 and worst < 4.0, "F1 %.2f worst %.1f ms" % (f1, worst)
    return "F1 %.2f (P %.2f R %.2f), worst timing error %.1f ms" % (f1, pr, rc, worst)


def t_tempo() -> str:
    got = []
    for bpm in (80.0, 93.0, 120.0, 140.0):
        e = analysis.estimate_tempo(demo.drum_break(bpm, 2), SR)["bpm"]
        assert abs(e - bpm) < 2.0, "%.1f -> %.1f" % (bpm, e)
        got.append("%.0f>%.1f" % (bpm, e))
    return "  ".join(got)


def t_pitch_key_class() -> str:
    f, c = analysis.estimate_pitch(demo.tone(220.0, 1.0), SR)
    assert abs(f - 220.0) < 1.5 and c > 0.9, "f0 %.1f" % f
    key, _ = analysis.estimate_key(demo.drum_break(with_melody=True), SR)
    assert key == "C major", key
    labels = {k: analysis.classify_sample(v, SR) for k, v in (("kick", demo.kick()), ("snare", demo.snare()), ("hat", demo.hat()),
                                                               ("open_hat", demo.hat(True)), ("bass", demo.bass()))}
    assert all(k == v for k, v in labels.items()), labels
    return "220 Hz -> %.1f, key %s, hit classes %s" % (f, key, "/".join(labels.values()))


def t_slices() -> str:
    b = demo.drum_break(92.0, 2)
    for mode in ("transient", "hybrid", "grid", "equal", "silence"):
        sl = slicer.make_slices(mode, b, SR, count=16, bpm=92.0)
        assert len(sl) >= 8, "%s gave %d" % (mode, len(sl))
        assert all(s.end > s.start and s.start >= 0 and s.end <= len(b) for s in sl), mode
    sl = slicer.make_slices("transient", b, SR)
    t = np.arange(SR) / SR
    sine = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)
    worst = max(abs(float(sine[slicer.snap_zero_crossing(sine, i)])) for i in range(500, 40000, 777))
    assert worst < 0.07, "zero-cross snap left a step of %.3f" % worst
    return "5 modes ok, %d transient slices, zero-crossing snap leaves at most %.3f of a step" % (len(sl), worst)


def t_stretch_pitch() -> str:
    x = dsp.ensure_2d(demo.tone(440.0, 2.0))
    y = dsp.time_stretch(x, 1.5)
    assert abs(len(y) / len(x) - 1.5) < 0.005 and abs(_f0(y) - 440.0) < 3.0, "stretch f0 %.1f" % _f0(y)
    z = dsp.pitch_shift(x, 7.0)
    want = 440.0 * 2 ** (7 / 12)
    assert len(z) == len(x) and abs(_f0(z) / want - 1) < 0.01, "pitch f0 %.1f want %.1f" % (_f0(z), want)
    return "x1.5 keeps 440 Hz (%.0f), +7 st gives %.1f Hz (want %.1f), length kept" % (_f0(y), _f0(z), want)


def t_voices() -> str:
    pr = Project("v")
    a = pr.add_asset(assetmod.asset_from_array(demo.tone(300.0, 2.0), "t"))
    for i in range(40):
        p = pr.pad(i)
        p.asset_id, p.start, p.end, p.mode = a.id, 0, a.frames, "gate"
        p.release = 0.01
    p0, p1 = pr.pad(0), pr.pad(1)
    p0.choke_group = p1.choke_group = 2
    pr.render_all()
    e = Engine(pr, max_voices=32)
    for i in range(40):
        e.trigger(i, 0.8)
    e.render(0.05)
    live = [v for v in e.voices if not v.releasing]
    assert len(live) <= 32, "live voices %d" % len(live)
    # choke
    e2 = Engine(pr)
    e2.trigger(0, 1.0)
    e2.render(0.1)
    e2.trigger(1, 1.0)
    e2.render(0.05)
    assert not any(v.pad == 0 and not v.releasing for v in e2.voices), "choke group failed"
    # varispeed: pitch +12 plays twice as fast
    p2 = pr.pad(5)
    p2.pitch, p2.mode = 12.0, "one_shot"
    pr.pad(5).sample = None
    pr.render_all()
    e3 = Engine(pr)
    e3.trigger(5, 1.0)
    y = e3.render(0.5)
    assert abs(_f0(y[: int(0.5 * SR)]) - 600.0) < 8.0, "+12 st f0 %.1f" % _f0(y)
    return "stealing holds 32 voices, choke cuts in <60 ms, +12 st doubles frequency"


def t_scheduler_accuracy() -> str:
    p = Pattern("P", 1)
    p.toggle_step(0, 0)
    p.toggle_step(1, 4)
    p.toggle_step(2, 5)
    p.swing = 0.25
    res = []
    for block in (32, 64, 441, 4096):
        s = Scheduler(SR)
        s.patterns, s.bpm = {"P": p}, 120.0
        s.pattern_name = "P"
        s.start(0)
        evs, f = [], 0
        while f < SR * 8:
            evs += [(f + o, k, e.pad) for o, k, e, _ in s.advance(f, block)]
            f += block
        res.append([x for x in evs if x[0] < SR * 8])
    assert all(r == res[0] for r in res), "event frames differ between block sizes"
    ons = [f for f, k, pad in res[0] if pad == 2]
    assert ons[0] == int(round((5 * 240 + 0.25 * 240) * 60.0 / (120 * PPQ) * SR)), "swing position %d" % ons[0]
    last = [f for f, _, pad in res[0] if pad == 0][-1]
    assert last == 3 * 96000, "drift after 4 loops: %d" % last
    return "%d events identical for 4 block sizes, swing exact, zero drift over 4 loops" % len(res[0])


def t_block_invariance() -> str:
    outs = []
    for block in (64, 256, 1000, 4096):
        pr = _project_with_kit()
        pr.render_all()
        e = Engine(pr, block=block, seed=pr.seed)
        for pad in pr.all_pads():
            pad.send_a, pad.send_b = 0.15, 0.25
        e.play()
        outs.append(e.render(4.0, block))
    d = max(float(np.abs(o - outs[0]).max()) for o in outs[1:])
    assert d < 1e-4, "max difference %.2e" % d
    return "4 s render identical for block 64/256/1000/4096 (max diff %.1e, with delay and reverb)" % d


def t_determinism() -> str:
    a = export.render_pattern(_project_with_kit(seed=7), None, 1)
    b = export.render_pattern(_project_with_kit(seed=7), None, 1)
    c = export.render_pattern(_project_with_kit(seed=8), None, 1)
    assert np.array_equal(a, b), "same seed gave different audio"
    assert a.shape != c.shape or not np.array_equal(a, c), "different seed gave identical audio"
    return "same seed = identical audio, different seed = different"


def t_recorders() -> str:
    rb = RingBuffer(1000)
    for i in range(7):
        rb.write(np.full((300, 2), i, np.float32))
    assert list(rb.read_last(900)[::300, 0]) == [4, 5, 6]
    pr = _project_with_kit()
    pr.render_all()
    e = Engine(pr)
    e.play()
    e.start_master_recording()
    y = e.render(2.0)
    rec = e.stop_master_recording()
    assert np.array_equal(rec, y), "master recorder is not bit-exact"
    sb = e.skipback(1.0)
    assert np.array_equal(sb, y[-SR:]), "skip-back mismatch"
    sig = np.zeros((SR, 2), np.float32)
    sig[20000:20500] = 0.5
    tr = ThresholdRecorder(SR, -30, preroll_ms=10, silence_stop_s=0.05)
    for i in range(0, SR, 256):
        if tr.push(sig[i:i + 256]) == "done":
            break
    pre = int(np.argmax(np.abs(tr.result()[:, 0]) > 0.1))
    assert pre == 480, "pre-roll %d" % pre
    return "master recorder bit-exact, skip-back exact, threshold pre-roll 480 frames"


def t_project() -> str:
    pr = _project_with_kit()
    before = export.render_pattern(pr, None, 1)
    with tempfile.TemporaryDirectory() as d:
        pr.save(os.path.join(d, "proj"))
        pr2 = Project.load(os.path.join(d, "proj"))
        after = export.render_pattern(pr2, None, 1)
        assert np.allclose(before, after, atol=1e-4), "render changed after save/load"
        n_ev = len(pr2.pattern.events)
        pr2.checkpoint()
        pr2.pattern.clear()
        assert pr2.undo() and len(pr2.pattern.events) == n_ev, "undo failed"
        assert pr2.redo() and len(pr2.pattern.events) == 0, "redo failed"
        a = next(iter(pr2.assets.values()))
        os.rename(os.path.join(d, "proj", a.path), os.path.join(d, "proj", "audio", "moved.wav"))
        pr3 = Project.load(os.path.join(d, "proj"))
        assert a.id in pr3.assets, "relink by hash failed"
    return "save/load render identical, undo/redo ok, moved file relinked by hash"


def t_creative() -> str:
    pr = Project("c")
    a = pr.add_asset(assetmod.asset_from_array(demo.drum_break(92, 2, with_melody=True), "b"))
    rep = creative.generate_kit(pr, a, creative.KitParams(seed=1))
    assert rep.chosen == 16 and {"kick", "snare", "hat"} <= set(rep.labels), rep
    plan = creative.parse_instruction("chop it into 12 pieces, pitch some down, reverse a few, two 4-bar patterns at 95 bpm with swing")
    assert plan["kit"]["pads"] == 12 and plan["pattern"]["count"] == 2 and plan["pattern"]["bars"] == 4 and plan["bpm"] == 95.0, plan
    rep2, pats = creative.run_plan(pr, a, plan, 2)
    assert rep2.chosen == 12 and len(pats) == 2 and pr.bpm == 95.0
    fl = creative.rearrange_chops(pr, 0, 1, "shuffle", 4)
    assert len(fl.events) == 16
    return "16 pads (%s), prompt plan executed, chop-flip ok" % ",".join(sorted(rep.labels))


def t_export() -> str:
    pr = _project_with_kit()
    with tempfile.TemporaryDirectory() as d:
        stems = export.export_stems(pr, d)
        assert stems and all(os.path.getsize(f) > 1000 for f in stems)
        mid = export.pattern_to_midi(pr, pr.pattern, os.path.join(d, "p.mid"))
        raw = open(mid, "rb").read()
        assert raw[:4] == b"MThd" and b"MTrk" in raw
    return "%d stems + midi file written" % len(stems)


def t_performance() -> str:
    pr = Project("perf")
    a = pr.add_asset(assetmod.asset_from_array(demo.tone(220.0, 10.0), "t"))
    for i in range(32):
        p = pr.pad(i)
        p.asset_id, p.start, p.end = a.id, 0, a.frames
        p.pitch = 0.0 if i % 2 == 0 else 0.37          # half plain copies, half interpolated
        p.send_a = 0.1
    pr.render_all()
    e = Engine(pr, block=256)
    for i in range(32):
        e.trigger(i, 0.5)
    e.render(0.1)
    n = int(SR * 2.0)
    t0 = time.perf_counter()
    e.render(2.0)
    load = (time.perf_counter() - t0) / 2.0
    assert e.voice_count >= 30, "voices %d" % e.voice_count
    assert load < 0.6, "32 voices use %.0f%% of real time" % (100 * load)
    return "32 voices + sends at block 256: %.0f%% of real time (%.1fx headroom)" % (100 * load, 1.0 / max(load, 1e-6))


def t_limiter() -> str:
    pr = Project("l")
    a = pr.add_asset(assetmod.asset_from_array(np.ones((SR, 2), np.float32) * 0.99, "loud"))
    for i in range(16):
        p = pr.pad(i)
        p.asset_id, p.start, p.end = a.id, 0, a.frames
    pr.render_all()
    e = Engine(pr)
    e.master_gain = 1.0
    for i in range(16):
        e.trigger(i, 1.0)
    y = e.render(0.5)
    assert float(np.abs(y).max()) < 1.0, "master exceeded full scale"
    return "16 hot voices: peak %.3f < 1.0" % float(np.abs(y).max())


CHECKS: List[Tuple[str, Callable[[], str]]] = [
    ("file io and resampling", t_io_resample),
    ("onset detection", t_onsets),
    ("tempo detection", t_tempo),
    ("pitch, key and hit classes", t_pitch_key_class),
    ("slicing modes and clean cuts", t_slices),
    ("time stretch and pitch shift", t_stretch_pitch),
    ("voices, choke, varispeed", t_voices),
    ("sequencer timing", t_scheduler_accuracy),
    ("block size invariance", t_block_invariance),
    ("determinism", t_determinism),
    ("recording taps", t_recorders),
    ("project save, load, undo", t_project),
    ("creative engine", t_creative),
    ("exports", t_export),
    ("master limiter", t_limiter),
    ("performance headroom", t_performance),
]


def run(verbose: bool = True) -> int:
    fails = 0
    t_all = time.time()
    for name, fn in CHECKS:
        t0 = time.time()
        try:
            detail = fn()
            ok = True
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            detail, ok = "%s: %s" % (exc.__class__.__name__, exc), False
        fails += 0 if ok else 1
        if verbose:
            print("  %-4s %-30s %5.2fs  %s" % ("PASS" if ok else "FAIL", name, time.time() - t0, detail))
    print("\n%d/%d checks passed in %.1fs" % (len(CHECKS) - fails, len(CHECKS), time.time() - t_all))
    return 1 if fails else 0
