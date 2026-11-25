"""`sampire --selftest`: headless proof that the instrument is accurate. Needs no audio device or files."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Callable, List, Tuple

import numpy as np

from . import analysis, asset as assetmod, creative, demo, dsp, export, fx as fxmod, library, loops, midi, providers, slicer, stems, taste
from .bus import NUM_BUSES
from .config import PPQ, SAMPLE_RATE
from .engine import Engine
from .pad import Pad
from .project import Project
from .recorder import RingBuffer, ThresholdRecorder
from .sequencer import Pattern, Scheduler

SR = SAMPLE_RATE


class Skip(Exception):
    """Raised by a check whose optional dependency is not installed."""


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


# ------------------------------------------------------------------ v0.2 checks
def _tone_project(n_pads: int = 4, secs: float = 4.0) -> Project:
    pr = Project("t")
    a = pr.add_asset(assetmod.asset_from_array(demo.tone(220.0, secs), "t"))
    for i in range(n_pads):
        p = pr.pad(i)
        p.asset_id, p.start, p.end, p.mode = a.id, 0, a.frames, "loop"
    pr.render_all()
    return pr


def t_fx_rack() -> str:
    x = demo.drum_break(92, 1)[: SR * 2]
    worst = 0.0
    for name in fxmod.list_effects():
        if name == "pedalboard":
            continue
        kw = {"mix": 0.5} if name in ("delay", "reverb") else {}
        outs = []
        for block in (64, 1000):
            f = fxmod.make_fx(name, SR, **kw)
            outs.append(np.concatenate([f.process(x[i:i + block]) for i in range(0, len(x), block)]))
        d = float(np.abs(outs[0] - outs[1]).max())
        assert d < 1e-4 and np.isfinite(outs[0]).all(), "%s differs by %.2e across block sizes" % (name, d)
        worst = max(worst, d)
        assert fxmod.from_spec(fxmod.make_fx(name, SR, **kw).spec(), SR).spec() == fxmod.make_fx(name, SR, **kw).spec(), name
    c = fxmod.make_fx("compressor", SR, threshold_db=-24, ratio=10, attack_ms=1)
    y = c.process(x[:SR])
    assert float(np.abs(y).max()) < 0.75 * float(np.abs(x[:SR]).max()), "compressor did not reduce peaks"
    e = fxmod.make_fx("eq", SR, high_db=-12, high_hz=4000)
    noise = np.random.default_rng(0).normal(0, 0.2, (SR, 2)).astype(np.float32)
    ye = e.process(noise)
    hf = lambda v: float(np.sum(np.abs(np.fft.rfft(v[:, 0])[len(v) // 4:]) ** 2))
    assert hf(ye) < 0.2 * hf(noise), "eq high cut missing"
    return "%d effects exact across block sizes (worst %.1e), specs round-trip, compressor and eq work" % (len(fxmod.list_effects()) - 1, worst)


def t_buses_sidechain() -> str:
    pr = _tone_project()
    pr.pad(1).bus = 1
    pr.pad(1).pitch = 7

    def run(key_on: bool, only=0):
        e = Engine(pr)
        e.only_bus, e.stem_mode, e.master_gain = only, True, 1.0
        e.buses[0].fx = [fxmod.make_fx("compressor", SR, threshold_db=-40, ratio=20, attack_ms=1, release_ms=100)]
        e.buses[0].sidechain = 1
        e.trigger(0, 0.5)
        if key_on:
            e.trigger(1, 1.0)
        e.render(0.2)
        y = e.render(0.2)
        return float(np.sqrt(np.mean(y ** 2)))

    free, ducked = run(False), run(True)
    assert ducked < 0.2 * free, "sidechain did not duck (%.3f vs %.3f)" % (ducked, free)

    def r(only):
        e = Engine(pr)
        e.master_gain, e.only_bus, e.stem_mode = 0.3, only, True
        e.buses[0].fx = []
        e.trigger(0, 0.8)
        e.trigger(1, 0.8)
        return e.render(0.4)

    full, s0, s1 = r(None), r(0), r(1)
    assert float(np.abs(full - (s0 + s1)).max()) < 1e-5, "bus stems do not sum to the mix"
    e = Engine(pr)
    e.buses[0].mute = True
    e.trigger(0, 1.0)
    assert float(np.abs(e.render(0.1)).max()) < 1e-6, "muted bus is audible"
    return "sidechain ducks %.0fx, bus stems sum to the mix, mute silences" % (free / max(ducked, 1e-9))


def t_voice_modulation() -> str:
    rng = np.random.default_rng(1)
    pr = Project("n")
    a = pr.add_asset(assetmod.asset_from_array(rng.normal(0, 0.3, (SR, 2)).astype(np.float32), "n"))
    p = pr.pad(0)
    p.asset_id, p.start, p.end, p.filter_kind, p.filter_cut = a.id, 0, a.frames, "lp", 500.0
    pr.render_all()
    e = Engine(pr)
    e.master_gain = 1.0
    e.trigger(0, 1.0)
    y = e.render(0.8)
    sp = np.abs(np.fft.rfft(dsp.mono(y))) ** 2
    fr = np.fft.rfftfreq(len(y), 1.0 / SR)
    ratio = sp[fr > 4000].sum() / sp[fr < 400].sum()
    assert ratio < 0.01, "realtime lowpass leaves %.3f HF/LF" % ratio
    outs = []
    for block in (64, 256, 1000):
        pr = _tone_project(3)
        pr.pad(0).filter_kind, pr.pad(0).filter_cut, pr.pad(0).lfo_target, pr.pad(0).lfo_depth, pr.pad(0).lfo_rate = "lp", 800.0, "cutoff", 1.0, 3.0
        pr.pad(1).lfo_target, pr.pad(1).lfo_depth, pr.pad(1).lfo_rate = "pitch", 0.5, 6.0
        pr.pad(2).lfo_target, pr.pad(2).lfo_depth, pr.pad(2).lfo_rate = "pan", 1.0, 2.0
        eng = Engine(pr, block=block)
        for i in range(3):
            eng.trigger(i, 1.0)
        outs.append(eng.render(1.0, block))
    d = max(float(np.abs(o - outs[0]).max()) for o in outs[1:])
    assert d < 1e-5, "LFO voices differ across block sizes by %.2e" % d
    # chromatic keys, bend, roll, protected voices, steal policy
    pr = _tone_project(4)
    e = Engine(pr)
    e.set_pitch_bend(12.0)
    e.trigger(0, 1.0)
    y = e.render(0.5)
    f = analysis.estimate_pitch(y[: SR // 2], SR)[0]
    assert abs(f - 440.0) < 6.0, "bend +12 st gave %.1f Hz" % f
    e = Engine(_tone_project(4))
    e.roll(0, 16, 4)
    e.render(1.2)
    assert e.flash.get(0) is not None
    pr = _tone_project(4)
    pr.pad(0).protect = True
    e = Engine(pr, max_voices=2)
    for i in range(4):
        e.trigger(i, 1.0)
    e.render(0.02)
    assert any(v.pad == 0 and not v.releasing for v in e.voices), "protected voice was stolen"
    return "lowpass %.4f HF/LF, LFO cutoff/pitch/pan exact across blocks (%.1e), bend, roll, protect ok" % (ratio, d)


def t_stems_loops_embed() -> str:
    drums = demo.drum_break(92, 2)
    chord = np.zeros_like(drums)
    chord[: len(demo.chord())] = dsp.ensure_2d(demo.chord())
    mix_ = dsp.ensure_2d(drums + chord)
    parts = stems.separate_dsp(mix_, SR)
    assert float(np.abs(sum(parts.values()) - mix_).max()) < 1e-3, "stems do not sum to the input"

    def corr(a, b):
        a, b = dsp.mono(a), dsp.mono(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    c_d, c_o = corr(parts["drums"], drums), corr(parts["other"] + parts["bass"], chord)
    assert c_d > 0.85 and c_o > 0.65, "separation quality drums %.2f chord %.2f" % (c_d, c_o)
    bar = demo.drum_break(100, 2)[: int(4 * 60 / 100 * SR)]
    sig = np.concatenate([np.zeros((int(0.3 * SR), 2), np.float32)] + [bar] * 4)
    cands = loops.find_loops(sig, SR)
    assert cands and abs(cands[0].length / SR - 2.4) < 0.03 and cands[0].bars == 1.0, cands[:1]
    seam = loops.make_seamless(bar, 30, SR)
    assert float(np.abs(seam[0] - seam[-1]).max()) < 0.01, "loop seam is not continuous"
    k = analysis.embedding(demo.kick(), SR)
    k2 = analysis.embedding(dsp.pitch_shift(dsp.ensure_2d(demo.kick()), -2), SR)
    h = analysis.embedding(demo.hat(), SR)
    assert analysis.similarity(k, k2) > analysis.similarity(k, h) + 0.2, "embedding does not rank a kick closer to a kick than to a hat"
    return "stems sum exactly (drums r=%.2f, harmonic r=%.2f), 1 bar loop found, seam clean, embeddings rank timbre" % (c_d, c_o)


def t_library() -> str:
    import zipfile

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "Drums", "kicks"))
        os.makedirs(os.path.join(d, "loops"))
        w = lambda rel, x: assetmod.save_audio(os.path.join(d, rel), x, SR)
        w("Drums/kicks/kick_deep.wav", demo.kick())
        w("Drums/kicks/kick_deep_copy.wav", demo.kick())
        w("Drums/kicks/kick_low.wav", dsp.pitch_shift(dsp.ensure_2d(demo.kick()), -2))
        w("Drums/snare_tight.wav", demo.snare())
        w("Drums/hat_closed.wav", demo.hat())
        w("loops/break_92bpm_Cmaj.wav", demo.drum_break(92, 2, with_melody=True))
        lib = library.Library(os.path.join(d, "lib.db"))
        added, _ = lib.scan([d])
        assert added == 6 and lib.scan([d]) == (0, 6), "scan/skip unchanged"
        assert [r.name for r in lib.find("bpm:85-100")] == ["break_92bpm_Cmaj"]
        assert {r.name for r in lib.find("tag:kick")} == {"kick_deep", "kick_deep_copy", "kick_low"}
        k = lib.search("kick_deep")[0]
        lib.add_tag(k.id, "punchy")
        lib.set_rating(k.id, 5)
        lib.set_favorite(k.id)
        assert [r.name for r in lib.find("fav rating:4 tag:punchy")] == ["kick_deep"]
        sim = lib.similar(k.id, 3)
        assert sim[0].name == "kick_deep_copy" and sim[1].name == "kick_low", [r.name for r in sim]
        assert [[r.name for r in g] for g in lib.duplicates()] == [["kick_deep", "kick_deep_copy"]]
        evil = os.path.join(d, "evil.zip")
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../evil.wav", b"x")
        try:
            lib.import_pack(evil, os.path.join(d, "evil_out"))
            raise AssertionError("zip slip not blocked")
        except ValueError:
            pass
        assert not os.path.exists(os.path.join(d, "evil.wav"))
        good = os.path.join(d, "good.zip")
        with zipfile.ZipFile(good, "w") as zf:
            zf.write(os.path.join(d, "Drums", "snare_tight.wav"), "Pack/Snares/snare_a.wav")
        dest, n = lib.import_pack(good, os.path.join(d, "pack_out"))
        assert n == 1 and "snare" in lib.find("snare_a")[0].tags
        lib.close()
    return "scan, filters, tags, rating, similarity, duplicates, zip-slip blocked, pack import tagged"


def t_midi() -> str:
    from types import SimpleNamespace as NS

    with tempfile.TemporaryDirectory() as d:
        pr = _tone_project(4)
        e = Engine(pr)
        r = midi.MidiRouter(e, midi.MidiMap(path=os.path.join(d, "m.json")))
        r.handle(NS(type="note_on", note=36, velocity=100))
        e.render(0.02)
        assert e.voice_count == 1, "note did not trigger its pad"
        r.handle(NS(type="control_change", control=7, value=127))
        assert abs(e.master_gain - 1.2) < 1e-6
        e.selected_pad = 2
        r.handle(NS(type="control_change", control=16, value=127))
        assert abs(pr.pad(2).gain_db - 6.0) < 1e-6
        r.handle(NS(type="program_change", program=2))
        assert e.active_bank == 2
        r.learn("bus1.gain")
        r.handle(NS(type="control_change", control=50, value=0))
        r.handle(NS(type="control_change", control=50, value=127))
        assert abs(e.buses[0].gain - 1.5) < 1e-6, "learned CC did not apply"
        r.learn("transport.toggle")
        r.handle(NS(type="note_on", note=99, velocity=90))
        r.handle(NS(type="note_on", note=99, velocity=90))
        e.render(0.01)
        assert e.playing, "learned transport note failed"
        t = [0.0]
        f = midi.ClockFollower(e, now=lambda: t[0])
        r.follower = f
        r.handle(NS(type="start"))
        for _ in range(60):
            t[0] += 60 / 100 / 24
            r.handle(NS(type="clock"))
        e.render(0.01)
        assert abs(e.sched.bpm - 100.0) < 0.2, "clock follow bpm %.2f" % e.sched.bpm
        pr2 = _tone_project(1)
        pr2.bpm = 120
        e2 = Engine(pr2)
        e2.clock_out = True
        e2.sync_from_project()
        e2.play()
        e2.render(2.0)
        n = 0
        while not e2.clock_q.empty():
            n += e2.clock_q.get() == "clock"
        assert abs(n - 96) <= 1, "clock out pulses %d" % n

        class Port:
            def close(self):
                pass

        class Fake:
            names = ["A"]

            def get_input_names(self):
                return list(self.names)

            def get_output_names(self):
                return ["Out"]

            def open_input(self, name, callback=None):
                return Port()

        fb, evs = Fake(), []
        m = midi.MidiManager(e, backend=fb, on_event=evs.append, poll_s=0.05)
        m.start()
        fb.names = ["A", "B"]
        time.sleep(0.25)
        assert set(m.status()["inputs"]) == {"A", "B"}
        fb.names = ["B"]
        time.sleep(0.25)
        assert m.status()["inputs"] == ["B"] and any("disconnected" in x for x in evs)
        m.stop()
    return "notes, CC, learn, program change, clock in (100 bpm), clock out (96 pulses), hot-plug survive unplug"


def t_providers_taste() -> str:
    bad = providers.validate_plan({"kit": {"pads": 999, "mode": "rm -rf", "ops": ["reverse", "os.system"], "evil": 1},
                                   "pattern": {"count": 99, "style": "x"}, "bpm": 1e9, "path": "/etc/passwd"})
    assert bad["kit"] == {"pads": 16, "ops": ["reverse"]} and bad["bpm"] == 300.0 and "path" not in bad and bad["pattern"]["count"] == 8
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "p.py")
        with open(script, "w") as fh:
            fh.write("import sys,json\njson.load(sys.stdin)\nprint(json.dumps({'kit':{'pads':6},'pattern':{'count':1,'bars':1,'style':'euclid'},'bpm':101}))")
        import sys as _sys

        plan = providers.plan_from_prompt("x", "command:%s %s" % (_sys.executable, script))
        assert plan["kit"]["pads"] == 6 and plan["bpm"] == 101.0 and plan["pattern"]["style"] == "euclid"
        try:
            providers.get("python:os:getcwd")
            raise AssertionError("python provider allowed without opt-in")
        except PermissionError:
            pass
        t = taste.Taste(path=os.path.join(d, "taste.json"))
        ops = ["reverse", "crush", "stutter"]
        assert np.allclose(t.preference("op:", ops), 1 / 3)
        for _ in range(6):
            t.record([{"op:reverse": 1.0}], True)
            t.record([{"op:crush": 1.0}], False)
        p = t.preference("op:", ops)
        assert p[0] > 0.5 and p[1] < 0.2, p
        assert taste.Taste(path=os.path.join(d, "taste.json")).n == 12, "taste not persisted"
        pr = Project("t")
        a = pr.add_asset(assetmod.asset_from_array(demo.drum_break(92, 2), "b"))
        base = creative.generate_kit(pr, a, creative.KitParams(seed=5, experimental=1.0))
        assert base.chosen == 16
        k2 = creative.generate_kit(pr, a, creative.KitParams(seed=5, experimental=1.0, taste=True), taste_model=t)
        ops_used = [o["op"] for pd in pr.banks[0].pads for o in pd.recipe.ops]
        assert "crush" not in ops_used or ops_used.count("reverse") >= ops_used.count("crush"), ops_used
    return "plans validated and clamped, command provider works, python provider gated, taste learns and persists"


def t_loudness_busstems() -> str:
    x = demo.drum_break(92, 2, with_melody=True)
    g = analysis.loudness_integrated(x, SR)
    assert -40 < g < -5 and abs(g - analysis.loudness_lufs(x, SR)) < 1.0
    tone = np.concatenate([demo.tone(220, 3.0)] * 1)
    y, info = dsp.normalize_lufs(dsp.ensure_2d(tone), SR, -20.0, -1.0)
    assert abs(info["after"] + 20.0) < 0.3 and info["true_peak_db"] <= -0.99, info
    pr = _project_with_kit()
    for i, p in enumerate(pr.banks[0].pads):
        p.bus = i % 2
    e = Engine(pr)
    e.buses[1].fx = [fxmod.make_fx("eq", SR, high_db=-6)]
    e.commit_settings()
    with tempfile.TemporaryDirectory() as d:
        files = export.export_bus_stems(pr, d)
        assert len(files) == 2
        stem_sum = sum(assetmod.read_audio(f)[0][: 100000] for f in files)
        ref = Engine(pr)
        ref.stem_mode = True
        ref.play()
        ref_audio = ref.render(len(stem_sum) / SR + 0.1)[: len(stem_sum)]
        assert float(np.abs(stem_sum - ref_audio).max()) < 1e-4, "bus stems with effects do not sum to the mix"
    z = export.render_pattern(pr, None, 1, lufs=-18.0)
    return "gated LUFS %.1f, normalise hits %.1f LUFS under a -1 dB true-peak ceiling, bus stems with fx sum to the mix" % (g, info["after"])


def t_robustness() -> str:
    pr = _project_with_kit()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p")
        pr.save(path)
        pr.autosave()
        time.sleep(1.05)
        pr.autosave()
        assert len(Project.autosaves(path)) >= 2
        with open(os.path.join(path, "project.json"), "w") as fh:
            fh.write("{broken")
        from .project import ProjectError

        try:
            Project.load(path)
            raise AssertionError("corrupt project loaded")
        except ProjectError:
            pass
        rec = Project.recover(path)
        assert len(rec.assets) == 1 and len(rec.pattern.events) == len(pr.pattern.events)
        try:
            Project.load(os.path.join(d, "missing"))
            raise AssertionError("missing project loaded")
        except ProjectError:
            pass
        rec.song_add(rec.current_pattern, 2)
        assert rec.song_seconds() > 5
    e = Engine(_tone_project(2))
    e.trigger(0, 1.0)
    e.render(0.1)
    dg = e.diagnostics()
    assert dg["voices"] == 1 and dg["sample_memory_mb"] > 0 and dg["device_state"] == "running"
    return "corrupt project reported cleanly, newest autosave recovered, song chain ok, diagnostics report"


def t_optional_backends() -> str:
    got = []
    if dsp.rubberband_available():
        dsp.set_stretch_backend("rubberband")
        try:
            x = dsp.ensure_2d(demo.tone(440.0, 2.0))
            y, z = dsp.time_stretch(x, 1.5), dsp.pitch_shift(x, 7.0)
            assert abs(len(y) / len(x) - 1.5) < 0.01 and len(z) == len(x)
            f = analysis.estimate_pitch(z, SR)[0]
            assert abs(f / (440 * 2 ** (7 / 12)) - 1) < 0.01, f
            got.append("rubberband")
        finally:
            dsp.set_stretch_backend("wsola")
    try:
        import aubio  # noqa: F401

        analysis.set_onset_backend("aubio")
        try:
            b = demo.drum_break(92, 2)
            pos, _ = analysis.detect_onsets(b, SR, 0.5)
            assert 14 <= len(pos) <= 26
            got.append("aubio")
        finally:
            analysis.set_onset_backend("builtin")
    except ImportError:
        pass
    try:
        import pedalboard  # noqa: F401

        f = fxmod.make_fx("pedalboard", SR, plugin="Reverb", params='{"room_size": 0.6}')
        y = f.process(demo.drum_break(92, 1)[:8192])
        assert y.shape == (8192, 2) and np.isfinite(y).all()
        got.append("pedalboard")
    except ImportError:
        pass
    if not got:
        raise Skip("none of rubberband, aubio, pedalboard installed")
    return "verified: " + ", ".join(got)


def t_soak() -> str:
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = soak(8.0, seed=3, verbose=True)
    assert rc == 0, buf.getvalue()
    return buf.getvalue().strip().replace("soak ", "")[:110]


CHECKS += [
    ("effects rack", t_fx_rack),
    ("buses and sidechain", t_buses_sidechain),
    ("voice filter, lfo, bend", t_voice_modulation),
    ("stems, loops, embeddings", t_stems_loops_embed),
    ("library and packs", t_library),
    ("midi, learn, clock", t_midi),
    ("plan providers and taste", t_providers_taste),
    ("loudness and bus stems", t_loudness_busstems),
    ("recovery and diagnostics", t_robustness),
    ("optional backends", t_optional_backends),
    ("soak", t_soak),
]


def run(verbose: bool = True) -> int:
    fails = skips = 0
    t_all = time.time()
    for name, fn in CHECKS:
        t0 = time.time()
        skipped = False
        try:
            detail = fn()
            ok = True
        except Skip as exc:
            detail, ok, skipped = str(exc), True, True
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            detail, ok = "%s: %s" % (exc.__class__.__name__, exc), False
        fails += 0 if ok else 1
        skips += 1 if skipped else 0
        if verbose:
            print("  %-4s %-30s %5.2fs  %s" % ("SKIP" if skipped else ("PASS" if ok else "FAIL"), name, time.time() - t0, detail))
    print("\n%d/%d checks passed (%d skipped) in %.1fs" % (len(CHECKS) - fails - skips, len(CHECKS), skips, time.time() - t_all))
    return 1 if fails else 0


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except (OSError, ValueError):
        return 0.0


def soak(seconds: float = 20.0, seed: int = 1, verbose: bool = True) -> int:
    """Stress the engine: random triggers, rolls, pitch bend, tempo and pattern changes, effects on every bus.
    Fails on NaN/inf, clipping past full scale, runaway voices, or memory growth."""
    rng = np.random.default_rng(seed)
    pr = _project_with_kit(seed=seed)
    pr.render_all()
    e = Engine(pr, block=256, seed=seed)
    from .fx import make_fx

    for i, p in enumerate(pr.banks[0].pads):
        p.bus = i % 4
        p.send_a, p.send_b = float(rng.random() < 0.4) * 0.2, float(rng.random() < 0.4) * 0.3
        if i % 5 == 0:
            p.lfo_target, p.lfo_depth, p.lfo_rate = "cutoff", 1.0, 2.0
            p.filter_kind, p.filter_cut = "lp", 3000.0
    e.buses[1].fx = [make_fx("compressor", SR, threshold_db=-30, ratio=6)]
    e.buses[1].sidechain = 0
    e.buses[2].fx = [make_fx("chorus", SR), make_fx("phaser", SR)]
    e.master_fx = [make_fx("eq", SR, high_db=-2), make_fx("tape", SR)]
    e.play()
    t_end = int(seconds * SR)
    names = list(pr.patterns)
    start_rss = _rss_mb()
    bad = []
    peak = 0.0
    maxv = 0
    t0 = time.time()
    next_event = 0
    while e.frame < t_end:
        if e.frame >= next_event:
            for _ in range(int(rng.integers(1, 4))):
                e.trigger(int(rng.integers(0, 16)), float(rng.uniform(0.2, 1.0)))
            r = rng.random()
            if r < 0.05:
                e.roll(int(rng.integers(0, 16)), 16, 6)
            elif r < 0.10:
                e.set_pitch_bend(float(rng.uniform(-2, 2)))
            elif r < 0.15:
                e.set_bpm(float(rng.uniform(70, 160)))
            elif r < 0.18:
                e.set_pattern(names[int(rng.integers(0, len(names)))])
            elif r < 0.20:
                e.stop_all()
            next_event = e.frame + int(rng.integers(500, 6000))
        y = e.render_block(int(rng.choice([64, 128, 256, 512])))
        if not np.isfinite(y).all():
            bad.append("non-finite output at frame %d" % e.frame)
            break
        peak = max(peak, float(np.abs(y).max()))
        maxv = max(maxv, e.voice_count)
    wall = time.time() - t0
    grew = _rss_mb() - start_rss
    if peak >= 1.0:
        bad.append("output reached full scale (%.4f)" % peak)
    if maxv > e.max_voices * 2 + 1:
        bad.append("voice count ran away (%d)" % maxv)
    if grew > 80:
        bad.append("memory grew %.0f MB" % grew)
    if len(e.live) > 200:
        bad.append("live trigger queue grew to %d" % len(e.live))
    if verbose:
        print("soak %.0fs of audio in %.1fs wall (%.1fx real time): peak %.3f, max voices %d, memory %+.1f MB -> %s" % (
            seconds, wall, seconds / max(wall, 1e-6), peak, maxv, grew, "FAIL: " + "; ".join(bad) if bad else "PASS"))
    return 1 if bad else 0
