"""Command line entry point: interface, headless pipeline, analysis, chopping, rendering, self-test."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .config import SAMPLE_RATE, load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sampire", description="SAMPIRE: terminal sampler, chopper and drum machine with an autonomous creative engine.")
    p.add_argument("files", nargs="*", help="audio file(s) to load (first becomes the source)")
    p.add_argument("--version", action="version", version="sampire " + __version__)
    p.add_argument("--demo", action="store_true", help="load a built-in synthetic drum break (no files or audio device needed)")
    p.add_argument("--project", metavar="DIR", help="open this project folder (created on first save)")
    p.add_argument("--no-output", action="store_true", help="run silent: meters, sequencer and recording still work")
    p.add_argument("--device", help="output device (index or name fragment)")
    p.add_argument("--input", dest="input_device", help="input device for sampling external audio")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--block", type=int, help="audio block size in frames (default 256)")
    p.add_argument("--seed", type=int, help="seed for the creative engine (same seed, same result)")
    p.add_argument("--selftest", action="store_true", help="headless proof that the DSP, sequencer and engine are accurate")
    p.add_argument("--no-midi", action="store_true", help="do not open MIDI inputs at start")
    p.add_argument("--provider", help="plan provider for prompts: rules (default), command:PROGRAM ARGS, python:module:function")
    p.add_argument("--allow-python-provider", action="store_true", help="permit python: providers (they run your own code)")
    p.add_argument("--stretch-backend", choices=["wsola", "rubberband"], help="time stretch / pitch shift engine")
    p.add_argument("--onset-backend", choices=["builtin", "aubio"], help="onset detector")
    p.add_argument("--soak", type=float, metavar="SECONDS", help="stress test: random triggers, effects and tempo changes, checks stability")
    p.add_argument("--diagnostics", action="store_true", help="run the engine briefly offline and print a health report")
    p.add_argument("--midi-ports", action="store_true", help="list MIDI inputs and outputs and exit")
    p.add_argument("--report", action="store_true", help="print an environment report to paste into a bug report")
    g = p.add_argument_group("headless tools")
    g.add_argument("--analyze", metavar="FILE", help="print tempo, key, onsets and loudness for a file")
    g.add_argument("--chop", metavar="FILE", help="slice a file and write the slices to --out")
    g.add_argument("--mode", default="hybrid", choices=["transient", "hybrid", "grid", "equal", "silence"], help="slicing mode")
    g.add_argument("--slices", type=int, default=16, help="slice count for --mode equal, pad count for --render")
    g.add_argument("--sensitivity", type=float, default=0.5, help="transient sensitivity 0..1")
    g.add_argument("--render", metavar="FILE", help="autonomous pipeline: analyse, chop, build a kit, write patterns, render to --out (or --demo)")
    g.add_argument("--prompt", help="plain-English instruction for --render, e.g. 'chop into 12 pieces, reverse a few, two 4-bar patterns'")
    g.add_argument("--style", default="break", choices=["break", "four_floor", "sparse", "euclid", "chops"])
    g.add_argument("--bars", type=int, default=2)
    g.add_argument("--bpm", type=float)
    g.add_argument("--swing", type=float, default=0.0)
    g.add_argument("--experimental", type=float, default=0.3, help="0..1 how many chops get transformed")
    g.add_argument("--out", metavar="PATH", help="output file or directory")
    g.add_argument("--stems", action="store_true", help="with --render: also write one wav per pad next to the output")
    g.add_argument("--save-project", metavar="DIR", help="with --render: also save the result as a project folder")
    g.add_argument("--lufs", type=float, metavar="LUFS", help="with --render: normalise the bounce to this integrated loudness (true peak ceiling -1 dB)")
    g.add_argument("--bus-stems", action="store_true", help="with --render: also write one stem per mixer bus (effects included)")
    g.add_argument("--split", metavar="FILE", help="separate a file into drums/bass/other stems")
    g.add_argument("--method", default="dsp", choices=["dsp", "demucs"], help="stem separation method (demucs needs pip install demucs)")
    g.add_argument("--find-loops", metavar="FILE", help="find loop regions; with --out writes the best one as a seamless wav")
    g.add_argument("--recover", metavar="DIR", help="list the autosaves of a project folder")
    lb = p.add_argument_group("library")
    lb.add_argument("--lib-scan", nargs="+", metavar="DIR", help="index folders of samples into the library")
    lb.add_argument("--find", metavar="QUERY", help="search the library, e.g. 'kick bpm:85-95 key:C tag:drums rating:3 fav'")
    lb.add_argument("--similar", metavar="FILE", help="find library samples that sound like this file")
    lb.add_argument("--pack", metavar="PATH", help="import a sample pack (folder or zip) into the library")
    return p


def _asset_from_args(args):
    from . import asset as assetmod
    from . import demo

    if args.files:
        return assetmod.load_audio(args.files[0])
    if args.demo:
        return assetmod.asset_from_array(demo.drum_break(92.0, 2, with_melody=True), "demo-break", origin="demo")
    return None


def cmd_analyze(path: str) -> int:
    from . import analysis, asset as assetmod

    a = assetmod.load_audio(path)
    r = analysis.analyze(a.data, a.sr)
    print("file        : %s" % path)
    print("duration    : %.2f s  (%d frames @ %d Hz)" % (r["duration"], r["frames"], r["sr"]))
    print("peak / rms  : %.1f dBFS / %.1f dBFS    loudness %.1f LUFS (ungated)" % (r["peak_db"], r["rms_db"], r["lufs"]))
    print("tempo       : %.1f BPM  (confidence %.2f)" % (r["bpm"], r["bpm_confidence"]))
    print("key         : %s  (%.2f)" % (r["key"], r["key_confidence"]))
    print("percussive  : %.0f %%" % (100 * r["percussive_ratio"]))
    print("onsets      : %d" % len(r["onsets"]))
    return 0


def cmd_chop(args) -> int:
    from . import analysis, asset as assetmod, export, slicer

    a = assetmod.load_audio(args.chop)
    bpm = analysis.estimate_tempo(a.data, a.sr)["bpm"] if args.mode in ("grid", "hybrid") else None
    sl = slicer.make_slices(args.mode, a.data, a.sr, count=args.slices, sensitivity=args.sensitivity, bpm=bpm)
    out = args.out or (os.path.splitext(args.chop)[0] + "_slices")
    files = export.export_slices(a, sl, out)
    print("%d slices (%s) written to %s" % (len(files), args.mode, out))
    return 0


def cmd_render(args) -> int:
    from . import creative, export, project

    a = _asset_from_args(args) if (args.files or args.demo) else None
    if a is None and args.render and args.render != "-":
        from . import asset as assetmod

        a = assetmod.load_audio(args.render)
    if a is None:
        print("--render needs an input file or --demo", file=sys.stderr)
        return 2
    seed = args.seed if args.seed is not None else 1
    pr = project.Project(os.path.splitext(os.path.basename(a.name))[0])
    pr.seed = seed
    pr.add_asset(a)
    if args.prompt:
        from . import providers

        plan = providers.plan_from_prompt(args.prompt, args.provider or "rules", {"bpm": a.analysis.get("bpm"), "duration": a.duration},
                                          bool(args.allow_python_provider))
        for n in plan["notes"]:
            print("note:", n)
        plan.setdefault("kit", {})
        if args.bpm:
            plan["bpm"] = args.bpm
        if plan["pattern"] is None:
            plan["pattern"] = {"count": 1, "bars": args.bars, "style": args.style, "density": 0.5}
        rep, pats = creative.run_plan(pr, a, plan, seed)
    else:
        from . import analysis

        a.analysis = analysis.analyze(a.data, a.sr)
        pr.bpm = args.bpm or (a.analysis["bpm"] or 92.0)
        pr.swing = args.swing
        rep = creative.generate_kit(pr, a, creative.KitParams(pads=min(args.slices, 16), mode=args.mode, sensitivity=args.sensitivity,
                                                              experimental=args.experimental, seed=seed))
        pt = creative.generate_pattern(pr, 0, args.bars, args.style, 0.5, 0.7, args.swing, seed)
        pr.patterns[pt.name] = pt
        pr.current_pattern = pt.name
        pats = [pt]
    print("source      : %s  (%.2f s, %.1f BPM, %s)" % (a.name, a.duration, a.analysis.get("bpm", 0), a.analysis.get("key", "?")))
    print("kit         : %d pads from %d slices, %d transformed  %s" % (rep.chosen, rep.slices, rep.transformed, rep.labels))
    for pt in pats:
        print("pattern     : %s  %d bars, %d events @ %.1f BPM" % (pt.name, pt.bars, len(pt.events), pr.bpm))
    y = export.render_pattern(pr, pr.current_pattern, 1, lufs=args.lufs)
    if args.lufs is not None:
        info = getattr(export.render_pattern, "last_info", {})
        print("loudness    : %.1f -> %.1f LUFS (gain %+.1f dB%s, true peak %.1f dB)" % (info.get("before", 0), info.get("after", 0), info.get("gain_db", 0),
                                                                               ", peak limited" if info.get("limited") else "", info.get("true_peak_db", 0)))
    out = args.out or "sampire_render.wav"
    if os.path.isdir(out):
        out = os.path.join(out, "sampire_render.wav")
    export.export_audio(out, y, pr.sr, 24)
    print("rendered    : %s  (%.1f s)" % (out, len(y) / pr.sr))
    if args.stems:
        stems = export.export_stems(pr, os.path.splitext(out)[0] + "_stems")
        print("stems       : %d files in %s" % (len(stems), os.path.splitext(out)[0] + "_stems"))
    if args.bus_stems:
        bs = export.export_bus_stems(pr, os.path.splitext(out)[0] + "_busstems")
        print("bus stems   : %d files in %s" % (len(bs), os.path.splitext(out)[0] + "_busstems"))
    if args.save_project:
        print("project     :", pr.save(args.save_project))
    return 0


def cmd_library(args) -> int:
    from . import asset as assetmod, library

    lib = library.Library()
    if args.lib_scan:
        added, skipped = lib.scan(args.lib_scan, lambda m: print(m, end="\r"))
        print("indexed %d new files, %d unchanged; library now holds %d samples (%s)" % (added, skipped, lib.count(), lib.path))
    if args.pack:
        dest, n = lib.import_pack(args.pack)
        print("imported %d samples from %s into %s" % (n, args.pack, dest))
    rows = []
    if args.find is not None:
        rows = lib.find(args.find, 60)
    if args.similar:
        a = assetmod.load_audio(args.similar)
        rows = lib.similar((a.data, a.sr), 15)
    for r in rows:
        print("%-32s %5.1fs %5.0f bpm %-9s %-5s %s%s" % (r.name[:32], r.dur, r.bpm, r.key[:9], "*" * r.rating,
                                                       ",".join(r.tags)[:36], ("  %.2f" % r.score) if r.score else ""))
    if rows:
        print("%d result(s)" % len(rows))
    return 0


def cmd_split(args) -> int:
    from . import asset as assetmod, export, stems

    a = assetmod.load_audio(args.split)
    parts = stems.separate(a.data, a.sr, args.method, args.split)
    out = args.out or (os.path.splitext(args.split)[0] + "_stems")
    for name, data in parts.items():
        f = os.path.join(out, "%s_%s.wav" % (a.name, name))
        export.export_audio(f, data, a.sr, 24)
        print("wrote", f)
    return 0


def cmd_find_loops(args) -> int:
    from . import asset as assetmod, export, loops

    a = assetmod.load_audio(args.find_loops)
    c = loops.find_loops(a.data, a.sr)
    if not c:
        print("no loop found (needs a steady tempo)")
        return 1
    for i, x in enumerate(c):
        print("%d) start %.3fs  length %.3fs  %g bar(s)  score %.2f" % (i + 1, x.start / a.sr, x.length / a.sr, x.bars, x.score))
    if args.out:
        y = loops.make_seamless(a.data[c[0].start:c[0].end], 20.0, a.sr)
        export.export_audio(args.out, y, a.sr, 24)
        print("wrote seamless loop", args.out)
    return 0


def cmd_report() -> int:
    import platform

    import numpy
    import scipy

    from . import __version__, analysis, dsp
    from .backend import list_devices

    print("SAMPire     :", __version__)
    print("platform    :", platform.platform())
    print("python      :", platform.python_version())
    print("numpy/scipy :", numpy.__version__, "/", scipy.__version__)
    try:
        import soundfile

        print("soundfile   :", soundfile.__version__, "| libsndfile", soundfile.__libsndfile_version__)
    except Exception as exc:
        print("soundfile   : unavailable (%s)" % exc)
    try:
        import sounddevice

        print("sounddevice :", sounddevice.__version__, "|", sounddevice.get_portaudio_version()[1])
    except Exception as exc:
        print("sounddevice : unavailable (%s)" % str(exc).splitlines()[0])
    for mod in ("mido", "rtmidi", "pedalboard", "aubio", "soxr", "demucs"):
        try:
            m = __import__(mod)
            print("%-12s: %s" % (mod, getattr(m, "__version__", "installed")))
        except Exception:
            print("%-12s: not installed" % mod)
    print("rubberband  :", "found" if dsp.rubberband_available() else "not found")
    print("audio devices:")
    for row in list_devices()[:12]:
        print("   ", row)
    from .midi import list_ports

    pr = list_ports()
    print("midi in     :", ", ".join(pr["inputs"]) or "none")
    print("midi out    :", ", ".join(pr["outputs"]) or "none")
    return 0


def cmd_diagnostics() -> int:
    import json

    from . import demo, asset as assetmod, creative, project
    from .engine import Engine

    pr = project.Project("diag")
    a = pr.add_asset(assetmod.asset_from_array(demo.drum_break(92, 2), "demo"))
    creative.generate_kit(pr, a)
    pt = creative.generate_pattern(pr, 0, 2)
    pr.patterns[pt.name] = pt
    pr.current_pattern = pt.name
    e = Engine(pr)
    e.play()
    e.render(3.0)
    d = e.diagnostics()
    print(json.dumps(d, indent=1))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stretch_backend:
        from . import dsp

        dsp.set_stretch_backend(args.stretch_backend)
    if args.onset_backend:
        from . import analysis

        analysis.set_onset_backend(args.onset_backend)
    if args.report:
        return cmd_report()
    if args.midi_ports:
        from .midi import list_ports

        pr = list_ports()
        print("inputs :", ", ".join(pr["inputs"]) or "none")
        print("outputs:", ", ".join(pr["outputs"]) or "none")
        return 0
    if args.soak:
        from .selftest import soak

        return soak(args.soak)
    if args.diagnostics:
        return cmd_diagnostics()
    if args.recover:
        from .project import Project

        for f in Project.autosaves(args.recover):
            print(f)
        return 0
    if args.lib_scan or args.find is not None or args.similar or args.pack:
        return cmd_library(args)
    if args.split:
        return cmd_split(args)
    if args.find_loops:
        return cmd_find_loops(args)
    if args.selftest:
        from .selftest import run

        return run()
    if args.list_devices:
        from .backend import list_devices

        print("\n".join(list_devices()))
        return 0
    if args.analyze:
        return cmd_analyze(args.analyze)
    if args.chop:
        return cmd_chop(args)
    if args.render or (args.out and (args.demo or args.files) and args.prompt):
        return cmd_render(args)
    from .ui.tui import run_tui

    cfg = load_config()
    return run_tui(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
