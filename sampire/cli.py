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
        plan = creative.parse_instruction(args.prompt)
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
    y = export.render_pattern(pr, pr.current_pattern, 1)
    out = args.out or "sampire_render.wav"
    if os.path.isdir(out):
        out = os.path.join(out, "sampire_render.wav")
    export.export_audio(out, y, pr.sr, 24)
    print("rendered    : %s  (%.1f s)" % (out, len(y) / pr.sr))
    if args.stems:
        stems = export.export_stems(pr, os.path.splitext(out)[0] + "_stems")
        print("stems       : %d files in %s" % (len(stems), os.path.splitext(out)[0] + "_stems"))
    if args.save_project:
        print("project     :", pr.save(args.save_project))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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
