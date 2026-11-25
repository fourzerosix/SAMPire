"""Exports: pattern and song renders, per-pad stems, pad and slice wavs, MIDI file. Writes only complete files."""

from __future__ import annotations

import os
import struct
from typing import Dict, List, Optional

import numpy as np

from . import asset as assetmod
from . import dsp
from .config import PPQ
from .engine import Engine
from .project import Project
from .sequencer import Pattern


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s) or "x"


def pattern_seconds(pr: Project, pat: Pattern, bars: Optional[int] = None) -> float:
    return (bars or pat.bars) * pat.beats_per_bar * 60.0 / pr.bpm


def render_pattern(pr: Project, pattern: Optional[str] = None, loops: int = 1, tail_s: float = 2.0, seed: Optional[int] = None,
                   only_pads: Optional[List[int]] = None, block: int = 256, only_bus: Optional[int] = None,
                   lufs: Optional[float] = None, ceiling_db: float = -1.0) -> np.ndarray:
    """Offline render of a pattern (or the song when pr.mode == 'song'). Deterministic for a given seed."""
    pr.render_all()
    eng = Engine(pr, block=block, seed=pr.seed if seed is None else seed)
    if pattern:
        pr.current_pattern = pattern
    pat = pr.pattern
    eng.sync_from_project()
    if only_bus is not None:
        eng.only_bus, eng.stem_mode = only_bus, True
    if only_pads is not None:
        saved = pat.events
        pat.events = [e for e in pat.events if e.pad in only_pads]
        pat.touch()
    try:
        eng.play()
        if pr.mode == "song" and pr.song:
            secs = sum(pr.patterns[n].bars * pr.patterns[n].beats_per_bar * 60.0 / pr.bpm * max(r, 1) for n, r in pr.song if n in pr.patterns)
        else:
            secs = pattern_seconds(pr, pat) * loops
        y = eng.render(secs + tail_s)
        if lufs is not None:
            y, info = dsp.normalize_lufs(y, pr.sr, lufs, ceiling_db)
            render_pattern.last_info = info
        return y
    finally:
        if only_pads is not None:
            pat.events = saved
            pat.touch()


def export_audio(path: str, data: np.ndarray, sr: int, bits: int = 24) -> str:
    ext = os.path.splitext(path)[1].lower()
    sub = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bits, "PCM_24")
    if ext == ".flac" and bits == 32:
        sub = "PCM_24"
    assetmod.save_audio(path, data, sr, sub)
    return path


def export_stems(pr: Project, outdir: str, pattern: Optional[str] = None, tail_s: float = 2.0, bits: int = 24) -> List[str]:
    """One wav per pad that plays in the pattern. All stems start at time zero so they line up in any DAW."""
    os.makedirs(outdir, exist_ok=True)
    pat = pr.patterns[pattern] if pattern else pr.pattern
    files = []
    for gi in sorted({e.pad for e in pat.events}):
        pad = pr.pad(gi)
        y = render_pattern(pr, pat.name, 1, tail_s, only_pads=[gi])
        f = os.path.join(outdir, "%02d_%s.wav" % (gi + 1, _safe(pad.name or "pad")))
        export_audio(f, y, pr.sr, bits)
        files.append(f)
    return files


def export_pads(pr: Project, outdir: str, bits: int = 24) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    pr.render_all()
    files = []
    for pad in pr.all_pads():
        if pad.sample is not None:
            f = os.path.join(outdir, "%02d_%s.wav" % (pad.index + 1, _safe(pad.name or "pad")))
            export_audio(f, pad.sample, pr.sr, bits)
            files.append(f)
    return files


def export_slices(asset, slices, outdir: str, bits: int = 24) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    files = []
    for i, s in enumerate(slices):
        f = os.path.join(outdir, "%s_%03d.wav" % (_safe(asset.name), i + 1))
        export_audio(f, dsp.declick(asset.data[s.start:s.end], asset.sr), asset.sr, bits)
        files.append(f)
    return files


# ----------------------------------------------------------------------------- MIDI (own tiny SMF writer: no dependency)
def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def pattern_to_midi(pr: Project, pat: Pattern, path: str, channel: int = 9) -> str:
    """Format 0 standard MIDI file at 960 PPQ. Pad notes follow each pad's midi_note (default 36 + index)."""
    evs = []
    for e in pat.events:
        note = int(np.clip(pr.pad(e.pad).midi_note, 0, 127))
        length = e.length or pat.step_ticks // 2
        vel = int(np.clip(round(e.vel * 127), 1, 127))
        evs.append((e.tick, 1, 0x90 | channel, note, vel))
        evs.append((e.tick + length, 0, 0x80 | channel, note, 0))
    evs.sort()
    tempo = int(round(60_000_000 / pr.bpm))
    track = bytearray()
    track += _vlq(0) + b"\xff\x51\x03" + tempo.to_bytes(3, "big")
    track += _vlq(0) + b"\xff\x58\x04" + bytes([pat.beats_per_bar, 2, 24, 8])
    last = 0
    for t, _, st, n, v in evs:
        track += _vlq(t - last) + bytes([st, n, v])
        last = t
    track += _vlq(max(pat.length_ticks - last, 0)) + b"\xff\x2f\x00"
    data = b"MThd" + struct.pack(">IHHH", 6, 0, 1, PPQ) + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def export_bus_stems(pr: Project, outdir: str, pattern: Optional[str] = None, tail_s: float = 2.0) -> List[str]:
    """One float32 wav per mixer bus that plays in the pattern, with the bus effects, gain and the pads' sends included.
    Master effects and the limiter are bypassed so the stems sum back to the mix (at the master gain)."""
    os.makedirs(outdir, exist_ok=True)
    pat = pr.patterns[pattern] if pattern else pr.pattern
    used = sorted({int(pr.pad(e.pad).bus) for e in pat.events})
    files = []
    for k in used:
        y = render_pattern(pr, pat.name, 1, tail_s, only_bus=k)
        name = "bus%d" % (k + 1)
        custom = (pr.fx_settings.get("buses") or [{}] * 4)[k].get("name", name)
        f = os.path.join(outdir, "%s.wav" % (name if custom == name else "%s_%s" % (name, _safe(custom))))
        export_audio(f, y, pr.sr, 32)
        files.append(f)
    return files
