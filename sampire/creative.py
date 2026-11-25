"""The Creative Engine: autonomous selection, editing, pad assignment and pattern generation.

Level 1 (algorithmic, deterministic for a seed). It only ever produces project edits (pad assignments, recipes,
patterns); it never touches the audio callback. Every result can be accepted, edited or regenerated."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import analysis, dsp
from .asset import AudioAsset
from .config import PPQ, PADS_PER_BANK
from .pad import Pad
from .project import Project
from .recipe import Recipe
from .sequencer import Pattern
from .slicer import Slice, make_slices

KIT_ORDER = ["kick", "snare", "hat", "open_hat", "tom", "perc", "bass", "tone"]


@dataclass
class KitParams:
    pads: int = 16
    mode: str = "hybrid"                 # transient | hybrid | grid | equal | silence
    sensitivity: float = 0.5
    experimental: float = 0.3            # 0 = leave every chop alone, 1 = transform nearly all of them
    keep_original: float = 0.5           # fraction of pads guaranteed untouched (recognisable source)
    prefer: str = "any"                  # any | percussive | melodic
    length: str = "any"                  # any | short | long
    bank: int = 0
    seed: int = 1
    tail_ms: Optional[float] = 600.0     # a chop ends at the next hit or after this long
    ops: Optional[List[str]] = None      # restrict the transform vocabulary (None = all sensible ones)
    taste: bool = False                  # bias choices toward what you kept before (see taste.py)


@dataclass
class KitReport:
    asset: str
    slices: int
    chosen: int
    labels: Dict[str, int] = field(default_factory=dict)
    transformed: int = 0
    notes: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- selection
def _score(sl: Slice, feat: dict, p: KitParams) -> float:
    s = 0.5 + sl.strength
    if feat["peak"] < 0.02:
        return -1.0                                      # silence / noise floor
    dur = feat["dur"]
    if p.length == "short":
        s += 0.6 if dur < 0.25 else -0.3
    elif p.length == "long":
        s += 0.6 if dur > 0.4 else -0.3
    perc = feat["decay"] < 0.35 and feat["clarity"] < 0.8
    if p.prefer == "percussive":
        s += 0.6 if perc else -0.4
    elif p.prefer == "melodic":
        s += 0.6 if feat["clarity"] > 0.7 else -0.4
    return s


def _vector(feat: dict) -> np.ndarray:
    return np.array([np.log10(feat["centroid"] + 50) / 4.3, feat["low"], feat["high"], min(feat["decay"], 1.0), feat["flat"] * 2, min(feat["dur"], 1.0)])


def _slice_feats(label: str, feat: dict) -> Dict[str, float]:
    return {"label:" + (label or "?"): 1.0, "len:" + ("short" if feat["dur"] < 0.25 else "long"): 1.0,
            "kind:" + ("percussive" if feat["decay"] < 0.35 and feat["clarity"] < 0.8 else "tonal"): 1.0}


def select_slices(asset: AudioAsset, slices: List[Slice], p: KitParams, rng: np.random.Generator, tm=None) -> List[Tuple[Slice, dict]]:
    """Pick N slices that are loud enough, suit the preferences, and differ from one another (farthest-point sampling)."""
    cand = []
    for sl in slices:
        x = asset.data[sl.start:sl.end]
        if len(x) < 64:
            continue
        feat = analysis.describe_hit(x[: int(0.4 * asset.sr)], asset.sr)
        sc = _score(sl, feat, p)
        if sc > 0:
            sl.label = analysis.classify_sample(x[: int(0.5 * asset.sr)], asset.sr)
            if tm is not None and tm.n > 0:
                sc += 0.5 * tm.score(_slice_feats(sl.label, feat))
            cand.append((sc, sl, feat))
    if not cand:
        return []
    cand.sort(key=lambda t: -t[0])
    chosen = [cand[0]]
    vecs = [_vector(cand[0][2])]
    rest = cand[1:]
    while rest and len(chosen) < p.pads:
        best_i, best_v = 0, -1.0
        for i, (sc, sl, feat) in enumerate(rest):
            d = min(float(np.linalg.norm(_vector(feat) - v)) for v in vecs)
            val = d * (0.6 + 0.4 * sc) + rng.random() * 0.02
            if val > best_v:
                best_i, best_v = i, val
        pick = rest.pop(best_i)
        chosen.append(pick)
        vecs.append(_vector(pick[2]))
    if len(chosen) < p.pads and cand:                    # fewer slices than pads: reuse the strongest
        k = 0
        while len(chosen) < p.pads:
            chosen.append(cand[k % len(cand)])
            k += 1
    return [(sl, feat) for _, sl, feat in chosen]


def order_for_drum_layout(items: List[Tuple[Slice, dict]]) -> List[Tuple[Slice, dict]]:
    """Kick, snare, hats first (the pads you play with your thumbs), everything else after, strongest first."""
    rank = {k: i for i, k in enumerate(KIT_ORDER)}
    return sorted(items, key=lambda it: (rank.get(it[0].label, 99), -it[0].strength))


# ----------------------------------------------------------------------------- transformations
_TRANSFORMS = {
    "reverse": lambda r, rng: r.add("reverse"),
    "pitch_down": lambda r, rng: r.add("pitch", semitones=float(-rng.integers(2, 8))),
    "pitch_up": lambda r, rng: r.add("pitch", semitones=float(rng.integers(2, 8))),
    "stretch": lambda r, rng: r.add("stretch", ratio=float(rng.choice([0.75, 1.25, 1.5, 2.0]))),
    "stutter": lambda r, rng: r.add("stutter", ms=float(rng.choice([30, 45, 60, 90])), repeats=int(rng.integers(3, 7))),
    "crush": lambda r, rng: r.add("bitcrush", bits=int(rng.integers(6, 11)), downsample=int(rng.choice([1, 2, 3]))),
    "drive": lambda r, rng: r.add("drive", amount=float(rng.uniform(0.3, 0.8))),
    "lowpass": lambda r, rng: r.add("filter", kind="lp", cutoff=float(rng.choice([800, 1500, 3000]))),
    "highpass": lambda r, rng: r.add("filter", kind="hp", cutoff=float(rng.choice([300, 800, 1500]))),
    "gate": lambda r, rng: r.add("gate", steps=str(rng.choice(["1011", "1101", "10101101", "1110"])), rate_hz=float(rng.choice([8, 12, 16]))),
    "tape_stop": lambda r, rng: r.add("tape_stop", ms=float(rng.choice([250, 400, 600]))),
    "granular": lambda r, rng: r.add("granular", grain_ms=float(rng.choice([25, 40, 70])), jitter=float(rng.uniform(0.2, 0.8))),
    "shuffle": lambda r, rng: r.add("shuffle", parts=int(rng.choice([3, 4, 6]))),
    "crackle": lambda r, rng: r.add("crackle", level=0.03),
}
_SAFE_FOR_DRUMS = ["reverse", "pitch_down", "pitch_up", "crush", "drive", "lowpass", "highpass", "stutter", "tape_stop"]
_SAFE_FOR_LONG = list(_TRANSFORMS)


def random_recipe(rng: np.random.Generator, label: str, vocabulary: Optional[List[str]] = None, depth: int = 1, tm=None) -> Recipe:
    vocab = vocabulary or (_SAFE_FOR_DRUMS if label in ("kick", "snare", "hat", "open_hat", "perc") else _SAFE_FOR_LONG)
    r = Recipe(seed=int(rng.integers(0, 2 ** 31 - 1)))
    prob = tm.preference("op:", list(vocab)) if (tm is not None and tm.n > 0) else None
    for name in rng.choice(vocab, size=min(depth, len(vocab)), replace=False, p=prob):
        _TRANSFORMS[str(name)](r, rng)
    return r


# ----------------------------------------------------------------------------- kit generation
def kit_features(project: Project, bank: int = 0) -> List[Dict[str, float]]:
    """Feature dicts for every filled pad in a bank, in the form Taste learns from."""
    out = []
    for pad in project.banks[bank % len(project.banks)].pads:
        if pad.empty:
            continue
        f = {"label:" + (pad.label or "?"): 1.0}
        a = project.assets.get(pad.asset_id)
        if a is not None:
            f.update(_slice_feats(pad.label, analysis.describe_hit(a.region(pad.start, pad.end)[: 24000], a.sr)))
        for o in pad.recipe.ops:
            f["op:" + str(o["op"])] = 1.0
        if pad.reverse:
            f["op:reverse"] = 1.0
        out.append(f)
    return out


def pattern_features(style: str, density: float) -> Dict[str, float]:
    return {"style:" + style: 1.0, "density:" + ("busy" if density > 0.6 else "sparse" if density < 0.35 else "mid"): 1.0}


def generate_kit(project: Project, asset: AudioAsset, params: Optional[KitParams] = None, slices: Optional[List[Slice]] = None,
                 taste_model=None) -> KitReport:
    """Analyse, chop, choose, transform and assign: one asset in, one populated bank out."""
    p = params or KitParams()
    rng = np.random.default_rng(p.seed)
    if not asset.analysis:
        asset.analysis = analysis.analyze(asset.data, asset.sr)
    bpm = asset.analysis.get("bpm") if asset.analysis.get("bpm_confidence", 0) > 0.25 else None
    if slices is None:
        slices = make_slices(p.mode, asset.data, asset.sr, count=p.pads, sensitivity=p.sensitivity, bpm=bpm, tail_ms=p.tail_ms) \
            if p.mode in ("transient", "hybrid") else make_slices(p.mode, asset.data, asset.sr, count=p.pads, bpm=bpm)
    rep = KitReport(asset.name, len(slices), 0)
    tm = taste_model if p.taste else None
    picks = select_slices(asset, slices, p, rng, tm)
    if not picks:
        rep.notes.append("nothing usable found; try a higher sensitivity")
        return rep
    picks = order_for_drum_layout(picks)[: p.pads]
    bank = project.banks[p.bank % len(project.banks)]
    n_keep = int(round(p.keep_original * len(picks)))
    keep_ids = set(rng.choice(len(picks), size=min(n_keep, len(picks)), replace=False).tolist()) if n_keep else set()
    for i, (sl, feat) in enumerate(picks):
        pad = bank.pads[i % PADS_PER_BANK]
        pad.clear()
        pad.asset_id, pad.start, pad.end = asset.id, sl.start, sl.end
        pad.label = sl.label
        pad.name = "%s%d" % ((sl.label or "chop")[:5], i + 1)
        if sl.label in ("kick", "snare", "tom", "perc", "bass", "tone"):
            pad.mode = "one_shot"
        if sl.label == "hat":
            pad.choke_group = 1
        if sl.label == "open_hat":
            pad.choke_group = 1
            pad.release = 0.05
        if sl.label == "tone":
            pad.attack = 0.002
        if i not in keep_ids and rng.random() < p.experimental * 1.5 and p.experimental > 0:
            pad.recipe = random_recipe(rng, sl.label, p.ops, depth=1 + int(rng.random() < p.experimental * 0.5), tm=tm)
            rep.transformed += 1
        rep.labels[sl.label or "?"] = rep.labels.get(sl.label or "?", 0) + 1
    for i in range(len(picks), PADS_PER_BANK):
        bank.pads[i].clear()
    rep.chosen = len(picks)
    project.render_all()
    project.dirty = True
    return rep


# ----------------------------------------------------------------------------- pattern generation
STYLES = ("break", "four_floor", "sparse", "euclid", "chops")


def _pads_by_label(project: Project, bank: int) -> Dict[str, List[Pad]]:
    out: Dict[str, List[Pad]] = {}
    for pad in project.banks[bank % len(project.banks)].pads:
        if not pad.empty:
            out.setdefault(pad.label or "perc", []).append(pad)
    return out


def generate_pattern(project: Project, bank: int = 0, bars: int = 2, style: str = "break", density: float = 0.5,
                     consistency: float = 0.7, swing: float = 0.0, seed: int = 1, name: Optional[str] = None, taste_model=None) -> Pattern:
    """Build a pattern from the pads in a bank. `consistency` is how much bar 1 repeats in later bars."""
    rng = np.random.default_rng(seed)
    if style == "auto":
        prob = taste_model.preference("style:", list(STYLES)) if (taste_model is not None and taste_model.n > 0) else None
        style = str(rng.choice(list(STYLES), p=prob))
    pat = Pattern(name or "G%d" % (len(project.patterns) + 1), bars)
    pat.swing = swing
    pat.style = style
    lab = _pads_by_label(project, bank)
    allp = [p for ps in lab.values() for p in ps]
    if not allp:
        return pat
    pick = lambda k: lab[k][int(rng.integers(0, len(lab[k])))] if k in lab else None
    kick, snare, hat, ohat = pick("kick"), pick("snare"), pick("hat"), pick("open_hat")
    others = [p for k, ps in lab.items() if k not in ("kick", "snare", "hat", "open_hat") for p in ps] or allp
    spb = 16
    base: List[Tuple[int, int, float]] = []              # (step in bar, pad index, vel)

    def bar_pattern() -> List[Tuple[int, int, float]]:
        ev = []
        if style == "four_floor":
            for s in (0, 4, 8, 12):
                if kick:
                    ev.append((s, kick.index, 1.0))
            for s in (4, 12):
                if snare:
                    ev.append((s, snare.index, 0.9))
            for s in range(2, 16, 4):
                if hat:
                    ev.append((s, hat.index, 0.7))
        elif style == "sparse":
            if kick:
                ev.append((0, kick.index, 1.0))
                if rng.random() < density:
                    ev.append((int(rng.choice([7, 10, 11])), kick.index, 0.8))
            if snare:
                ev.append((int(rng.choice([4, 12])), snare.index, 0.9))
            for _ in range(int(2 + 4 * density)):
                ev.append((int(rng.integers(0, 16)), others[int(rng.integers(0, len(others)))].index, float(rng.uniform(0.4, 0.8))))
        elif style == "euclid":
            for p in allp[:6]:
                hits = int(np.clip(round(2 + density * 6 + rng.integers(-1, 2)), 1, 12))
                rot = int(rng.integers(0, 16))
                for i in range(16):
                    if (i * hits) % 16 < hits:
                        ev.append(((i + rot) % 16, p.index, float(rng.uniform(0.5, 1.0))))
        elif style == "chops":
            order = [p for p in allp]
            rng.shuffle(order)
            for i in range(16):
                if rng.random() < 0.35 + 0.6 * density:
                    ev.append((i, order[i % len(order)].index, float(rng.uniform(0.5, 1.0))))
        else:                                             # break
            if kick:
                ev.append((0, kick.index, 1.0))
                for s in (7, 10, 11, 14, 3, 6):
                    if rng.random() < 0.15 + 0.5 * density:
                        ev.append((s, kick.index, float(rng.uniform(0.6, 0.9))))
            if snare:
                ev.append((4, snare.index, 0.95))
                ev.append((12, snare.index, 0.95))
                if rng.random() < density:
                    ev.append((int(rng.choice([7, 9, 15])), snare.index, 0.35))     # ghost note
            if hat:
                for s in range(0, 16, 2):
                    ev.append((s, hat.index, 0.8 if s % 4 == 0 else 0.55))
                if density > 0.6:
                    for s in range(1, 16, 2):
                        if rng.random() < density - 0.4:
                            ev.append((s, hat.index, 0.3))
            if ohat and rng.random() < 0.6:
                ev.append((int(rng.choice([6, 14])), ohat.index, 0.7))
            for _ in range(int(density * 3)):
                ev.append((int(rng.integers(0, 16)), others[int(rng.integers(0, len(others)))].index, float(rng.uniform(0.4, 0.8))))
        return ev

    first = bar_pattern()
    for b in range(bars):
        ev = first if (b == 0 or rng.random() < consistency) else bar_pattern()
        for s, pad_i, v in ev:
            pat.add((b * spb + s) * pat.step_ticks, pad_i, v)
    pat.touch()
    return pat


def vary_pattern(pat: Pattern, amount: float = 0.3, seed: int = 1, project: Optional[Project] = None) -> Pattern:
    """Return a mutated copy: drops, shifts, ghost notes, ratchets and probability, scaled by `amount`."""
    rng = np.random.default_rng(seed)
    out = pat.copy(pat.name + "'")
    pads = sorted({e.pad for e in out.events})
    keep = []
    for e in out.events:
        r = rng.random()
        if r < 0.10 * amount and e.tick % (PPQ) != 0:
            continue                                        # drop off-beat hits, keep the backbone
        if r < 0.25 * amount:
            e.tick = (e.tick + int(rng.choice([-1, 1])) * out.step_ticks) % out.length_ticks
        elif r < 0.35 * amount:
            e.ratchet = int(rng.choice([2, 3]))
        elif r < 0.45 * amount:
            e.prob = float(rng.uniform(0.5, 0.9))
        keep.append(e)
    out.events = keep
    if pads:
        for _ in range(int(round(4 * amount))):
            out.add(int(rng.integers(0, out.steps)) * out.step_ticks, int(rng.choice(pads)), float(rng.uniform(0.25, 0.5)))
    out.touch()
    return out


def rearrange_chops(project: Project, bank: int = 0, bars: int = 1, strategy: str = "shuffle", seed: int = 1, steps_per_bar: int = 16) -> Pattern:
    """Play the bank's chops one after another (the classic chop-flip), in a new order."""
    rng = np.random.default_rng(seed)
    pads = [p for p in project.banks[bank % len(project.banks)].pads if not p.empty]
    pat = Pattern("F%d" % (len(project.patterns) + 1), bars)
    if not pads:
        return pat
    n = len(pads)
    order = list(range(n))
    if strategy == "shuffle":
        rng.shuffle(order)
    elif strategy == "reverse":
        order.reverse()
    elif strategy == "every_other":
        order = order[::2] + order[1::2]
    seq = []
    total = steps_per_bar * bars
    i = 0
    while len(seq) < total:
        k = order[i % n]
        rep = 2 if (strategy == "stutter" and rng.random() < 0.4) else 1
        seq.extend([k] * rep)
        i += 1
    for s, k in enumerate(seq[:total]):
        pat.add(s * pat.step_ticks, pads[k].index, 0.9)
    pat.touch()
    return pat


# ----------------------------------------------------------------------------- instructions (Level 4, offline and rule based)
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "sixteen": 16, "twelve": 12}


def _num(text: str, pat: str, default: Optional[int] = None) -> Optional[int]:
    m = re.search(pat, text)
    if not m:
        return default
    v = m.group(1)
    return int(v) if v.isdigit() else _NUM.get(v, default)


def parse_instruction(text: str) -> dict:
    """Turn a plain-English request into a plan (a dict of kit and pattern parameters).

    Rule based and fully offline. The plan schema is stable, so a language model can be plugged in later by
    returning the same dict from `Plan Provider`; SAMPire itself never sends your audio or text anywhere."""
    t = text.lower()
    plan: dict = {"kit": {}, "pattern": None, "notes": []}
    k = plan["kit"]
    n = _num(t, r"(\d+|one|two|three|four|five|six|eight|twelve|sixteen)\s*(?:pieces|chops|slices|pads|parts)")
    if n:
        k["pads"] = int(np.clip(n, 2, 16))
    ops: List[str] = []
    for word, op in (("reverse", "reverse"), ("backwards", "reverse"), ("pitch down", "pitch_down"), ("lower", "pitch_down"),
                     ("pitch up", "pitch_up"), ("higher", "pitch_up"), ("stretch", "stretch"), ("stutter", "stutter"),
                     ("glitch", "stutter"), ("lo-fi", "crush"), ("lofi", "crush"), ("crush", "crush"), ("distort", "drive"),
                     ("dirty", "drive"), ("vinyl", "crackle"), ("tape", "tape_stop"), ("granular", "granular"),
                     ("scatter", "granular"), ("gate", "gate"), ("muffled", "lowpass"), ("dark", "lowpass"), ("bright", "highpass")):
        if word in t and op not in ops:
            ops.append(op)
    if re.search(r"pitch\b[^.,;]*\bdown", t) and "pitch_down" not in ops:
        ops.append("pitch_down")
    if re.search(r"pitch\b[^.,;]*\bup", t) and "pitch_up" not in ops:
        ops.append("pitch_up")
    if ops:
        k["ops"] = ops
        k["experimental"] = 0.7
    if any(w in t for w in ("experimental", "weird", "wild", "crazy", "unpredictable")):
        k["experimental"] = 0.9
    if any(w in t for w in ("subtle", "gentle", "clean", "recognizable", "recognisable")):
        k["experimental"] = 0.15
        k["keep_original"] = 0.8
    if "percussive" in t or "drums" in t:
        k["prefer"] = "percussive"
    if "melodic" in t or "melody" in t:
        k["prefer"] = "melodic"
    if "short" in t:
        k["length"] = "short"
    if "long" in t:
        k["length"] = "long"
    if "interesting" in t or "transient" in t:
        k["mode"] = "transient"
    pat = _num(t, r"(\d+|one|two|three|four|five|six)\s*(?:different\s*)?(?:(?:\d+|one|two|four|eight)[- ]bars?\s*)?patterns?")
    bars = _num(t, r"(\d+|one|two|four|eight)[- ]bars?", 2)
    if pat or "pattern" in t or "beat" in t or "rhythm" in t or "sequence" in t:
        style = "break"
        if "four on the floor" in t or "four-on-the-floor" in t or "house" in t:
            style = "four_floor"
        elif "sparse" in t or "minimal" in t:
            style = "sparse"
        elif "euclid" in t or "polyrhythm" in t:
            style = "euclid"
        elif "chop" in t and "rearrange" in t or "flip" in t:
            style = "chops"
        plan["pattern"] = {"count": int(pat or 1), "bars": int(bars or 2), "style": style,
                           "density": 0.7 if any(w in t for w in ("busy", "dense")) else (0.25 if "sparse" in t else 0.5)}
    b = _num(t, r"(\d{2,3})\s*bpm")
    if b:
        plan["bpm"] = float(b)
    sw = re.search(r"swing(?:ing)?\s*(?:of\s*)?(\d+)", t)
    if sw:
        plan["swing"] = float(np.clip(int(sw.group(1)) / 100.0, 0.0, 0.5))
    elif "swing" in t or "swung" in t or "shuffle" in t:
        plan["swing"] = 0.2
    if not k and not plan["pattern"] and "bpm" not in plan:
        plan["notes"].append("no recognised instructions; try: 'chop it into 16 pieces, reverse a few, make two 4-bar patterns at 92 bpm'")
    return plan


def assign_slices(project: Project, asset: AudioAsset, slices: List[Slice], bank: int = 0) -> KitReport:
    """Put slices on pads in time order (classic MPC chop layout): slice 1 on pad 1, slice 2 on pad 2, and so on."""
    b = project.banks[bank % len(project.banks)]
    rep = KitReport(asset.name, len(slices), 0)
    for i, pad in enumerate(b.pads):
        if i < len(slices):
            sl = slices[i]
            pad.clear()
            pad.asset_id, pad.start, pad.end = asset.id, int(sl.start), int(sl.end)
            pad.label = analysis.classify_sample(asset.data[sl.start:sl.end][: asset.sr // 2], asset.sr)
            pad.name = "ch%d" % (i + 1)
            rep.labels[pad.label] = rep.labels.get(pad.label, 0) + 1
            rep.chosen += 1
        else:
            pad.clear()
    project.render_all()
    project.dirty = True
    return rep


def run_plan(project: Project, asset: AudioAsset, plan: dict, seed: int = 1, bank: int = 0, taste_model=None) -> Tuple[KitReport, List[Pattern]]:
    kp = KitParams(seed=seed, bank=bank, taste=taste_model is not None, **plan.get("kit", {}))
    rep = KitReport(asset.name, 0, 0)
    pats: List[Pattern] = []
    project.checkpoint()
    if plan.get("kit") is not None and (plan.get("kit") or plan.get("pattern") is None):
        rep = generate_kit(project, asset, kp, taste_model=taste_model)
    if plan.get("bpm"):
        project.bpm = plan["bpm"]
    if "swing" in plan:
        project.swing = plan["swing"]
    spec = plan.get("pattern")
    if spec:
        for i in range(spec["count"]):
            pt = generate_pattern(project, bank, spec["bars"], spec["style"], spec["density"], 0.7, plan.get("swing", 0.0), seed + i, taste_model=taste_model)
            project.patterns[pt.name] = pt
            pats.append(pt)
        if pats:
            project.current_pattern = pats[0].name
    return rep, pats
