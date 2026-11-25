"""Offline DSP primitives (NumPy/SciPy only). Everything here works on float32 (n, 2) arrays."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import lfilter, resample_poly

F32 = np.float32


# ----------------------------------------------------------------------------- basics
def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-12)))


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Return a float32 (n, 2) array from mono, stereo or multichannel input."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return np.ascontiguousarray(x, dtype=F32)


def mono(x: np.ndarray) -> np.ndarray:
    return ensure_2d(x).mean(axis=1)


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if len(x) else 0.0


def normalize(x: np.ndarray, target_db: float = -0.3) -> np.ndarray:
    p = peak(x)
    if p < 1e-9:
        return x
    return (x * (db_to_lin(target_db) / p)).astype(F32)


def fade(x: np.ndarray, in_ms: float = 0.0, out_ms: float = 0.0, sr: int = 48000) -> np.ndarray:
    x = ensure_2d(x).copy()
    n = len(x)
    ni = min(int(sr * in_ms / 1000.0), n)
    no = min(int(sr * out_ms / 1000.0), n)
    if ni > 1:
        x[:ni] *= np.linspace(0.0, 1.0, ni, dtype=F32)[:, None]
    if no > 1:
        x[n - no:] *= np.linspace(1.0, 0.0, no, dtype=F32)[:, None]
    return x


def declick(x: np.ndarray, sr: int = 48000, in_ms: float = 0.5, out_ms: float = 1.5) -> np.ndarray:
    """Bake tiny fades into a rendered sample so voices never start or end on a discontinuity."""
    return fade(x, in_ms, out_ms, sr)


def soft_limit(x: np.ndarray, knee: float = 0.9) -> np.ndarray:
    """Transparent below the knee, smooth saturation above it. Output magnitude is always < 1."""
    a = np.abs(x)
    over = a > knee
    if not over.any():
        return x
    room = 1.0 - knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + room * 0.999 * np.tanh((a[over] - knee) / room))
    return y


# ----------------------------------------------------------------------------- resampling
def resample_ratio(x: np.ndarray, ratio: float) -> np.ndarray:
    """Change length by `ratio` (output = input * ratio) with a polyphase filter. Pitch moves with it."""
    x = ensure_2d(x)
    if abs(ratio - 1.0) < 1e-9 or len(x) == 0:
        return x.copy()
    fr = Fraction(ratio).limit_denominator(240)
    y = resample_poly(x.astype(np.float64), fr.numerator, fr.denominator, axis=0)
    return np.ascontiguousarray(y, dtype=F32)


def resample_sr(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return ensure_2d(x)
    try:
        import soxr  # optional, very high quality

        return ensure_2d(soxr.resample(ensure_2d(x), sr_in, sr_out, quality="HQ"))
    except Exception:
        pass
    fr = Fraction(sr_out, sr_in).limit_denominator(1000)
    y = resample_poly(ensure_2d(x).astype(np.float64), fr.numerator, fr.denominator, axis=0)
    return np.ascontiguousarray(y, dtype=F32)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n].copy()
    return np.pad(x, ((0, n - len(x)), (0, 0)))


# ----------------------------------------------------------------------------- time stretch / pitch
def wsola(x: np.ndarray, ratio: float, frame: int = 1024, tol: int = 192) -> np.ndarray:
    """Waveform-similarity overlap-add. ratio = output_length / input_length. Pitch is preserved."""
    x = ensure_2d(x)
    n = len(x)
    if abs(ratio - 1.0) < 1e-6:
        return x.copy()
    if n < frame * 3:
        return resample_ratio(x, ratio)  # too short for OLA: behave like a classic sampler
    hs = frame // 2
    ha = hs / ratio
    nout = int(round(n * ratio))
    nframes = int(np.ceil(nout / hs)) + 1
    win = np.hanning(frame + 1)[:frame].astype(F32)       # periodic Hann: sums to 1 at 50% overlap
    win0 = win.copy()
    win0[:hs] = 1.0                                       # no fade-in on the very first frame
    pad = frame + tol
    xp = np.pad(x, ((pad, pad + frame * 2), (0, 0)))
    ms = xp.mean(axis=1)
    out = np.zeros((nframes * hs + frame, 2), dtype=F32)
    pos = pad
    out[:frame] += xp[pos:pos + frame] * win0[:, None]
    prev = pos
    dec = 4
    for k in range(1, nframes):
        target = pad + int(round(k * ha))
        nat = prev + hs
        lo = max(target - tol, 0)
        hi = min(target + tol, len(ms) - frame - 1)
        if hi <= lo:
            pos = lo
        else:
            ref = ms[nat:nat + frame:dec]
            cand = ms[lo:hi + frame:dec]
            nref = len(ref)
            if len(cand) < nref + 1:
                pos = target
            else:
                c = np.correlate(cand, ref, mode="valid")
                e = np.cumsum(np.concatenate(([0.0], cand * cand)))
                en = e[nref:nref + len(c)] - e[:len(c)]
                score = c / np.sqrt(en + 1e-9)
                best = int(np.argmax(score)) * dec
                pos = lo + best
        out[k * hs:k * hs + frame] += xp[pos:pos + frame] * win[:, None]
        prev = pos
    return out[:nout].copy()


def time_stretch(x: np.ndarray, ratio: float) -> np.ndarray:
    return wsola(x, ratio)


def pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """Pitch shift at constant duration: resample (pitch and speed move) then WSOLA back to length."""
    x = ensure_2d(x)
    if abs(semitones) < 1e-6:
        return x.copy()
    f = 2.0 ** (semitones / 12.0)
    n = len(x)
    y = resample_ratio(x, 1.0 / f)
    if len(y) < 3072:
        return y  # one-shots this short: resampling alone (classic sampler behaviour)
    return fit_length(wsola(y, f), n)


# ----------------------------------------------------------------------------- filters
def biquad_coeffs(kind: str, fc: float, q: float, sr: int, gain_db: float = 0.0):
    """RBJ cookbook biquad. kinds: lp, hp, bp, notch, peak, lowshelf, highshelf."""
    fc = float(np.clip(fc, 10.0, sr * 0.49))
    w0 = 2.0 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * max(q, 0.05))
    A = 10.0 ** (gain_db / 40.0)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bp":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "notch":
        b = [1.0, -2 * cw, 1.0]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
    elif kind == "highshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
    else:
        raise ValueError("unknown filter kind: %s" % kind)
    b = np.array(b, dtype=np.float64) / a[0]
    a = np.array(a, dtype=np.float64) / a[0]
    return b, a


def apply_filter(x: np.ndarray, kind: str, fc: float, q: float = 0.707, sr: int = 48000, gain_db: float = 0.0):
    b, a = biquad_coeffs(kind, fc, q, sr, gain_db)
    return lfilter(b, a, ensure_2d(x), axis=0).astype(F32)


# ----------------------------------------------------------------------------- creative transforms
def bitcrush(x: np.ndarray, bits: int = 8, downsample: int = 1) -> np.ndarray:
    x = ensure_2d(x)
    if downsample > 1:
        held = np.repeat(x[::downsample], downsample, axis=0)[: len(x)]
        x = fit_length(held, len(x))
    if bits < 24:
        levels = 2.0 ** (max(int(bits), 1) - 1)
        x = np.round(x * levels) / levels
    return x.astype(F32)


def drive(x: np.ndarray, amount: float = 0.5) -> np.ndarray:
    k = 1.0 + 8.0 * float(np.clip(amount, 0.0, 1.0))
    y = np.tanh(ensure_2d(x) * k) / np.tanh(k)
    return y.astype(F32)


def tape_stop(x: np.ndarray, ms: float = 400.0, sr: int = 48000) -> np.ndarray:
    """Slow the tail of the sample down to a halt, like pulling the plug on a turntable."""
    x = ensure_2d(x)
    n = len(x)
    m = min(int(sr * ms / 1000.0), n)
    if m < 16:
        return x.copy()
    head, tail = x[: n - m], x[n - m:]
    speed = np.linspace(1.0, 0.02, m)
    idx = np.cumsum(speed) - speed[0]
    idx = idx[idx < m - 1]
    i0 = idx.astype(np.int64)
    fr = (idx - i0).astype(F32)[:, None]
    seg = tail[i0] * (1 - fr) + tail[i0 + 1] * fr
    seg = fade(seg, 0, 30, sr)
    return np.concatenate([head, seg]).astype(F32)


def stutter(x: np.ndarray, slice_ms: float = 60.0, repeats: int = 4, sr: int = 48000) -> np.ndarray:
    x = ensure_2d(x)
    m = max(int(sr * slice_ms / 1000.0), 32)
    seg = fade(x[:m], 0.3, 0.6, sr)
    return np.concatenate([np.tile(seg, (max(repeats, 1), 1)), x[m:]]).astype(F32)


def gate_pattern(x: np.ndarray, steps: str = "1011", rate_hz: float = 16.0, depth: float = 1.0, sr: int = 48000):
    x = ensure_2d(x)
    pat = np.array([1.0 if c == "1" else 1.0 - depth for c in steps], dtype=F32)
    t = (np.arange(len(x)) * rate_hz / sr).astype(np.int64) % len(pat)
    env = pat[t]
    # smooth the edges (1 ms) so the gate itself does not click
    k = max(int(sr * 0.001), 2)
    env = np.convolve(env, np.ones(k, dtype=F32) / k, mode="same")
    return (x * env[:, None]).astype(F32)


def granular(x: np.ndarray, grain_ms: float = 40.0, jitter: float = 0.5, density: float = 1.0, seed: int = 0, sr: int = 48000):
    """Scatter windowed grains of the input back over the same length (deterministic for a seed)."""
    x = ensure_2d(x)
    n = len(x)
    g = max(int(sr * grain_ms / 1000.0), 64)
    if n < g * 2:
        return x.copy()
    rng = np.random.default_rng(seed)
    win = np.hanning(g).astype(F32)[:, None]
    out = np.zeros_like(x)
    hop = max(int(g / (2.0 * max(density, 0.1))), 1)
    for t in range(0, n - g, hop):
        src = int(np.clip(t + rng.normal(0, jitter * n * 0.15), 0, n - g))
        out[t:t + g] += x[src:src + g] * win
    return normalize(out, -1.0) if peak(out) > 1.0 else out


def vinyl_crackle(n: int, sr: int = 48000, density: float = 0.0008, level: float = 0.04, seed: int = 0):
    rng = np.random.default_rng(seed)
    pops = (rng.random(n) < density).astype(F32) * rng.uniform(-1, 1, n).astype(F32)
    hiss = rng.normal(0, 0.15, n).astype(F32)
    y = pops + hiss * 0.15
    y = lfilter(*biquad_coeffs("hp", 800.0, 0.707, sr), y)
    return np.stack([y, np.roll(y, 7)], axis=1).astype(F32) * level
