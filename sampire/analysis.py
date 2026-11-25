"""Sample intelligence, level 1 and 2: pure NumPy/SciPy analysis (onsets, tempo, key, pitch, hit class).

Everything here is advisory. Results are cached on the asset and can always be overridden by the user.
Optional backends (librosa, aubio-ledfx) can be plugged in through `Backend`; the defaults need neither.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, lfilter

from . import dsp

N_FFT = 1024
HOP = 128


# ----------------------------------------------------------------------------- spectra
def stft_mag(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Magnitude spectrogram shaped (bins, frames). Frame t is centred on sample t * hop."""
    m = dsp.mono(x) if x.ndim == 2 else x
    p = np.pad(m, (n_fft // 2, n_fft // 2))
    if len(p) < n_fft:
        p = np.pad(p, (0, n_fft - len(p)))
    frames = sliding_window_view(p, n_fft)[::hop]
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    return np.abs(np.fft.rfft(frames * win, axis=1)).T.astype(np.float32)


def _flux(mag: np.ndarray, lag: int = 2) -> np.ndarray:
    """SuperFlux-style spectral flux: compare each frame with a frequency-max-filtered earlier frame,
    which suppresses the false onsets that gliding tones (a kick's pitch sweep) would otherwise cause."""
    s = np.log1p(100.0 * mag)
    if s.shape[1] <= lag:
        return np.zeros(s.shape[1])
    ref = maximum_filter1d(s, size=5, axis=0)
    flux = np.maximum(s[:, lag:] - ref[:, :-lag], 0.0).sum(axis=0)
    return np.concatenate([np.zeros(lag), flux])


def onset_envelope(x: np.ndarray, sr: int, band: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Log-spectral flux, computed per band (low/mid/high) so quiet hats are not drowned by a loud kick.

    `band` limits the bins (Hz) to isolate one region, e.g. (30, 160) for the kick onset curve."""
    mag = stft_mag(x)
    f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    if band is not None:
        flux = _flux(mag[(f >= band[0]) & (f < band[1])])
        top = flux.max()
        return (flux / top).astype(np.float32) if top > 0 else flux.astype(np.float32)
    edges = [(20.0, 250.0), (250.0, 3000.0), (3000.0, 20000.0)]
    curves = [_flux(mag[(f >= lo) & (f < hi)]) for lo, hi in edges]
    floor = 0.15 * max(c.max() for c in curves)
    env = np.max([c / max(c.max(), floor, 1e-9) for c in curves], axis=0)
    top = env.max()
    return (env / top).astype(np.float32) if top > 0 else env.astype(np.float32)


def _refine_onset(m: np.ndarray, centre: int, sr: int) -> int:
    """Find the true attack start near a spectral-flux peak: the point of steepest RMS rise (4 ms windows)."""
    w = max(int(0.004 * sr), 16)
    lo = max(centre - N_FFT * 2, w)
    hi = min(centre + N_FFT // 2, len(m) - w)
    if hi <= lo:
        return int(centre)
    base = lo - w
    seg = m[base:hi + w].astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(seg * seg)])
    i = np.arange(lo, hi) - base
    after = np.sqrt((cs[i + w] - cs[i]) / w)
    before = np.sqrt((cs[i] - cs[i - w]) / w)
    score = after - before
    return int(lo + int(np.argmax(score)))


def detect_onsets(x: np.ndarray, sr: int, sensitivity: float = 0.5, min_gap_ms: float = 35.0,
                  band: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (onset sample positions, strengths 0..1). sensitivity 0 = only big hits, 1 = everything."""
    m = dsp.mono(x)
    env = onset_envelope(x, sr, band)
    if len(env) < 4 or env.max() <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    win = max(int(0.35 * sr / HOP), 3)
    local = uniform_filter1d(env, win)
    k = 0.30 * (1.0 - float(np.clip(sensitivity, 0, 1))) + 0.04
    thr = local + k * env.max()
    dist = max(int(min_gap_ms * 1e-3 * sr / HOP), 1)
    pk, _ = find_peaks(env, height=thr, distance=dist)
    centres = pk * HOP
    pos = np.array([_refine_onset(m, int(c), sr) for c in centres], dtype=np.int64)
    strength = env[pk].astype(np.float32)
    head = int(0.01 * sr)
    if len(m) > head and np.max(np.abs(m[:head])) > 0.2 * np.max(np.abs(m)) and (len(pos) == 0 or pos[0] > head * 2):
        pos = np.concatenate([[0], pos]).astype(np.int64)      # a hit that starts at sample 0 has no flux
        strength = np.concatenate([[1.0], strength]).astype(np.float32)
    if len(pos):
        order = np.argsort(pos)
        pos, strength = pos[order], strength[order]
        keep = np.concatenate([[True], np.diff(pos) > int(min_gap_ms * 1e-3 * sr * 0.5)])
        pos, strength = pos[keep], strength[keep]
    return pos, strength


# ----------------------------------------------------------------------------- tempo
def estimate_tempo(x: np.ndarray, sr: int, lo: float = 60.0, hi: float = 200.0) -> Dict[str, object]:
    """Autocorrelation tempo estimate with a mild log-Gaussian prior around 110 BPM."""
    env = onset_envelope(x, sr)
    fps = sr / float(HOP)
    if len(env) < int(fps * 2):
        return {"bpm": 0.0, "confidence": 0.0, "beats": np.zeros(0), "downbeats": np.zeros(0)}
    e = env - env.mean()
    nfft = 1 << int(np.ceil(np.log2(len(e) * 2)))
    f = np.fft.rfft(e, nfft)
    ac = np.fft.irfft(f * np.conj(f))[: len(e)]
    ac = ac / (ac[0] + 1e-12)
    lags = np.arange(len(ac))

    def at(lag: np.ndarray) -> np.ndarray:
        return np.interp(lag, lags, ac, right=0.0)

    cands = np.arange(lo, hi + 0.25, 0.25)
    score = np.zeros(len(cands))
    for i, b in enumerate(cands):
        lag = fps * 60.0 / b
        s = at(lag) + 0.5 * at(2 * lag) + 0.25 * at(4 * lag)
        prior = np.exp(-0.5 * (np.log2(b / 105.0) / 0.7) ** 2)
        score[i] = s * prior
    j = int(np.argmax(score))
    bpm = float(cands[j])
    conf = float(np.clip(score[j] / (np.mean(np.abs(score)) * 4.0 + 1e-9), 0.0, 1.0))
    # phase: where do the beats sit?
    period = fps * 60.0 / bpm
    phases = np.arange(0.0, period, 0.5)
    best_p, best_v = 0.0, -1.0
    for ph in phases:
        idx = np.arange(ph, len(env) - 1, period)
        v = float(np.interp(idx, np.arange(len(env)), env).sum())
        if v > best_v:
            best_p, best_v = ph, v
    beat_frames = np.arange(best_p, len(env) - 1, period)
    beats = beat_frames * HOP / float(sr)
    downbeats = beats[:0]
    if len(beats) >= 8:
        low = onset_envelope(x, sr, band=(30.0, 160.0))
        vals = np.interp(beat_frames, np.arange(len(low)), low)
        off = int(np.argmax([vals[o::4].sum() for o in range(4)]))
        downbeats = beats[off::4]
    return {"bpm": bpm, "confidence": conf, "beats": beats, "downbeats": downbeats}


# ----------------------------------------------------------------------------- pitch and key
def estimate_pitch(x: np.ndarray, sr: int, fmin: float = 40.0, fmax: float = 2000.0) -> Tuple[float, float]:
    """Autocorrelation f0 estimate. Returns (hz, clarity 0..1); (0, 0) when nothing tonal is found."""
    m = dsp.mono(x)
    if len(m) < 512:
        return 0.0, 0.0
    # use the loudest 8192 samples, skipping the first few ms of attack noise
    skip = min(int(0.01 * sr), len(m) // 4)
    m = m[skip:]
    if len(m) > 8192:
        e = uniform_filter1d(m.astype(np.float64) ** 2, 1024)
        c = int(np.argmax(e))
        a = int(np.clip(c - 4096, 0, len(m) - 8192))
        m = m[a:a + 8192]
    m = (m - m.mean()) * np.hanning(len(m))
    nfft = 1 << int(np.ceil(np.log2(len(m) * 2)))
    f = np.fft.rfft(m, nfft)
    r = np.fft.irfft(np.abs(f) ** 2)[: len(m)]
    if r[0] < 1e-9:
        return 0.0, 0.0
    r = r / r[0]
    lmin, lmax = int(sr / fmax), min(int(sr / fmin), len(r) - 2)
    if lmax <= lmin + 2:
        return 0.0, 0.0
    seg = r[lmin:lmax]
    top = float(seg.max())
    if top < 0.35:
        return 0.0, max(top, 0.0)
    cand = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] > 0.9 * top))[0]
    i = (int(cand[0]) + 1) if len(cand) else int(np.argmax(seg))
    lag = lmin + i
    a, b, c = r[lag - 1], r[lag], r[lag + 1]
    den = a - 2 * b + c
    shift = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    return float(sr / (lag + shift)), float(np.clip(top, 0.0, 1.0))


_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_KMAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KMIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * np.log2(max(f, 1e-6) / 440.0)


def midi_name(m: float) -> str:
    n = int(round(m))
    return "%s%d" % (_NOTES[n % 12], n // 12 - 1)


def estimate_key(x: np.ndarray, sr: int) -> Tuple[str, float]:
    """Krumhansl-Schmuckler key estimate from a chroma vector. Returns (name, correlation)."""
    mag = stft_mag(x, n_fft=4096, hop=1024)
    if mag.shape[1] >= 9:
        mag = median_filter(mag, size=(1, 9))      # keep the sustained (harmonic) part, drop drum hits
    f = np.fft.rfftfreq(4096, 1.0 / sr)
    sel = (f > 90.0) & (f < 1800.0)
    pc = np.round(12 * np.log2(f[sel] / 440.0) + 69).astype(int) % 12
    chroma = np.zeros(12)
    spec = (mag[sel] ** 0.5).sum(axis=1)      # compressed so one loud kick cannot own the chroma
    np.add.at(chroma, pc, spec)
    if chroma.sum() <= 0:
        return "unknown", 0.0
    chroma = chroma / chroma.sum()
    best, name = -2.0, "unknown"
    for k in range(12):
        for prof, mode in ((_KMAJ, "major"), (_KMIN, "minor")):
            c = np.corrcoef(chroma, np.roll(prof, k))[0, 1]
            if c > best:
                best, name = float(c), "%s %s" % (_NOTES[k], mode)
    return name, best


def percussive_ratio(x: np.ndarray) -> float:
    """Median-filter harmonic/percussive split of the spectrogram. 1.0 = all drums, 0.0 = all sustained."""
    mag = stft_mag(x)
    if mag.shape[1] < 20:
        return 0.5
    h = median_filter(mag, size=(1, 17))
    p = median_filter(mag, size=(17, 1))
    mp = p ** 2 / (h ** 2 + p ** 2 + 1e-12)
    return float((mp * mag ** 2).sum() / ((mag ** 2).sum() + 1e-12))


# ----------------------------------------------------------------------------- loudness
_K1 = (np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
       np.array([1.0, -1.69065929318241, 0.73248077421585]))
_K2 = (np.array([1.0, -2.0, 1.0]), np.array([1.0, -1.99004745483398, 0.99007225036621]))


def loudness_lufs(x: np.ndarray, sr: int) -> float:
    """BS.1770 K-weighted loudness, ungated, valid at 48 kHz. Good enough for relative level matching."""
    if sr != 48000 or len(x) == 0:
        return float("-inf")
    y = dsp.ensure_2d(x).astype(np.float64)
    y = lfilter(*_K1, y, axis=0)
    y = lfilter(*_K2, y, axis=0)
    ms = float(np.mean(y ** 2, axis=0).sum())
    return -0.691 + 10.0 * np.log10(ms + 1e-12)


# ----------------------------------------------------------------------------- one-shot description
def describe_hit(x: np.ndarray, sr: int) -> Dict[str, float]:
    """Cheap acoustic features of a short sample, used for classification and for ranking variety."""
    m = dsp.mono(x)
    n = len(m)
    if n < 64:
        return {"dur": n / float(sr), "peak": 0.0, "rms": 0.0, "centroid": 0.0, "low": 0.0, "high": 0.0,
                "flat": 0.0, "decay": 0.0, "zcr": 0.0, "f0": 0.0, "clarity": 0.0}
    seg = m[: int(min(n, 0.25 * sr))]
    w = np.hanning(len(seg))
    sp = np.abs(np.fft.rfft(seg * w)) ** 2
    f = np.fft.rfftfreq(len(seg), 1.0 / sr)
    tot = sp.sum() + 1e-12
    centroid = float((f * sp).sum() / tot)
    low = float(sp[f < 200].sum() / tot)
    high = float(sp[f > 5000].sum() / tot)
    band = sp[(f > 100) & (f < 12000)] + 1e-12
    flat = float(np.exp(np.mean(np.log(band))) / np.mean(band))
    env = uniform_filter1d(np.abs(m), max(int(0.015 * sr), 8))
    pk = float(env.max())
    ip = int(np.argmax(env))
    after = np.where(env[ip:] < pk * 0.1)[0]
    decay = float((after[0] if len(after) else n - ip) / sr)
    zcr = float(np.mean(np.abs(np.diff(np.sign(m))) > 0))
    f0, clarity = estimate_pitch(x, sr)
    return {"dur": n / float(sr), "peak": float(np.max(np.abs(m))), "rms": float(np.sqrt(np.mean(m ** 2))),
            "centroid": centroid, "low": low, "high": high, "flat": flat, "decay": decay, "zcr": zcr,
            "f0": f0, "clarity": clarity}


def classify_hit(feat: Dict[str, float]) -> str:
    """Rule-based hit classifier. Labels: kick, snare, hat, open_hat, tom, bass, tone, perc.

    Deliberately simple and explainable; swap in an ML model through `Backend.classify` later."""
    c, low, high, flat = feat["centroid"], feat["low"], feat["high"], feat["flat"]
    decay, f0, clar, dur = feat["decay"], feat["f0"], feat["clarity"], feat["dur"]
    if low > 0.55 and c < 500:
        if clar > 0.8 and decay > 0.35 and dur > 0.3:
            return "bass"
        return "kick"
    if high > 0.4 or c > 5000:
        return "open_hat" if decay > 0.12 else "hat"
    if clar > 0.85 and f0 > 0 and flat < 0.2:
        return "bass" if f0 < 130 else "tone"
    if 1000 < c < 5000 and flat > 0.02:
        return "snare"
    if 150 < c <= 1000 and decay < 0.4:
        return "tom"
    return "perc"


class Backend:
    """Extension point: subclass and register to replace any analysis step with a library or ML model."""

    def classify(self, x: np.ndarray, sr: int) -> str:
        return classify_hit(describe_hit(x, sr))


BACKEND = Backend()


def classify_sample(x: np.ndarray, sr: int) -> str:
    return BACKEND.classify(x, sr)


# ----------------------------------------------------------------------------- full report
def analyze(x: np.ndarray, sr: int, sensitivity: float = 0.5) -> Dict[str, object]:
    """The Level 1 analysis pass run on import. All values are json-safe."""
    x = dsp.ensure_2d(x)
    pos, strength = detect_onsets(x, sr, sensitivity)
    t = estimate_tempo(x, sr)
    key, kconf = estimate_key(x, sr) if len(x) > 4096 else ("unknown", 0.0)
    pk = dsp.peak(x)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0
    return {
        "duration": len(x) / float(sr),
        "frames": int(len(x)),
        "sr": int(sr),
        "peak_db": dsp.lin_to_db(pk),
        "rms_db": dsp.lin_to_db(rms),
        "lufs": loudness_lufs(x, sr),
        "bpm": float(t["bpm"]),
        "bpm_confidence": float(t["confidence"]),
        "beats": [float(b) for b in t["beats"][:512]],
        "downbeats": [float(b) for b in t["downbeats"][:128]],
        "onsets": [int(p) for p in pos[:2048]],
        "onset_strength": [float(s) for s in strength[:2048]],
        "key": key,
        "key_confidence": float(kconf),
        "percussive_ratio": percussive_ratio(x) if len(x) > 4096 else 0.5,
    }
