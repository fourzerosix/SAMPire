"""Capture: rolling skip-back buffer, master/resample recorder, threshold recorder for sampling hits."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import dsp

F32 = np.float32


class RingBuffer:
    """Fixed-size circular buffer of stereo frames. Writes never allocate."""

    def __init__(self, frames: int, channels: int = 2):
        self.cap = int(frames)
        self.buf = np.zeros((self.cap, channels), dtype=F32)
        self.head = 0
        self.filled = 0

    def write(self, x: np.ndarray) -> None:
        n = len(x)
        if n == 0:
            return
        if n >= self.cap:
            self.buf[:] = x[-self.cap:]
            self.head, self.filled = 0, self.cap
            return
        end = self.head + n
        if end <= self.cap:
            self.buf[self.head:end] = x
        else:
            k = self.cap - self.head
            self.buf[self.head:] = x[:k]
            self.buf[:n - k] = x[k:]
        self.head = end % self.cap
        self.filled = min(self.filled + n, self.cap)

    def read_last(self, n: int) -> np.ndarray:
        n = min(int(n), self.filled)
        if n <= 0:
            return np.zeros((0, self.buf.shape[1]), dtype=F32)
        start = (self.head - n) % self.cap
        if start + n <= self.cap:
            return self.buf[start:start + n].copy()
        return np.concatenate([self.buf[start:], self.buf[:self.head]])

    def clear(self) -> None:
        self.head = self.filled = 0


class MasterRecorder:
    """Collects blocks in memory while armed (no file I/O in the audio path); `stop()` returns one array."""

    def __init__(self, max_seconds: float = 600.0, sr: int = 48000):
        self.blocks: List[np.ndarray] = []
        self.active = False
        self.frames = 0
        self.max_frames = int(max_seconds * sr)

    def start(self) -> None:
        self.blocks, self.frames, self.active = [], 0, True

    def write(self, x: np.ndarray) -> None:
        if self.active:
            if self.frames + len(x) > self.max_frames:
                self.active = False
                return
            self.blocks.append(x.copy())
            self.frames += len(x)

    def stop(self) -> np.ndarray:
        self.active = False
        out = np.concatenate(self.blocks) if self.blocks else np.zeros((0, 2), dtype=F32)
        self.blocks = []
        return out


class ThresholdRecorder:
    """Waits for the signal to cross a threshold, keeps a short pre-roll, then records until stopped,
    a fixed length is reached, or silence returns. Feed it blocks with push()."""

    def __init__(self, sr: int, threshold_db: float = -40.0, preroll_ms: float = 40.0, fixed_frames: Optional[int] = None,
                 silence_stop_s: Optional[float] = None, max_s: float = 60.0):
        self.sr = sr
        self.thr = dsp.db_to_lin(threshold_db)
        self.pre = int(sr * preroll_ms / 1000.0)
        self.ring = RingBuffer(max(self.pre, 1))
        self.fixed = fixed_frames
        self.silence_stop = int(silence_stop_s * sr) if silence_stop_s else None
        self.max_frames = int(max_s * sr)
        self.state = "armed"                    # armed | recording | done
        self.blocks: List[np.ndarray] = []
        self.frames = 0
        self.quiet = 0

    def push(self, x: np.ndarray) -> str:
        x = dsp.ensure_2d(x)
        if self.state == "armed":
            hot = np.where(np.abs(x).max(axis=1) > self.thr)[0]
            if len(hot):
                tail = self.ring.read_last(self.pre)
                begin = max(len(tail) + int(hot[0]) - self.pre, 0)     # keep exactly `pre` frames before the trigger
                seg = np.concatenate([tail, x])[begin:]
                self.state = "recording"
                self.blocks, self.frames, self.quiet = [], 0, 0
                self._take(seg)
            else:
                self.ring.write(x)
        elif self.state == "recording":
            self._take(x)
        return self.state

    def _take(self, x: np.ndarray) -> None:
        if self.fixed is not None and self.frames + len(x) >= self.fixed:
            x = x[: self.fixed - self.frames]
            self.blocks.append(x.copy())
            self.frames += len(x)
            self.state = "done"
            return
        self.blocks.append(x.copy())
        self.frames += len(x)
        if self.silence_stop:
            self.quiet = 0 if float(np.abs(x).max()) > self.thr else self.quiet + len(x)
            if self.quiet >= self.silence_stop:
                self.state = "done"
        if self.frames >= self.max_frames:
            self.state = "done"

    def stop(self) -> None:
        if self.state == "recording":
            self.state = "done"

    def result(self) -> np.ndarray:
        return np.concatenate(self.blocks) if self.blocks else np.zeros((0, 2), dtype=F32)
