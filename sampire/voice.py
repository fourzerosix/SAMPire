"""One playing instance of a pad sample. Fast paths avoid per-block allocation where possible."""

from __future__ import annotations

import numpy as np

F32 = np.float32
_AR = np.arange(8192, dtype=np.float64)


class Voice:
    __slots__ = ("pad", "data", "rate", "gains", "pos", "loop", "mode", "group", "send_a", "send_b", "a", "d", "sus", "r",
                 "stage", "t", "lvl", "rel_from", "dead", "seq", "started", "tag")

    def __init__(self, pad: int, data: np.ndarray, rate: float, gains: np.ndarray, a: int, d: int, sus: float, r: int,
                 loop: bool, mode: str, group: int, send_a: float, send_b: float, seq: int, started: int, tag: int = 0):
        self.pad, self.data, self.rate, self.gains = pad, data, float(rate), gains.astype(F32)
        self.pos = 0.0
        self.loop, self.mode, self.group = loop, mode, group
        self.send_a, self.send_b = send_a, send_b
        self.a, self.d, self.sus, self.r = int(a), int(d), float(sus), max(int(r), 1)
        self.stage = 0 if self.a > 0 else (1 if (self.d > 0 and self.sus < 1.0) else 2)
        self.t = 0
        self.lvl = 0.0 if self.a > 0 else 1.0
        self.rel_from = 1.0
        self.dead = False
        self.seq, self.started, self.tag = seq, started, tag

    # -- control
    @property
    def releasing(self) -> bool:
        return self.stage == 3

    def release(self, frames: int = 0) -> None:
        if self.stage == 3 or self.dead:
            return
        self.rel_from = self.lvl if self.lvl > 0 else 1.0
        self.stage, self.t = 3, 0
        if frames:
            self.r = max(int(frames), 1)

    # -- envelope
    def _envelope(self, n: int):
        """Scalar when the level is steady for the whole block, otherwise a float32 vector."""
        if self.stage == 2:
            self.lvl = self.sus
            return self.sus
        env = np.empty(n, dtype=F32)
        i = 0
        while i < n:
            if self.stage == 0:
                m = min(n - i, self.a - self.t)
                env[i:i + m] = (self.t + 1 + _AR[:m]) / self.a
                self.t += m
                i += m
                if self.t >= self.a:
                    self.stage, self.t = 1, 0
                    if self.d <= 0 or self.sus >= 1.0:
                        self.stage = 2
            elif self.stage == 1:
                m = min(n - i, self.d - self.t)
                env[i:i + m] = 1.0 + (self.sus - 1.0) * (self.t + 1 + _AR[:m]) / self.d
                self.t += m
                i += m
                if self.t >= self.d:
                    self.stage = 2
            elif self.stage == 2:
                env[i:] = self.sus
                i = n
            else:
                m = min(n - i, self.r - self.t)
                if m > 0:
                    env[i:i + m] = self.rel_from * (1.0 - (self.t + 1 + _AR[:m]) / self.r)
                    self.t += m
                    i += m
                if self.t >= self.r:
                    env[i:] = 0.0
                    self.dead = True
                    i = n
        self.lvl = float(env[-1]) if n else self.lvl
        return env

    # -- audio
    def render_into(self, out: np.ndarray) -> bool:
        """Add this voice into `out` (n, 2). Returns False once the voice has finished."""
        n = len(out)
        if self.dead or n == 0:
            return not self.dead
        data, ln = self.data, len(self.data)
        env = self._envelope(n)
        if self.rate == 1.0 and not self.loop:
            p = int(self.pos)
            m = min(n, ln - p)
            if m <= 0:
                return False
            seg = data[p:p + m]
            if isinstance(env, float):
                out[:m] += seg * (self.gains * env)
            else:
                out[:m] += seg * self.gains * env[:m, None]
            self.pos = p + m
            return not self.dead and self.pos < ln
        # general path: linear interpolation, optional looping
        idx = self.pos + self.rate * _AR[:n]
        if self.loop:
            idx = np.mod(idx, ln)
            i0 = idx.astype(np.int64)
            fr = (idx - i0).astype(F32)[:, None]
            seg = data[i0] * (1.0 - fr) + data[(i0 + 1) % ln] * fr
            m = n
            self.pos = float((self.pos + self.rate * n) % ln)
        else:
            m = int(np.searchsorted(idx, ln - 1.0))
            if m <= 0:
                return False
            i0 = idx[:m].astype(np.int64)
            fr = (idx[:m] - i0).astype(F32)[:, None]
            seg = data[i0] * (1.0 - fr) + data[i0 + 1] * fr
            self.pos = float(self.pos + self.rate * n)
        if isinstance(env, float):
            out[:m] += seg * (self.gains * env)
        else:
            out[:m] += seg * self.gains * env[:m, None]
        return not self.dead and (self.loop or m == n)
