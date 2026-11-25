"""Braille canvas: 2x4 dots per terminal cell for smooth waveforms and meters (same idea as btop)."""

from __future__ import annotations

from typing import List

import numpy as np

_BITS = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]]


class Canvas:
    def __init__(self, cells_w: int, cells_h: int):
        self.cw, self.ch = max(cells_w, 1), max(cells_h, 1)
        self.w, self.h = self.cw * 2, self.ch * 4
        self.buf = np.zeros((self.ch, self.cw), dtype=np.uint16)

    def set(self, x: int, y: int) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            self.buf[y // 4, x // 2] |= _BITS[x % 2][y % 4]

    def vline(self, x: int, y0: int, y1: int) -> None:
        if y0 > y1:
            y0, y1 = y1, y0
        for y in range(max(y0, 0), min(y1, self.h - 1) + 1):
            self.set(x, y)

    def rows(self) -> List[str]:
        return ["".join(chr(0x2800 + int(v)) for v in row) for row in self.buf]


def waveform(data: np.ndarray, cells_w: int, cells_h: int) -> Canvas:
    """Min/max envelope of a stereo or mono array, centred vertically."""
    c = Canvas(cells_w, cells_h)
    n = len(data)
    if n == 0:
        for x in range(c.w):
            c.set(x, c.h // 2)
        return c
    m = data.mean(axis=1) if data.ndim == 2 else data
    mid = (c.h - 1) / 2.0
    edges = np.linspace(0, n, c.w + 1).astype(int)
    for x in range(c.w):
        seg = m[edges[x]:max(edges[x + 1], edges[x] + 1)]
        lo, hi = float(seg.min()), float(seg.max())
        c.vline(x, int(round(mid - hi * mid)), int(round(mid - lo * mid)))
    return c


def markers(c: Canvas, positions, total: int) -> None:
    """Dotted vertical ticks at sample positions (slice boundaries)."""
    for p in positions:
        x = int(p / max(total, 1) * (c.w - 1))
        for y in range(0, c.h, 2):
            c.set(x, y)
