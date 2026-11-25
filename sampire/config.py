"""Global constants and the small JSON settings file (~/.config/sampire/config.json)."""

from __future__ import annotations

import json
import os
from pathlib import Path

SAMPLE_RATE = 48000        # internal rate; every imported asset is converted to this
PPQ = 960                  # sequencer ticks per quarter note (same as the Akai MPC Sample)
BLOCK = 256                # frames per audio callback (5.3 ms at 48 kHz)
MAX_VOICES = 32            # simultaneous stereo voices (SP-404MKII and MPC Sample class)
PADS_PER_BANK = 16         # 4x4 grid
NUM_BANKS = 4              # A-D, 64 pads total
SKIPBACK_SECONDS = 60      # rolling capture length
DEFAULT_BPM = 92.0

DEFAULTS = {
    "device": None,
    "input_device": None,
    "block": BLOCK,
    "theme": "default",
    "seed": 1,
    "last_dir": "",
    "latency_comp_ms": 0.0,
}


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "sampire"


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(config_dir() / "config.json", "r", encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: dict) -> None:
    try:
        d = config_dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "config.json.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        os.replace(tmp, d / "config.json")
    except OSError:
        pass
