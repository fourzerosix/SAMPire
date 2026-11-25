"""Plan providers: anything that turns a prompt into a creative plan. The plan is a plain dict (see creative.run_plan)
and every plan is validated and clamped here, so a provider (including a language model) can only ever request things
SAMPire already knows how to do: no file paths, no commands, no code.

Built in:
  rules                      offline keyword parser (default, sends nothing anywhere)
  command:<program args>     runs YOUR program; it gets {"prompt", "context"} as JSON on stdin and must print a plan as JSON
  python:<module>:<function> imports and calls function(prompt, context) -> plan dict (explicit opt-in, runs your code)
"""

from __future__ import annotations

import importlib
import json
import shlex
import subprocess
from typing import Callable, Dict, Optional

from . import creative

PlanFn = Callable[[str, dict], dict]
_REG: Dict[str, PlanFn] = {}
MODES = ("transient", "hybrid", "grid", "equal", "silence")
PREFER = ("any", "percussive", "melodic")
LENGTH = ("any", "short", "long")


def register(name: str, fn: PlanFn) -> None:
    _REG[name] = fn


def _clamp(v, lo, hi, cast=float, default=None):
    try:
        return cast(min(max(float(v), lo), hi))
    except (TypeError, ValueError):
        return default


def validate_plan(plan: dict) -> dict:
    """Return a safe copy of a plan: only known keys, clamped numbers, whitelisted names. Unknown keys are dropped."""
    out: dict = {"kit": {}, "pattern": None, "notes": []}
    if not isinstance(plan, dict):
        return out
    k = plan.get("kit") or {}
    if isinstance(k, dict):
        kit = out["kit"]
        if "pads" in k:
            kit["pads"] = _clamp(k["pads"], 2, 16, int, 16)
        if k.get("mode") in MODES:
            kit["mode"] = k["mode"]
        for key in ("sensitivity", "experimental", "keep_original"):
            if key in k:
                kit[key] = _clamp(k[key], 0.0, 1.0, float, 0.5)
        if k.get("prefer") in PREFER:
            kit["prefer"] = k["prefer"]
        if k.get("length") in LENGTH:
            kit["length"] = k["length"]
        if "tail_ms" in k:
            kit["tail_ms"] = _clamp(k["tail_ms"], 50, 4000, float, 600.0)
        if isinstance(k.get("ops"), list):
            ops = [o for o in k["ops"] if isinstance(o, str) and o in creative._TRANSFORMS]
            if ops:
                kit["ops"] = ops
    p = plan.get("pattern")
    if isinstance(p, dict):
        style = p.get("style") if p.get("style") in creative.STYLES + ("auto",) else "break"
        out["pattern"] = {"count": _clamp(p.get("count", 1), 1, 8, int, 1), "bars": _clamp(p.get("bars", 2), 1, 16, int, 2),
                          "style": style, "density": _clamp(p.get("density", 0.5), 0.0, 1.0, float, 0.5)}
    if "bpm" in plan:
        b = _clamp(plan["bpm"], 40, 300, float, None)
        if b:
            out["bpm"] = b
    if "swing" in plan:
        out["swing"] = _clamp(plan["swing"], 0.0, 0.5, float, 0.0)
    if isinstance(plan.get("notes"), list):
        out["notes"] = [str(n)[:200] for n in plan["notes"][:5]]
    return out


def _command_provider(cmdline: str) -> PlanFn:
    argv = shlex.split(cmdline)

    def run(prompt: str, context: dict) -> dict:
        r = subprocess.run(argv, input=json.dumps({"prompt": prompt, "context": context}), capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError("plan command failed: %s" % r.stderr.strip()[:200])
        if len(r.stdout) > 100_000:
            raise RuntimeError("plan command printed too much")
        return json.loads(r.stdout)

    return run


def _python_provider(spec: str) -> PlanFn:
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def get(name: str = "rules", allow_python: bool = False) -> PlanFn:
    if name in _REG:
        return _REG[name]
    if name.startswith("command:"):
        return _command_provider(name[len("command:"):])
    if name.startswith("python:"):
        if not allow_python:
            raise PermissionError("python: providers run your own code; pass allow_python=True (CLI: --allow-python-provider)")
        return _python_provider(name[len("python:"):])
    raise ValueError("unknown plan provider %r (rules, command:..., python:...)" % name)


def plan_from_prompt(prompt: str, provider: str = "rules", context: Optional[dict] = None, allow_python: bool = False) -> dict:
    fn = get(provider, allow_python)
    return validate_plan(fn(prompt, context or {}))


register("rules", lambda prompt, context: creative.parse_instruction(prompt))
