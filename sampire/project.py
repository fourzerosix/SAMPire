"""Project model and persistence. A project is a portable folder:

    MyProject/
      project.json      banks, pads, recipes, patterns, song, tempo (human readable, diff friendly)
      project.db        SQLite index: assets (hashes, analysis), takes, edit history
      audio/sources/    imported originals (never modified)
      audio/recordings/ takes and resamples
      autosave/         rolling snapshots
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import asset as assetmod
from .asset import AudioAsset
from .config import DEFAULT_BPM, NUM_BANKS, PADS_PER_BANK, SAMPLE_RATE
from .pad import Bank, Pad, make_banks
from .sequencer import Pattern

FORMAT = 1
AUTOSAVE_KEEP = 5


class ProjectError(Exception):
    """A project folder is missing, corrupt or from an incompatible version. The message is safe to show the user."""


class Project:
    def __init__(self, name: str = "untitled", sr: int = SAMPLE_RATE):
        self.name = name
        self.path: Optional[str] = None
        self.sr = sr
        self.bpm = DEFAULT_BPM
        self.swing = 0.0
        self.seed = 1
        self.assets: Dict[str, AudioAsset] = {}
        self.banks: List[Bank] = make_banks(NUM_BANKS)
        self.patterns: Dict[str, Pattern] = {"P1": Pattern("P1", 1)}
        self.current_pattern = "P1"
        self.song: List[Tuple[str, int]] = []
        self.mode = "pattern"
        self.fx_settings: dict = {}
        self.midi_map: dict = {}               # project level MIDI mapping (overrides the global one)
        self.tags: dict = {}
        self._undo: List[str] = []
        self._redo: List[str] = []
        self.dirty = False

    # -- pads
    def pad(self, gi: int) -> Pad:
        return self.banks[(gi // PADS_PER_BANK) % len(self.banks)].pads[gi % PADS_PER_BANK]

    def all_pads(self):
        for b in self.banks:
            for p in b.pads:
                yield p

    @property
    def pattern(self) -> Pattern:
        if self.current_pattern not in self.patterns:
            self.patterns[self.current_pattern] = Pattern(self.current_pattern, 1)
        return self.patterns[self.current_pattern]

    # -- assets
    def add_asset(self, a: AudioAsset) -> AudioAsset:
        self.assets[a.id] = a
        self.dirty = True
        return a

    def render_all(self, force: bool = False) -> None:
        for p in self.all_pads():
            if not p.empty:
                p.render(self.assets, self.sr, force)

    # -- undo / redo (JSON snapshots of everything except audio)
    def _state(self) -> str:
        return json.dumps(self._model(), sort_keys=True)

    def checkpoint(self) -> None:
        s = self._state()
        if not self._undo or self._undo[-1] != s:
            self._undo.append(s)
            self._undo = self._undo[-100:]
        self._redo.clear()
        self.dirty = True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._state())
        self._apply(json.loads(self._undo.pop()))
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._state())
        self._apply(json.loads(self._redo.pop()))
        return True

    def _model(self) -> dict:
        return {
            "format": FORMAT, "name": self.name, "sr": self.sr, "bpm": self.bpm, "swing": self.swing, "seed": self.seed,
            "mode": self.mode, "current_pattern": self.current_pattern, "song": [[n, r] for n, r in self.song],
            "fx": self.fx_settings, "midi_map": self.midi_map, "banks": [b.to_dict() for b in self.banks],
            "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
        }

    def _apply(self, m: dict) -> None:
        self.name = m.get("name", self.name)
        self.bpm = float(m.get("bpm", self.bpm))
        self.swing = float(m.get("swing", 0.0))
        self.seed = int(m.get("seed", 1))
        self.mode = m.get("mode", "pattern")
        self.current_pattern = m.get("current_pattern", "P1")
        self.song = [(n, int(r)) for n, r in m.get("song", [])]
        self.fx_settings = m.get("fx", {})
        self.midi_map = m.get("midi_map", {})
        self.banks = [Bank.from_dict(b) for b in m.get("banks", [])] or make_banks(NUM_BANKS)
        self.patterns = {k: Pattern.from_dict(v) for k, v in m.get("patterns", {}).items()} or {"P1": Pattern("P1", 1)}
        self.render_all()

    # -- persistence
    def save(self, path: Optional[str] = None, subtype: str = "FLOAT") -> str:
        path = os.path.abspath(path or self.path or self.name)
        for d in ("audio/sources", "audio/recordings", "autosave"):
            os.makedirs(os.path.join(path, d), exist_ok=True)
        db = sqlite3.connect(os.path.join(path, "project.db"))
        db.execute("CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, name TEXT, path TEXT, sha1 TEXT, sr INTEGER,"
                   " frames INTEGER, meta TEXT, analysis TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, note TEXT)")
        db.execute("DELETE FROM assets")
        for a in self.assets.values():
            sub = "recordings" if a.meta.get("origin") in ("resample", "record", "skipback", "generated") else "sources"
            rel = a.path if (a.path and not os.path.isabs(a.path)) else "audio/%s/%s_%s.wav" % (sub, _safe(a.name), a.id)
            full = os.path.join(path, rel)
            if not os.path.exists(full) or a.meta.get("dirty"):
                assetmod.save_audio(full, a.data, a.sr, subtype)
                a.meta.pop("dirty", None)
                a.sha1 = assetmod.sha1_of_file(full)       # hash of the file we just wrote, used to relink moved files
            a.path = rel
            a.sha1 = a.sha1 or assetmod.sha1_of_file(full)
            db.execute("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?)",
                       (a.id, a.name, rel, a.sha1, a.sr, a.frames, json.dumps(a.meta), json.dumps(a.analysis)))
        db.execute("INSERT INTO history (ts, note) VALUES (?, ?)", (time.time(), "save"))
        db.commit()
        db.close()
        m = self._model()
        m["assets"] = [a.id for a in self.assets.values()]
        tmp = os.path.join(path, "project.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=1)
        os.replace(tmp, os.path.join(path, "project.json"))
        self.path, self.dirty = path, False
        self.autosave()
        return path

    def autosave(self) -> Optional[str]:
        """Write a timestamped snapshot of the model (not audio) and keep the newest few."""
        if not self.path:
            return None
        d = os.path.join(self.path, "autosave")
        try:
            os.makedirs(d, exist_ok=True)
            snap = os.path.join(d, "project-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
            tmp = snap + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._model(), fh)
            os.replace(tmp, snap)
            old = sorted(f for f in os.listdir(d) if f.startswith("project-") and f.endswith(".json"))[:-AUTOSAVE_KEEP]
            for f in old:
                os.remove(os.path.join(d, f))
            return snap
        except OSError:
            return None

    @staticmethod
    def autosaves(path: str) -> List[str]:
        d = os.path.join(os.path.abspath(path), "autosave")
        try:
            return sorted((os.path.join(d, f) for f in os.listdir(d) if f.startswith("project-") and f.endswith(".json")), reverse=True)
        except OSError:
            return []

    @staticmethod
    def recover(path: str) -> "Project":
        """Open the newest readable autosave of a project (after a crash or a corrupted project.json)."""
        for snap in Project.autosaves(path):
            try:
                return Project.load(path, model_file=snap)
            except ProjectError:
                continue
        raise ProjectError("no usable autosave found in %s" % os.path.join(path, "autosave"))

    @staticmethod
    def load(path: str, model_file: Optional[str] = None) -> "Project":
        path = os.path.abspath(path)
        mf = model_file or os.path.join(path, "project.json")
        try:
            with open(mf, "r", encoding="utf-8") as fh:
                m = json.load(fh)
        except FileNotFoundError:
            raise ProjectError("not a SAMPire project (no project.json in %s)" % path)
        except (OSError, ValueError) as exc:
            raise ProjectError("project file is unreadable (%s); try Project.recover() / ':recover'" % exc)
        if not isinstance(m, dict) or int(m.get("format", 1)) > FORMAT:
            raise ProjectError("project was made by a newer SAMPire (format %s)" % (m.get("format") if isinstance(m, dict) else "?"))
        p = Project(str(m.get("name", os.path.basename(path))), int(m.get("sr", SAMPLE_RATE)))
        p.path = path
        try:
            p._apply(m)
        except Exception as exc:  # noqa: BLE001
            raise ProjectError("project model is damaged: %s" % exc)
        dbf = os.path.join(path, "project.db")
        if os.path.exists(dbf):
            try:
                db = sqlite3.connect(dbf)
                rows = db.execute("SELECT id,name,path,sha1,sr,frames,meta,analysis FROM assets").fetchall()
                db.close()
            except sqlite3.Error as exc:
                raise ProjectError("project database is damaged (%s)" % exc)
            for (aid, name, rel, sha1, sr, frames, meta, ana) in rows:
                full = os.path.join(path, rel)
                if not os.path.exists(full):
                    found = relink(path, sha1)
                    if found is None:
                        continue
                    full = found
                try:
                    d, fsr = assetmod.read_audio(full)
                except Exception:
                    continue
                p.assets[aid] = AudioAsset(id=aid, name=name, data=assetmod.dsp.ensure_2d(d), sr=int(fsr), path=rel, sha1=sha1,
                                           meta=json.loads(meta or "{}"), analysis=json.loads(ana or "{}"))
        p.render_all()
        p.dirty = False
        return p

    # -- song chain
    def song_add(self, name: str, reps: int = 1) -> None:
        if name not in self.patterns:
            raise ValueError("no pattern named %r" % name)
        self.song.append((name, max(int(reps), 1)))

    def song_remove(self, index: int) -> None:
        del self.song[index]

    def song_clear(self) -> None:
        self.song = []

    def song_move(self, index: int, new_index: int) -> None:
        item = self.song.pop(index)
        self.song.insert(max(min(new_index, len(self.song)), 0), item)

    def song_seconds(self) -> float:
        return sum(self.patterns[n].bars * self.patterns[n].beats_per_bar * 60.0 / self.bpm * r for n, r in self.song if n in self.patterns)

    def asset_names(self) -> List[str]:
        return ["%s (%.1fs)" % (a.name, a.duration) for a in self.assets.values()]


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40] or "asset"


def relink(project_dir: str, sha1: str, search: Optional[str] = None) -> Optional[str]:
    """Find a moved or renamed audio file by content hash."""
    for root in {search or project_dir, project_dir}:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith((".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3")):
                    full = os.path.join(dirpath, f)
                    try:
                        if assetmod.sha1_of_file(full) == sha1:
                            return full
                    except OSError:
                        pass
    return None
