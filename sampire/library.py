"""Sample library: folder scanning, tags, ratings, favourites, BPM/key filters, duplicate detection,
similarity search, and safe sample-pack import. SQLite at ~/.local/share/sampire/library.db (override with SAMPIRE_LIBRARY)."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import analysis, dsp
from . import asset as assetmod
from .config import SAMPLE_RATE

AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3")
TAG_WORDS = {
    "kick": "kick", "kik": "kick", "bd": "kick", "snare": "snare", "sd": "snare", "clap": "clap", "hat": "hat", "hihat": "hat",
    "hh": "hat", "tom": "tom", "perc": "perc", "cymbal": "cymbal", "crash": "cymbal", "ride": "cymbal", "bass": "bass",
    "808": "808", "loop": "loop", "break": "break", "breaks": "break", "vocal": "vocal", "vox": "vocal", "fx": "fx", "riser": "fx",
    "oneshot": "one-shot", "one-shot": "one-shot", "shot": "one-shot", "chord": "chord", "pad": "pad", "lead": "lead", "pluck": "pluck",
    "guitar": "guitar", "piano": "piano", "keys": "keys", "synth": "synth", "drums": "drums", "foley": "foley", "texture": "texture",
}


def data_dir() -> str:
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "sampire")


def db_path() -> str:
    return os.environ.get("SAMPIRE_LIBRARY") or os.path.join(data_dir(), "library.db")


def parse_filename_meta(path: str) -> Dict[str, object]:
    """Tags, BPM and key hints from folder and file names (kick, 120bpm, Am, ...)."""
    parts = re.split(r"[\\/]", os.path.splitext(path)[0].lower())[-4:]
    tags, bpm, key = set(), None, None
    for part in parts:
        for tok in re.split(r"[^a-z0-9#+-]+", part):
            if tok in TAG_WORDS:
                tags.add(TAG_WORDS[tok])
            elif tok.endswith("s") and tok[:-1] in TAG_WORDS:
                tags.add(TAG_WORDS[tok[:-1]])
        m = re.search(r"(\d{2,3})\s*-?\s*bpm", part)
        if m and 40 <= int(m.group(1)) <= 300:
            bpm = float(m.group(1))
        k = re.search(r"(?:^|[_\s-])([a-g][#b]?)(maj|major|min|minor|m)(?:$|[_\s-])", part)
        if k:
            note = k.group(1)[0].upper() + k.group(1)[1:]
            key = "%s %s" % (note, "minor" if k.group(2) in ("m", "min", "minor") else "major")
    return {"tags": sorted(tags), "bpm": bpm, "key": key}


@dataclass
class Row:
    id: int
    path: str
    name: str
    dur: float
    bpm: float
    key: str
    label: str
    rating: int
    fav: bool
    tags: List[str]
    peak_db: float = 0.0
    score: float = 0.0


class Library:
    def __init__(self, path: Optional[str] = None):
        self.path = path or db_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, name TEXT, dur REAL, sr INTEGER, channels INTEGER,
            bpm REAL, bpm_conf REAL, key TEXT, peak_db REAL, lufs REAL, perc REAL, label TEXT, filehash TEXT,
            added REAL, last_used REAL, rating INTEGER DEFAULT 0, fav INTEGER DEFAULT 0, tags TEXT DEFAULT '',
            emb BLOB, size INTEGER, mtime REAL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_hash ON samples(filehash)")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ scanning
    def scan(self, folders: Iterable[str], progress: Optional[Callable[[str], None]] = None, rescan: bool = False,
             extra_tags: Iterable[str] = ()) -> Tuple[int, int]:
        """Index every audio file under the folders. Unchanged files (size+mtime) are skipped. Returns (added, skipped)."""
        added = skipped = 0
        for folder in folders:
            for dirpath, _, files in os.walk(os.path.expanduser(folder)):
                for f in sorted(files):
                    if not f.lower().endswith(AUDIO_EXT):
                        continue
                    full = os.path.abspath(os.path.join(dirpath, f))
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    row = self.db.execute("SELECT size, mtime FROM samples WHERE path=?", (full,)).fetchone()
                    if row and not rescan and row[0] == st.st_size and abs(row[1] - st.st_mtime) < 1e-3:
                        skipped += 1
                        continue
                    try:
                        self._index(full, st, extra_tags)
                        added += 1
                        if progress and added % 10 == 0:
                            progress("indexed %d files..." % added)
                    except Exception:
                        skipped += 1
        self.db.commit()
        return added, skipped

    def _index(self, full: str, st, extra_tags: Iterable[str]) -> None:
        d, fsr = assetmod.read_audio(full)
        x = dsp.ensure_2d(d)
        if fsr != SAMPLE_RATE:
            x = dsp.resample_sr(x, fsr, SAMPLE_RATE)
        sr = SAMPLE_RATE
        dur = len(x) / sr
        meta = parse_filename_meta(full)
        head = x[: 30 * sr]
        bpm = conf = 0.0
        key = ""
        label = ""
        if dur >= 1.5:
            t = analysis.estimate_tempo(head, sr)
            bpm, conf = float(t["bpm"]), float(t["confidence"])
            key = analysis.estimate_key(head, sr)[0] if len(head) > 4096 else ""
            if key == "unknown":
                key = ""
        else:
            label = analysis.classify_sample(head, sr)
        if meta["bpm"]:
            bpm, conf = float(meta["bpm"]), 1.0
        if meta["key"]:
            key = str(meta["key"])
        tags = set(meta["tags"]) | {t.lower() for t in extra_tags}
        if label:
            tags.add(label)
        if dur >= 1.5:
            tags.add("loop" if conf > 0.3 else "long")
        elif dur < 1.5:
            tags.add("one-shot")
        emb = analysis.embedding(head, sr).astype(np.float32).tobytes()
        fh = assetmod.sha1_of_file(full)
        self.db.execute(
            "INSERT OR REPLACE INTO samples (path,name,dur,sr,channels,bpm,bpm_conf,key,peak_db,lufs,perc,label,filehash,added,last_used,rating,fav,tags,emb,size,mtime)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,COALESCE((SELECT last_used FROM samples WHERE path=?),0),"
            " COALESCE((SELECT rating FROM samples WHERE path=?),0), COALESCE((SELECT fav FROM samples WHERE path=?),0),"
            " ?,?,?,?)",
            (full, os.path.splitext(os.path.basename(full))[0], dur, fsr, d.shape[1], bpm, conf, key, dsp.lin_to_db(dsp.peak(x)),
             analysis.loudness_lufs(head, sr), 0.0, label, fh, time.time(), full, full, full, ",".join(sorted(tags)), emb,
             st.st_size, st.st_mtime))

    # ------------------------------------------------------------------ queries
    def _rows(self, cur) -> List[Row]:
        return [Row(r[0], r[1], r[2], r[3], r[4] or 0.0, r[5] or "", r[6] or "", r[7], bool(r[8]), [t for t in (r[9] or "").split(",") if t], r[10])
                for r in cur.fetchall()]

    COLS = "id,path,name,dur,bpm,key,label,rating,fav,tags,peak_db"

    def search(self, text: str = "", bpm: Optional[Tuple[float, float]] = None, key: str = "", tags: Iterable[str] = (),
               min_rating: int = 0, fav_only: bool = False, recent: bool = False, limit: int = 200) -> List[Row]:
        q, args = ["1=1"], []
        for w in text.split():
            q.append("(lower(name) LIKE ? OR lower(path) LIKE ? OR tags LIKE ?)")
            args += ["%" + w.lower() + "%"] * 3
        if bpm:
            q.append("bpm BETWEEN ? AND ?")
            args += [bpm[0], bpm[1]]
        if key:
            q.append("lower(key) LIKE ?")
            args.append(key.lower() + "%")
        for t in tags:
            q.append("(',' || tags || ',') LIKE ?")
            args.append("%," + t.lower() + ",%")
        if min_rating:
            q.append("rating >= ?")
            args.append(min_rating)
        if fav_only:
            q.append("fav = 1")
        order = "last_used DESC" if recent else "name COLLATE NOCASE"
        cur = self.db.execute("SELECT %s FROM samples WHERE %s ORDER BY %s LIMIT ?" % (self.COLS, " AND ".join(q), order), args + [limit])
        return self._rows(cur)

    def find(self, query: str, limit: int = 200) -> List[Row]:
        """Parse 'kick bpm:85-95 key:C tag:drums rating:3 fav recent' style queries."""
        words, bpm, key, tags, rating, fav, recent = [], None, "", [], 0, False, False
        for tok in query.split():
            if tok.startswith("bpm:"):
                a = tok[4:].split("-")
                try:
                    bpm = (float(a[0]), float(a[-1]))
                except ValueError:
                    pass
            elif tok.startswith("key:"):
                key = tok[4:]
            elif tok.startswith("tag:"):
                tags.append(tok[4:])
            elif tok.startswith("rating:"):
                rating = int(tok[7:] or 0)
            elif tok == "fav":
                fav = True
            elif tok == "recent":
                recent = True
            else:
                words.append(tok)
        return self.search(" ".join(words), bpm, key, tags, rating, fav, recent, limit)

    def get(self, ident) -> Optional[Row]:
        cur = self.db.execute("SELECT %s FROM samples WHERE %s" % (self.COLS, "id=?" if isinstance(ident, int) else "path=?"), (ident,))
        rows = self._rows(cur)
        return rows[0] if rows else None

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0])

    # ------------------------------------------------------------------ curation
    def _tags(self, ident) -> List[str]:
        r = self.get(ident)
        return r.tags if r else []

    def add_tag(self, ident, tag: str) -> None:
        t = sorted(set(self._tags(ident)) | {tag.lower().strip()})
        self._set(ident, "tags", ",".join(t))

    def remove_tag(self, ident, tag: str) -> None:
        t = sorted(set(self._tags(ident)) - {tag.lower().strip()})
        self._set(ident, "tags", ",".join(t))

    def set_rating(self, ident, rating: int) -> None:
        self._set(ident, "rating", int(np.clip(rating, 0, 5)))

    def set_favorite(self, ident, fav: bool = True) -> None:
        self._set(ident, "fav", 1 if fav else 0)

    def touch(self, ident) -> None:
        self._set(ident, "last_used", time.time())

    def _set(self, ident, col: str, val) -> None:
        self.db.execute("UPDATE samples SET %s=? WHERE %s" % (col, "id=?" if isinstance(ident, int) else "path=?"), (val, ident))
        self.db.commit()

    def remove_missing(self) -> int:
        gone = [r[0] for r in self.db.execute("SELECT path FROM samples") if not os.path.exists(r[0])]
        for p in gone:
            self.db.execute("DELETE FROM samples WHERE path=?", (p,))
        self.db.commit()
        return len(gone)

    # ------------------------------------------------------------------ similarity and duplicates
    def _matrix(self):
        rows = self.db.execute("SELECT id, emb FROM samples WHERE emb IS NOT NULL").fetchall()
        ids = [r[0] for r in rows]
        mat = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 60), dtype=np.float32)
        return ids, mat

    def similar(self, target, k: int = 10, exclude_self: bool = True) -> List[Row]:
        """Most similar samples to a library id/path, or to an in-memory array (x, sr) tuple."""
        if isinstance(target, tuple):
            vec = analysis.embedding(target[0], target[1])
            self_id = None
        else:
            r = self.get(target)
            if r is None:
                return []
            vec = np.frombuffer(self.db.execute("SELECT emb FROM samples WHERE id=?", (r.id,)).fetchone()[0], dtype=np.float32)
            self_id = r.id
        ids, mat = self._matrix()
        if not len(ids):
            return []
        d = np.linalg.norm(mat - vec, axis=1) / (0.5 * (np.linalg.norm(mat, axis=1) + float(np.linalg.norm(vec))) + 1e-6)
        sim = 1.0 / (1.0 + d * d)
        order = np.argsort(-sim)
        out = []
        for i in order:
            if exclude_self and ids[i] == self_id:
                continue
            row = self.get(int(ids[i]))
            if row:
                row.score = float(sim[i])
                out.append(row)
            if len(out) >= k:
                break
        return out

    def duplicates(self, near: bool = False, threshold: float = 0.985) -> List[List[Row]]:
        """Groups of identical files (same hash); with near=True also groups of near-identical audio."""
        groups: List[List[Row]] = []
        for (h,) in self.db.execute("SELECT filehash FROM samples GROUP BY filehash HAVING COUNT(*) > 1").fetchall():
            groups.append(self._rows(self.db.execute("SELECT %s FROM samples WHERE filehash=?" % self.COLS, (h,))))
        if near:
            ids, mat = self._matrix()
            seen = {r.id for g in groups for r in g}
            for i in range(len(ids)):
                if ids[i] in seen:
                    continue
                d = np.linalg.norm(mat - mat[i], axis=1) / (0.5 * (np.linalg.norm(mat, axis=1) + np.linalg.norm(mat[i])) + 1e-6)
                sim = 1.0 / (1.0 + d * d)
                grp = [j for j in range(len(ids)) if j != i and sim[j] >= threshold and ids[j] not in seen]
                if grp:
                    rows = [self.get(int(ids[j])) for j in [i] + grp]
                    rows = [r for r in rows if r]
                    if len(rows) > 1:
                        groups.append(rows)
                        seen.update(r.id for r in rows)
        return groups

    # ------------------------------------------------------------------ packs
    def import_pack(self, source: str, dest: Optional[str] = None, progress: Optional[Callable[[str], None]] = None) -> Tuple[str, int]:
        """Import a folder or a zip of samples into the library. Zips are extracted safely (no path traversal,
        size and file-count limits). Folder and file names become tags. Returns (destination, files added)."""
        source = os.path.expanduser(source)
        pack = os.path.splitext(os.path.basename(source.rstrip("/\\")))[0] or "pack"
        if os.path.isdir(source):
            folder = source
        elif zipfile.is_zipfile(source):
            folder = dest or os.path.join(data_dir(), "packs", re.sub(r"[^A-Za-z0-9_.-]", "_", pack))
            safe_extract(source, folder)
        else:
            raise ValueError("pack must be a folder or a .zip file")
        added, _ = self.scan([folder], progress, extra_tags=[re.sub(r"[^a-z0-9]+", "-", pack.lower()).strip("-")])
        return folder, added


MAX_PACK_BYTES = 4 * 1024 ** 3
MAX_PACK_FILES = 20000


def safe_extract(zip_path: str, dest: str) -> int:
    """Extract only regular audio/text files, refuse absolute paths and '..' escapes, cap total size and count."""
    os.makedirs(dest, exist_ok=True)
    root = os.path.realpath(dest)
    n = total = 0
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        if len(infos) > MAX_PACK_FILES:
            raise ValueError("pack has too many files (%d)" % len(infos))
        for info in infos:
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ".." in name.split("/"):
                raise ValueError("unsafe path in pack: %r" % info.filename)
            if not name.lower().endswith(AUDIO_EXT + (".txt", ".md", ".json")):
                continue
            total += info.file_size
            if total > MAX_PACK_BYTES:
                raise ValueError("pack is larger than the %d GB limit" % (MAX_PACK_BYTES // 1024 ** 3))
            target = os.path.realpath(os.path.join(root, name))
            if not (target == root or target.startswith(root + os.sep)):
                raise ValueError("unsafe path in pack: %r" % info.filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    return n
