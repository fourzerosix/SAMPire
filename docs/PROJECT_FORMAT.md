# Project format (version 1)

```
MyProject/
  project.json        human readable model: tempo, swing, seed, banks, pads, recipes, patterns, song
  project.db          SQLite: assets (id, name, path, sha1, sr, frames, meta, analysis) and a save history
  audio/sources/      imported originals (never rewritten)
  audio/recordings/   takes, skip-back captures, resamples
  autosave/           rolling snapshots of project.json
```

- Audio is stored as float32 wav so a reload is bit exact.
- `project.json` is written atomically (temp file then rename).
- A missing audio file is searched for by content hash under the project folder (`project.relink`).
- Undo and redo are in-memory JSON snapshots of the model (not audio) capped at 100 steps.
- Rendered pad samples are derived data: they are rebuilt from the source region plus the recipe on load.
