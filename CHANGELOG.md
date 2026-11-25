# Changelog

## 0.2.0

Everything the 0.1 design document listed as planned, except Ableton Link, a GUI and an out-of-process engine.

- Seven workspaces: Perform, Edit, Sequence, Mixer, Create, Library, Diagnostics (F1-F7).
- Library: folder scan, auto tags, ratings, favourites, BPM/key/tag search, similarity search, duplicate detection, safe sample-pack import, terminal drag and drop.
- Stems (built-in DSP split, optional Demucs), loop-point finder with seamless crossfade, manual marker editor, chops-like-this.
- Pads: realtime filter and LFO, chromatic mode, round robin variants, note repeat, voice stealing policies, protected pads, pitch bend.
- Effects: compressor (sidechain), EQ, chorus, flanger, phaser, tape wobble, vinyl, ring mod; four mixer buses with effect chains; effects saved in projects; optional Pedalboard backend.
- Sequencer: velocity lane and roll view, per-step edits, song chain editing, MIDI clock in and out.
- MIDI: default maps, learn (global and project), aftertouch, pitch bend, program change, hot-plug.
- Creative: pluggable and validated plan providers, learned taste with keep/reject.
- Export: per-bus stems with effects, loudness-targeted bounce with true-peak ceiling.
- Reliability: timestamped autosaves, `:recover`, friendly project errors, `--soak`, `--diagnostics`, 27 selftest checks, 39 pytest tests.
- Optional backends verified: Rubber Band, aubio, Pedalboard.
- Fixed: `:mode` command collided with an internal flag (caught by the new UI test).
- Breaking: `:stems` now separates stems (per-pad export is `:stemfiles`); `:midi PATH` is now `:midi-file PATH`.

## 0.1.0

First release.

- Real-time engine: 32 voices, ADSR, choke groups, velocity, pan, varispeed, delay and reverb sends, master chain, soft limiter.
- Sample-accurate sequencer at 960 PPQ with swing, humanize, probability, ratchets, parameter locks, euclid, song chains.
- Analysis: onsets, tempo, beats, key, pitch, loudness, percussive ratio, hit classification.
- Slicing: transient, grid, equal, silence, hybrid with zero-crossing snap.
- Nineteen nondestructive edit ops (recipes).
- Creative engine: kit generation, pattern generation and variation, chop-flip, offline plain-English instructions.
- Capture: master recorder, 60 s skip-back, resample, threshold input recorder.
- Exports: bounce, stems, pad wavs, slices, MIDI.
- Project folders with SQLite index, undo/redo, autosave, relink by hash.
- btop-style curses interface, 5 themes, mouse pads, MIDI input with learn.
- Headless CLI (`--analyze`, `--chop`, `--render`), `--selftest` (16 checks), pytest suite, CI, installer, Dockerfile.
