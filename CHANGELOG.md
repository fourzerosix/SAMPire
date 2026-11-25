# Changelog

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
