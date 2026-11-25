# Roadmap

Legend: [x] shipped, [ ] not yet.

## 0.1 foundation
- [x] Engine, voices, effects, sequencer, recorder taps, exports, project format
- [x] Analysis, slicing, recipes, creative engine, offline instructions
- [x] Terminal UI, headless CLI, selftest, CI, installer, Docker

## 0.2 everything that was planned
- [x] Library: scan, tags, ratings, search, similarity, duplicates, safe pack import, drag and drop
- [x] Stems (DSP), loop finder, marker editor, chops-like-this
- [x] Chromatic mode, realtime per-pad filter and LFO, round robin variants, note repeat, voice stealing policies, protected pads
- [x] Compressor, EQ, chorus, flanger, phaser, tape, vinyl, ring mod; buses with sidechain; mixer workspace
- [x] Piano-roll and velocity lane, song chains
- [x] MIDI framework (learn, hot-plug, bend, aftertouch, program change), MIDI clock in and out
- [x] Plan providers, learned taste, keep/reject
- [x] Per-bus stems, loudness-targeted export, true peak
- [x] Crash recovery, soak test, diagnostics
- [x] Optional backends: Rubber Band, aubio, Pedalboard

## Next: testing on real hardware (this is the point of the GitHub release)
- [ ] Audio output and input on Linux, macOS and Rocky 9; latency and dropout report
- [ ] A real MIDI pad controller (notes, CC, learn, hot-plug) and a real MIDI clock source
- [ ] Demucs end to end; a real VST3 through Pedalboard
- [ ] Tune classification and key detection on real recordings (collect failures as test cases)

## Decide after testing
- [ ] Engine in its own process (only if dropouts show up at block 256)
- [ ] Ableton Link (needs a native binding)
- [ ] Hardware controller profiles with LED feedback
- [ ] GUI frontend
- [ ] Native engine (see NOTES_DIGEST.md)
