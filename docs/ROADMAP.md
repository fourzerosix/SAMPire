# Roadmap

Legend: [x] shipped in 0.1.0, [ ] planned.

## 0.1 foundation
- [x] Engine, voices, effects, sequencer, recorder taps, exports, project format
- [x] Analysis, slicing, recipes, creative engine, offline instructions
- [x] Terminal UI, headless CLI, selftest, CI, installer, Docker
- [ ] Verify audio output, input sampling and MIDI on Linux, macOS and Rocky 9 hardware

## 0.2 instrument feel
- [ ] Waveform marker editor (add, move, delete slice points; audition)
- [ ] Per-pad realtime filter and LFO, chromatic mode
- [ ] Note repeat and roll with live record
- [ ] Piano-roll and velocity lane in the step grid
- [ ] FX panel with macro knobs; compressor and EQ

## 0.3 performance and robustness
- [ ] Engine in its own process behind the command queue (removes GIL contention)
- [ ] MIDI clock in and out
- [ ] Optional Pedalboard backend for studio-grade effects
- [ ] Optional Rubber Band and aubio backends

## 0.4 intelligence
- [ ] Pluggable plan provider (bring your own model), same plan schema
- [ ] Embedding based similarity ("more like this pad")
- [ ] Optional stem separation plugin
- [ ] Learn from keeps and discards

## Later
- [ ] GUI frontend over the same engine
- [ ] Plugin hosting, Ableton Link, sample pack import
