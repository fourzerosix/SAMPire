# Architecture

## Threads

| Thread | Does | Never does |
|---|---|---|
| Audio callback (PortAudio) or silent pacing thread | `Engine.render_block(n)` | file I/O, analysis, locks, waiting on the UI |
| UI thread (curses) | drawing, key handling, creative engine, analysis, rendering pad samples, exports | touching voices directly |
| MIDI thread (mido callback) | translating messages into engine commands | rendering or editing the project |

The UI talks to the engine through `Engine.cmds` (a `queue.SimpleQueue`): trigger, release, play, stop, bpm, pattern,
panic, swap project. Commands are applied at the top of the next block, so the audio path never waits on the interface.
Pad parameters are plain attributes read once per voice start. Rendered samples are swapped by reference, which is atomic
in CPython.

## Time

The sequencer keeps an integer frame counter and a tick anchor: `tick(frame) = anchor_tick + (frame - anchor_frame) * ticks_per_frame`.
An event fires in the block that contains `round(frame_of(event_tick))`, which is the same frame for any block size.
Tempo changes re-anchor at the current frame, so tick time stays continuous. The engine splits each block at event
boundaries and mixes the segments in between, so triggers are sample accurate.

## Voices

`Voice.render_into(out)` has three paths: unity-rate copy (two NumPy ops per block), linear-interpolated varispeed, and
looping. The ADSR envelope is a scalar in sustain and a short vector only during attack, decay and release. Starts and
ends are click-free: sample heads and tails get fades baked in when the pad is rendered; choke and steal use a 4 ms release.

## Signal flow and taps

```
voice (filter, LFO, envelope) -> bus[pad.bus] raw -> bus fx chain (sidechain key = another bus's raw signal) -> bus gain / mute --+
voice -> send A -> delay  -> * return A -------------------------------------------------------------------------------------+-> * master gain
voice -> send B -> reverb -> * return B -------------------------------------------------------------------------------------+      |
audition, input monitor ---------------------------------------------------------------------------------------------------+      v
                                                                                         master fx chain -> soft limiter -> OUT
taps: skip-back ring, master recorder, resample recorder all read OUT (post limiter). Stem export bypasses master fx and limiter.
```

Everything stateful in the rack is exact for any block size: feedback structures run in chunks no longer than their shortest delay, modulation uses an
absolute sample counter, noise is generated on a fixed grid, and per-voice filter/LFO updates run on a grid aligned to the voice's own age. The selftest
renders the same project with block sizes 64 to 4096 and compares.

## MIDI and sync

`MidiManager` (threads) -> `MidiRouter.handle(msg)` -> engine commands or `params.apply`. Clock in: `ClockFollower` estimates tempo from the median pulse
interval and nudges the sequencer anchor every beat (snap above 20 ms, otherwise a 10 percent slew). Clock out: the engine counts 24-per-quarter pulses
from the sequencer tick position and a thread sends them. Both are best effort: timing comes from Python threads, not a hardware clock.

## Data model

- `AudioAsset`: id, name, float32 (n, 2) at 48 kHz, file path, content hash, analysis cache, lineage metadata.
- `Pad`: asset id + frame region + `Recipe` + playback settings. The rendered sample is derived and never saved.
- `Recipe`: ordered ops + seed. `apply(region)` is deterministic.
- `Pattern`: bars, step grid, swing, humanize, pad offsets, list of `Event` (tick, pad, velocity, length, probability, ratchet, condition, parameter locks).
- `Project`: assets, 4 banks, patterns, song, tempo, seed, effect settings, undo/redo snapshots.

## Extension points

- `analysis.Backend`: replace hit classification (or any analysis step) with a library or model.
- `creative.parse_instruction` returns a plan dict; any provider that returns the same schema can drive `run_plan`.
- `recipe.OPS`: add a function `(x, sr, rng, params) -> array` and it is available to recipes, the creative engine and the UI.
- `fx.REGISTRY`: add an effect class with `wet(x)` and it can be used as a master insert.
- `backend.py`: any object with `start()`, `stop()`, `info()` can drive `Engine.render_block`.
