# SAMPire design overview

This is the high-level plan: what SAMPire is, what it needs, every feature area, the components that deliver them,
what is already built in v0.1, and what is deliberately left for later. It is based on the original rough notes
(feature ideas, SP-404 / MPC research, architecture sketches) plus a verification pass on the claims that mattered.

## 1. What SAMPire is

A sampler you can play and a sampler that plays itself. It takes audio in (files, a live input, its own output),
finds the interesting parts, edits them autonomously, assigns them to pads, sequences them, and records what comes
out. The hardware references are the Roland SP-404 line and the Akai MPC line; the software reference for look and
feel is DanEQ (btop-style braille terminal UI, `--selftest`, `--demo`, install script, Docker, CI).

Design principles, in priority order:

1. **Audio first.** The audio path never blocks on the interface, the disk, or the creative engine.
2. **Sample accurate.** Timing is computed from an integer frame counter, never from wall-clock or block size.
3. **Nondestructive.** Sources are never modified. Edits are recipes applied to a region; results are derived data.
4. **Deterministic.** Every random decision is seeded. Same seed, same project, same audio, bit for bit.
5. **Explainable automation.** The creative engine produces ordinary project edits (pad assignments, recipes, patterns)
   that you can inspect, undo, and keep editing by hand.
6. **Works anywhere.** No audio device? It runs silent. No GPU, no network, no cloud. Headless rendering for servers and CI.

## 2. Corrections and decisions versus the rough notes

| Rough-notes idea | Decision | Why |
|---|---|---|
| Desktop GUI (PySide6) as the main UI | Terminal UI first (curses, braille waveforms) | Matches DanEQ, works over SSH/ThinLinc, zero GUI dependencies. The engine is UI-agnostic, so a GUI stays possible. |
| `aubio` for onset/tempo | Pure NumPy/SciPy by default; `aubio`/`librosa` as optional backends | The original `aubio` package has had no PyPI release since 0.4.9 (2019) and often has no wheel for current Python; the `aubio-ledfx` fork does ship wheels. A hard dependency on either is an install risk. |
| `python-rtmixer` for the audio path | `sounddevice` callback with NumPy block mixing | `rtmixer` is a thin PortAudio mixer with old wheels; it cannot host per-voice DSP. A sampler needs envelopes, choke groups and sends inside the callback. |
| Separate native audio engine (C++/Rust) | Python engine now, process boundary designed in | NumPy block mixing uses about 11 percent of real time for 32 voices at block 256 (measured by `--selftest`). The command-queue boundary lets the engine move to its own process or language later without touching the UI. |
| Rubber Band for stretch | Built-in WSOLA, optional later | No native dependency for v0.1; quality is good for loops and bass, softer on dense drum transients (documented). |
| Neural stem separation (Demucs) in core | Optional plugin, roadmap | Heavy, GPU-hungry, and upstream maintenance has been uneven. Keep it out of the core install. |
| Vector database for similarity search | In-memory feature vectors now, SQLite index on disk | Fewer than tens of thousands of samples per project does not need a vector DB. |

## 3. Prerequisites

**Runtime**

- Python 3.9 or newer (CI covers 3.9, 3.10, 3.12, 3.13).
- NumPy and SciPy (DSP), SoundFile (libsndfile: wav, flac, aiff, ogg; mp3 with libsndfile 1.1+).
- Optional: `sounddevice` + PortAudio for sound (without it SAMPire runs silent), `mido` + `python-rtmidi` for MIDI
  controllers, `soxr` for the highest quality sample-rate conversion.
- A terminal at least 80x24 with UTF-8 (braille glyphs) and 256 colours; 120x40 is comfortable.

**System packages**

- Debian/Ubuntu: `libportaudio2 libsndfile1 libasound2-dev`
- RHEL/Rocky 9: `portaudio` (EPEL), `libsndfile`, `alsa-lib-devel` (for `python-rtmidi`)
- macOS: `brew install portaudio libsndfile`

**For development**

- `pytest`, `pyte` (drives the real curses UI in a pseudo-terminal), `build`, `twine`.
- Knowledge worth having: block-based audio processing, why allocation and locks are avoided in audio callbacks,
  PPQ-based sequencing, and the basics of onset detection (spectral flux) and WSOLA.

**Hardware (optional)**

- Audio interface for low latency and external input sampling; any class-compliant MIDI pad controller.

## 4. Feature catalogue

Status: **done** = in v0.1 and covered by `--selftest` or pytest; **partial** = works with limits; **planned** = see roadmap.

### 4.1 Input and library

| Feature | Status |
|---|---|
| Load wav/flac/aiff/ogg (mp3 where libsndfile supports it), any sample rate, mono or stereo | done |
| Everything converted once to 48 kHz stereo float32; originals never touched | done |
| Content hash per asset, relink moved files by hash on project load | done |
| Built-in synthetic drum break (`--demo`) for first run, tests and CI | done |
| Live external input, threshold recording with pre-roll | done (engine and `:sample`; needs an input device) |
| Rolling skip-back buffer of the master output (60 s) | done |
| Resample: record the master bus into a new asset with lineage (generation counter) | done |
| Folder browser, tags, ratings, similarity search | planned |
| Drag-and-drop import in a GUI | planned |

### 4.2 Analysis ("sample intelligence")

| Feature | Status |
|---|---|
| Peak, RMS, BS.1770 K-weighted loudness (ungated, 48 kHz) | done |
| Onset detection: multi-band SuperFlux-style flux, steepest-rise refinement to about 2 ms | done |
| Tempo (autocorrelation with prior), beat phase, downbeat guess from low-band energy | done |
| Key estimate (chroma with harmonic filtering, Krumhansl profiles) | done (reliable on tonal material; drums-only is "unknown") |
| Monophonic pitch (autocorrelation) | done |
| Percussive vs harmonic ratio (median-filter HPSS) | done |
| Hit classification: kick, snare, hat, open hat, tom, bass, tone, perc | partial (rule based; clap lands in snare; a pluggable `Backend` hook exists for an ML model) |
| Stem separation (drums/bass/vocals/other) | planned (optional plugin) |
| Embedding-based "find similar" | planned |

### 4.3 Slicing and chopping

| Feature | Status |
|---|---|
| Transient, grid (musical subdivision), equal parts, silence, hybrid (transient nudged to the grid, gap fill) | done |
| Zero-crossing snap, pre-roll, minimum/maximum length, tail length | done |
| Slice ranking (strength, variety), automatic pad layout (kick, snare, hats first) | done |
| Manual slice editing on a waveform (add, move, delete markers) | planned (TUI marker editor) |
| Loop-point detection and seamless-loop search | planned |

### 4.4 Autonomous editing (recipes)

Nineteen transforms, each deterministic for a seed: reverse, gain, normalize, fade, pitch (duration preserved),
stretch (pitch preserved), rate (varispeed), stutter, truncate, loop, filter (lp/hp/bp/notch/shelf), drive, bitcrush,
rhythmic gate, tape stop, granular scatter, vinyl crackle, mono, shuffle (re-order sub-slices).
Recipes chain, serialize to JSON, and re-render on demand. Status: **done**.

### 4.5 Pads and playback

| Feature | Status |
|---|---|
| 4 banks x 16 pads, global pad index, bank switching | done |
| Modes: one-shot, gate, loop, toggle; reverse flag | done |
| Velocity sensitivity, gain, pan (constant power), pitch + fine tune (varispeed), ADSR | done |
| Choke / mute groups (closed vs open hat), 4 ms click-free cuts | done |
| 32-voice polyphony with oldest-voice stealing | done |
| Round-robin and random sample variants per pad | done (engine and data model; no UI yet) |
| Per-pad send levels to delay and reverb buses | done |
| Baked per-pad low/high-pass | done |
| Chromatic keyboard mode, per-pad realtime filter/LFO | planned |

### 4.6 Sequencer

| Feature | Status |
|---|---|
| 960 PPQ integer ticks, patterns of any bar count, step grid (16ths) | done |
| Sample-accurate scheduling independent of block size, zero drift | done (verified for block 32 to 4096) |
| Swing (global or per pattern), per-pad microtiming, humanize (time and velocity, seeded) | done |
| Probability, ratchets, conditional triggers (`1:2`), per-hit parameter locks | done (data model and engine; edit via commands) |
| Note length for gate/loop pads | done |
| Live pad recording into the pattern with quantize and latency compensation | done |
| Euclidean generator, quantize, shift, reverse, extend | done |
| Song mode (chain patterns with repeats) | done (engine/data; simple chain editing is planned) |
| MIDI clock in/out, Ableton Link | planned |

### 4.7 Effects

| Feature | Status |
|---|---|
| Delay send (tempo-syncable, filtered feedback), reverb send (Schroeder room) | done |
| Master chain: filter, drive, bitcrusher; soft limiter (never exceeds full scale) | done |
| All effects exact for any block size (chunked feedback) | done |
| Compressor, EQ, flanger/phaser/chorus, vinyl sim, tape wobble | planned (Pedalboard integration as an optional backend) |
| Per-bus routing graph (groups, sidechain) | planned |

### 4.8 Recording and export

| Feature | Status |
|---|---|
| Record the master output to wav (in memory while recording, written on stop) | done |
| Offline bounce of a pattern or song, deterministic | done |
| Stems (one wav per pad, aligned at time zero) | done |
| Pad wavs, raw slices, MIDI file of a pattern (own writer, no dependency) | done |
| 16, 24, 32-bit float; wav and flac | done |
| Per-bus stems with effects, loudness-targeted export | planned |

### 4.9 Creative engine

| Feature | Status |
|---|---|
| Kit generation: analyse, slice, choose varied slices, label, transform some, assign, set modes and choke groups | done |
| Controls: pad count, slicing mode, sensitivity, experimental amount, keep-original ratio, percussive/melodic preference, length preference, seed | done |
| Pattern generation: break, four on the floor, sparse, euclidean, chops; density, consistency across bars, swing | done |
| Pattern variation (drop, shift, ratchet, probability, ghost notes) and chop-flip (play the chops in a new order) | done |
| Plain-English instructions, parsed offline into a plan (`:prompt`, `--prompt`) | done (rule based) |
| Pluggable plan provider so an external language model can emit the same plan schema | planned |
| Learned taste (what you keep vs discard) | planned |

### 4.10 Project, history, interface

| Feature | Status |
|---|---|
| Project folder: human readable `project.json`, SQLite index, audio copies | done |
| 100-step undo/redo (snapshots of everything except audio), rolling autosave | done |
| btop-style TUI: 4x4 pad grid with flash, braille waveform with slice markers and playhead, step grid with playhead, meters, command line, 5 themes, mouse pads | done |
| MIDI notes to pads, CC mapping with learn, MIDI start/stop | done (needs `sampire[midi]`; untested on hardware in CI) |
| Headless CLI: `--analyze`, `--chop`, `--render` (whole pipeline), `--selftest` | done |
| Piano-roll, marker editor, FX panel | planned |
| GUI frontend over the same engine | planned |

## 5. Components

```
                 +--------------------------- UI thread ---------------------------+
 keyboard/mouse  |  ui/tui.py (App)        creative.py       project.py  (model)   |
 MIDI thread --> |  commands, drawing      kits, patterns    banks, pads, patterns |
                 +--------------------------+--------------------------------------+
                                            | command queue (no locks in audio path)
                 +--------------------------v--------------------------------------+
                 |  engine.py  render_block(n)   <-- called by backend.py callback  |
                 |   sequencer.Scheduler -> events at exact frames                  |
                 |   voice.py  x32 -> dry / send A / send B -> fx.py -> limiter     |
                 |   taps: skip-back ring, master recorder, resample recorder       |
                 +--------------------------+--------------------------------------+
                                            |
                          sounddevice (PortAudio)  or  silent backend  or  offline render
```

Offline modules used by both the UI and the headless CLI: `asset.py` (I/O), `analysis.py`, `slicer.py`, `recipe.py` and
`dsp.py` (edits), `export.py`. See `ARCHITECTURE.md` for threading rules and the data model.

## 6. Quality bar (what `--selftest` proves)

Sixteen checks, all headless and deterministic: file I/O and resampling, onset F1 and timing error, tempo across
80 to 140 BPM, pitch/key/class, slice cuts, stretch and pitch accuracy, voice stealing/choke/varispeed, sequencer
timing identical for block sizes 32 to 4096, full-engine render identical for block 64/256/1000/4096 with effects,
determinism per seed, bit-exact recording taps, project round trip with undo and relink, creative pipeline, exports,
limiter ceiling, and CPU headroom for 32 voices.

## 7. Risks and honest limits

- **Python and the GIL.** The UI redraw can compete with the audio callback. Mitigation today: lean drawing, cached
  waveforms, 2 percent CPU measured while playing. If you hear dropouts at small block sizes, raise `--block` to 512.
  The planned fix is moving the engine into its own process behind the existing command queue.
- **Not yet tested on real hardware, macOS, or Rocky 9.** The sandbox this was built in has no audio device. The silent
  backend, offline render, and pty-driven UI are tested; PortAudio output, input sampling and MIDI are written to the
  documented APIs but unverified on your machine. `--selftest` and `--list-devices` are the first things to run.
- **Classification is heuristic.** It is tuned on synthetic drums and will mislabel real material sometimes. Labels only
  drive pad order and choke defaults; you can rename and reassign freely.
- **Time stretch is WSOLA.** Excellent on tones and loops, softer on dense transients. Short one-shots fall back to
  varispeed, like a classic sampler.
- **Licensing.** Sampling other people's records for release has legal consequences. The tool does not check.

## 8. Where to go next

See `ROADMAP.md`. The shortest path to a much better instrument: waveform marker editor, per-pad realtime filter and
LFO, the engine in its own process, and a Pedalboard-backed effects chain.
