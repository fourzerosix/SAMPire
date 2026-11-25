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
| Separate native audio engine (C++/Rust) | Python engine now, process boundary designed in | NumPy block mixing uses roughly 15 to 25 percent of real time for 32 voices with sends at block 256 (measured by `--selftest`). The command-queue boundary lets the engine move to its own process or language later without touching the UI. |
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

## 4. Feature catalogue (v0.2.0)

Status: **done** = implemented and covered by `--selftest` or pytest in a headless environment; **done, hardware untested** = implemented
against a documented API but not yet exercised with a physical device; **not included** = deliberately left out (reason given).
Everything listed as planned in v0.1 is now in, except the items under "not included" in section 4.11.

### 4.1 Input and library

| Feature | Status |
|---|---|
| Load wav/flac/aiff/ogg (mp3 where libsndfile supports it), any rate, mono or stereo; converted once to 48 kHz stereo float32; originals untouched | done |
| Content hash per asset; moved files relinked by hash | done |
| Built-in synthetic drum break (`--demo`) | done |
| Live input, threshold recording with pre-roll, input monitoring | done, hardware untested |
| 60 s skip-back of the master, resample the bus into a pad with lineage | done |
| **Library**: SQLite index of folders, BPM/key/label analysis on scan, tags (auto from folder and file names), ratings, favourites, recents | done |
| **Search**: text, `bpm:85-95`, `key:C`, `tag:kick`, `rating:3`, `fav`, `recent` | done |
| **Similarity search** ("sounds like this pad / this file") from a 60-value timbre embedding | done |
| **Duplicate detection** (identical files; near-identical audio with `dups near`) | done |
| **Sample pack import** from a zip or folder, safe extraction (no path traversal, size and count caps), tags from names | done |
| **Drag and drop**: dropping files or folders on the terminal loads them (bracketed paste) | done (terminals that support bracketed paste) |

### 4.2 Analysis

| Feature | Status |
|---|---|
| Peak, RMS, ungated LUFS; **BS.1770 gated integrated LUFS and 4x true peak** | done |
| Onsets (multi-band SuperFlux style, about 2 ms), tempo, beat phase, downbeats, key, pitch, percussive ratio | done |
| Hit classification (kick, snare, hat, open hat, tom, bass, tone, perc) | done, heuristic; hook for an ML model |
| **Stem separation**: built-in DSP split (drums / bass / other, sums back to the input) | done |
| Stem separation with Demucs (neural) via `--method demucs` | adapter written, **untested** (Demucs is not installed in the test environment) |
| **Optional backends**: aubio onsets (F1 0.97 vs 0.93 built in), Rubber Band stretch/pitch | done, verified |

### 4.3 Slicing and chopping

| Feature | Status |
|---|---|
| Transient, grid, equal, silence, hybrid; zero-crossing snap; pre-roll; tail length; ranking and drum layout | done |
| **Manual marker editor** (Edit workspace): add, delete, jump, zoom, audition, put chops on pads in time order | done |
| **Loop-point detection** (whole-bar candidates scored by how well the end joins the start) and **seamless loop** crossfade | done |
| **Chops like this one** (`:like`): fill free pads with the slices most similar to a pad | done |

### 4.4 Autonomous editing (recipes)

Twenty nondestructive transforms, each deterministic for a seed: reverse, gain, normalize, fade, pitch, stretch, rate, stutter, truncate, loop,
filter, drive, bitcrush, rhythmic gate, tape stop, granular, vinyl crackle, mono, shuffle, seamless loop. **done**

### 4.5 Pads and playback

| Feature | Status |
|---|---|
| 4 banks x 16 pads; modes one-shot, gate, loop, toggle; velocity, gain, pan, pitch, fine tune, ADSR; choke groups; 32 voices | done |
| **Realtime per-pad filter (lp/hp/bp) and LFO** (pitch, cutoff, amp, pan; sine/tri/square/saw), exact for any block size | done |
| **Chromatic mode** (pad keys and MIDI notes play one pad at pitch) | done |
| **Round robin / random variants** with a `:variant` command | done |
| **Note repeat / roll** with live recording into the pattern | done |
| **Voice stealing policies** (oldest, quietest) and **protected pads** | done |
| **Pitch bend** (all playing voices), aftertouch mapping | done |
| Send levels to delay and reverb | done |

### 4.6 Sequencer

| Feature | Status |
|---|---|
| 960 PPQ; sample-accurate, block-size independent, zero drift; swing, microtiming, humanize, probability, ratchets, conditions, parameter locks, note length | done |
| **Piano-roll style grid with velocity glyphs and a velocity lane**; edit velocity, probability, ratchet, length per step | done |
| **Song chains** with repeats, reorder, song mode (`:song add/del/move/on/off`) | done |
| Live recording with quantize and latency compensation; euclid, quantize, shift, reverse, extend | done |
| **MIDI clock in** (follows tempo, start/stop/continue, song position, phase nudging) | done (simulated clock tested; hardware untested) |
| **MIDI clock out** (24 pulses per quarter, start/stop) | done (96 pulses in 2 s at 120 BPM verified; hardware untested) |

### 4.7 Effects and routing

| Feature | Status |
|---|---|
| Delay, reverb, filter, drive, crusher, **compressor, 3-band EQ, chorus, flanger, phaser, tape wobble, vinyl, ring modulator** | done, all exact for any block size |
| **Per-bus routing**: 4 group buses with effect chains, gain, mute, and **sidechain** (ducks 27x in the test) | done |
| **Mixer workspace** with meters and gain-reduction readout | done |
| **Pedalboard backend** (studio effects, VST3/AU plugin hosting) | done, built-in effects verified; hosting a real VST3 **untested**, and a plugin crash would take the process down (no isolation) |
| Effects, buses and sends saved in the project and covered by undo | done |

### 4.8 Recording and export

| Feature | Status |
|---|---|
| Master recorder, offline bounce, stems per pad, pad wavs, slices, MIDI file | done |
| **Per-bus stems with effects** (sum back to the mix; master effects and limiter bypassed) | done |
| **Loudness-targeted export** (`--lufs -14`, true-peak ceiling -1 dB, reports when the ceiling limited it) | done |

### 4.9 Creative engine

| Feature | Status |
|---|---|
| Kit generation, pattern generation (+ `auto` style), variation, chop-flip, offline instructions | done |
| **Pluggable plan provider**: `rules`, `command:PROGRAM`, `python:module:function` (opt-in); plans are validated and clamped (no paths, commands or code can pass through) | done |
| **Learned taste**: keep / reject generated kits and patterns; a small local model biases slice choice, transforms and pattern style; inspectable and resettable | done |
| Accept / reject loop in the Create workspace (reject undoes and teaches) | done |

### 4.10 Project, history, interface, reliability

| Feature | Status |
|---|---|
| Project folder (JSON + SQLite + audio), 100-step undo/redo, **timestamped autosave (keeps 5), crash recovery (`:recover`)**, friendly errors for corrupt or newer projects | done |
| **Seven workspaces**: Perform, Edit, Sequence, Mixer, Create, Library, Diagnostics (F1-F7, `^N`, `:view`) | done |
| **MIDI framework**: default pad map (36 + index), CC defaults (7 master, 10 pan, 16-31 pad parameters), learn with global and per-project maps, program change selects bank, pitch bend, aftertouch, **hot-plug and unplug survival** | done, hardware untested |
| **Diagnostics**: device state, CPU, voices, underruns, memory, backends, MIDI | done |
| **Soak test** (`--soak`), headless pipeline, `--selftest` (27 checks), 39 pytest tests incl. the real curses UI in a pty | done |

### 4.11 Not included, and why

| Item | Why |
|---|---|
| Ableton Link | Needs the native Link library and a Python binding I could not verify here. MIDI clock in and out cover sync. A `SyncSource`-style adapter is the right place for it later. |
| Graphical (Qt/web) frontend | The terminal is the frontend. The engine, project model and command layer are UI independent, so a GUI can be added, but it is a separate project. Drag and drop works through the terminal. |
| Engine in its own process | Keeping state (pad samples, patterns, effects) in sync across processes is a large source of bugs. Measured headroom is high (32 voices with sends at roughly 15 to 25 percent of real time; the 20 s soak with effects on every bus runs 3x faster than real time). Revisit only if your hardware tests show dropouts at block 256. |
| Native (C++/JUCE) engine and plugin formats from the second notes | A different project. Kept as a long-term option; the command-queue boundary is the seam. See `NOTES_DIGEST.md`. |
| Crash isolation for hosted plugins | Needs a separate plugin-host process. |

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
                 |   voice.py  x32 (filter, LFO) -> bus 1-4 (fx, sidechain, gain)  |
                 |       -> + send A / send B -> master fx -> limiter               |
                 |   taps: skip-back ring, master recorder, resample recorder       |
                 +--------------------------+--------------------------------------+
                                            |
                          sounddevice (PortAudio)  or  silent backend  or  offline render
```

Offline modules used by both the UI and the headless CLI: `asset.py` (I/O), `analysis.py`, `slicer.py`, `recipe.py` and
`dsp.py` (edits), `export.py`. See `ARCHITECTURE.md` for threading rules and the data model.

## 6. Quality bar (what `--selftest` proves)

Twenty-seven checks, all headless and deterministic (sixteen core checks plus effects, buses, voice modulation, stems/loops/embeddings, library, MIDI, providers/taste, loudness and bus stems, recovery, optional backends, soak): file I/O and resampling, onset F1 and timing error, tempo across
80 to 140 BPM, pitch/key/class, slice cuts, stretch and pitch accuracy, voice stealing/choke/varispeed, sequencer
timing identical for block sizes 32 to 4096, full-engine render identical for block 64/256/1000/4096 with effects,
determinism per seed, bit-exact recording taps, project round trip with undo and relink, creative pipeline, exports,
limiter ceiling, and CPU headroom for 32 voices.

## 7. Risks and honest limits

- **Python and the GIL.** The UI redraw can compete with the audio callback. Mitigation today: lean drawing, cached
  waveforms, 2 percent CPU measured while playing. If you hear dropouts at small block sizes, raise `--block` to 512.
  If your hardware tests show it, the fix is moving the engine into its own process behind the existing command queue (see 4.11).
- **Not yet tested on real hardware, macOS, or Rocky 9.** The sandbox this was built in has no audio device and no MIDI hardware. The silent
  backend, offline render, simulated MIDI (fake ports and a simulated clock), and the pty-driven UI are tested; PortAudio output,
  input sampling, real MIDI devices, Demucs and hosted plugins are unverified on your machine. Run `sampire --selftest`,
  `sampire --list-devices`, `sampire --midi-ports` and `sampire --diagnostics` first.
- **Classification is heuristic.** It is tuned on synthetic drums and will mislabel real material sometimes. Labels only
  drive pad order and choke defaults; you can rename and reassign freely.
- **Time stretch is WSOLA.** Excellent on tones and loops, softer on dense transients. Short one-shots fall back to
  varispeed, like a classic sampler.
- **Licensing.** Sampling other people's records for release has legal consequences. The tool does not check.

## 8. Where to go next

See `ROADMAP.md`. First: hardware testing and bug reports from real use. Then whatever those tests show matters most.
