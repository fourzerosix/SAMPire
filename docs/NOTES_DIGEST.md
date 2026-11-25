# What came from the second rough notes

The second notes describe a native C++/JUCE instrument (audio engine, plugin formats, a GUI shell, a long hardware-and-DSP specification).
SAMPire is a Python terminal instrument, so the notes were mined for ideas that carry over rather than followed as a build plan.

## Taken into SAMPire 0.2

| Idea in the notes | Where it landed |
|---|---|
| MIDI framework: default note map (36-51 bank A, 52-67 bank B, ...), CC defaults (1 mod, 7 volume, 10 pan, 16-31 assignable), pitch bend, program change selects banks, aftertouch | `midi.py`: same defaults; pad notes are 36 + pad index, so bank B is 52-67 |
| MIDI learn, global and project level maps | `MidiMap` with `global` and `project` scopes; `:midi learn TARGET [project]` |
| Hot-plug and disconnect safety | `MidiManager` rescans every 2 s and survives unplugging (tested with fake ports) |
| MIDI clock in and out, start/stop/continue, song position | `ClockFollower`, engine clock-out pulses |
| One set of controllable parameters for UI and hardware | `params.py` registry |
| Workspaces (performance, sample editor, sequencer, mixer, creative, browser) and a diagnostics panel | seven curses workspaces, F1-F7 |
| Creative workspace with accept / reject and previewing | Create workspace: generate, keep, reject (reject undoes) |
| Voice manager: stealing policies, protected voices | `steal oldest|quietest`, `:protect` |
| Documented signal flow and tap points (what each recorder hears) | top of `engine.py` and `ARCHITECTURE.md` |
| Bus/group routing with sidechain | 4 buses, per-bus effect chains, sidechain source |
| Effect list (compressor, EQ, chorus, flanger, phaser, tape, vinyl, ring mod) | `fx.py`, all block-size exact |
| Effect interface with prepare/reset/parameter changes | `FX` base: `DEFAULTS`, `set()`, `spec()`, state kept across calls |
| Sample library: tags, ratings, BPM/key filters, duplicates, similarity, pack import | `library.py` |
| Loudness-targeted export, true peak | `--lufs`, gated BS.1770, 4x oversampled peak |
| Reliability: autosave, crash recovery, corrupt project handling, soak tests | `Project.recover`, `ProjectError`, `--soak` |
| Test discipline: determinism, block-size invariance, long-running stability | selftest checks 9, 10, 17, 27 |

## Left for later (and why)

- **C++/JUCE engine, VST3/AU/CLAP plugin builds, standalone desktop GUI.** A separate project with a separate toolchain. The Python engine's command queue and
  project format are the seam if you ever want a native audio core.
- **Plugin hosting with crash isolation.** Optional Pedalboard hosting exists; isolation needs a host process.
- **Ableton Link.** Needs a native binding; MIDI clock covers sync for now.
- **Hardware controller profiles** (per-device LED feedback, pad colours). Needs the devices to test; the generic note and CC map works with class-compliant controllers.
- **Licence review for a native build.** Not relevant to the Python package (GPL-3.0-only).
