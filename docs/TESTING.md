# First test checklist

SAMPire 0.2 is verified headless (no audio device, no MIDI hardware). These steps cover what only your machines can tell us.
Paste the output of `sampire --report` into any issue you open.

## 1. Install and prove the basics

```bash
git clone https://github.com/fourzerosix/SAMPire && cd SAMPire
scripts/install.sh              # Linux / macOS
sampire --report                # versions, audio devices, MIDI ports, optional backends
sampire --selftest              # expect 27 checks (a few SKIP if optional backends are missing)
sampire --diagnostics
```

## 2. Sound

- `sampire --list-devices`, then `sampire --demo` (add `--device N` if the default is wrong). Press a pad key: you should hear a hit.
- Press `Space`: the pattern plays. Watch the header: `xr` is the underrun counter and should stay 0.
- If you hear dropouts, try `--block 512`, close other heavy programs, and note the CPU figure on the Diagnostics view (F7).
- Over ThinLinc / remote desktop: audio depends on the session's PulseAudio forwarding. Run headless (`--render`) if there is no audio path.

## 3. Your own audio

- `sampire your_loop.wav` or drop a file onto the terminal. Check tempo and key in the Create view (F5).
- F2 (Edit): `g` for automatic markers, move with the arrows, `m` / `k` to add or delete, `p` to audition, `o` to put them on pads.
- `:loop find`, `:stems`, `:like 3`.
- `^T` records the output; `^B` captures the last 30 seconds to a free pad; `^Y` starts/stops resampling.

## 4. MIDI (if you have a controller)

- `sampire --midi-ports`, then `sampire --demo`. Hit a pad: notes 36-51 are bank A. `:midi learn bus1.gain` and move a knob.
- Unplug and replug the controller while it runs: the Diagnostics view should show it disconnect and reconnect, and nothing should crash.
- Clock: `:midi follow on` with an external clock source, or `:midi out on` and watch your DAW follow SAMPire.

## 5. Library

- `:lib scan ~/Samples` (runs in the background), then F6. `/` searches, Enter previews, `:similar` finds sounds like the selected one.
- `:pack some_pack.zip` imports a sample pack.

## 6. Stability

- `sampire --soak 120` for a two-minute stress test. Leave `sampire --demo` playing for an hour and check Diagnostics for memory growth.

## What to report

Version, `--report` output, what you did, what you expected, what happened, and the last lines of the message log. If a render sounds wrong,
`sampire --render your.wav --seed N --out bad.wav` reproduces it exactly, so the seed and the input are all that is needed.
