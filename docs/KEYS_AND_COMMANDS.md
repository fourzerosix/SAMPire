# Keys and commands

## Keys

| Key | Action |
|---|---|
| `1 2 3 4 / q w e r / a s d f / z x c v` | play pads 1 to 16 of the current bank (shift = softer) |
| arrows | move pad selection, or the cursor in the step grid |
| `Tab` | focus: pads or step grid |
| `Enter` | trigger the pad, or toggle the step under the cursor |
| `Backspace` | remove the step under the cursor |
| `Space` | play / stop |
| `+` `-` | tempo |
| `[` `]` | swing |
| `<` `>` | previous / next bank |
| `,` `.` | previous / next pattern |
| `^G` | generate a kit from the source |
| `^P` | generate a pattern |
| `^V` | vary the current pattern |
| `^F` | chop-flip: chops in a new order |
| `^X` | clear the pattern |
| `^T` | start / stop recording the master output (saved in `recordings/`) |
| `^B` | skip-back: last 30 s of output to a free pad |
| `^Y` | start / stop resampling the master bus into a free pad |
| `^U` `^R` | undo / redo |
| `^W` `^O` | save / open |
| `^L` | next theme |
| `:` | command line |
| `?` | help overlay |
| `^C` or `^Q` | quit |

## Commands (`:`)

Load and make: `open PATH`, `demo`, `chop MODE [N]`, `kit [N]`, `pattern [STYLE]`, `vary [AMOUNT]`, `flip [STRATEGY]`, `prompt TEXT`, `sample [SECONDS]`.

Transport: `bpm N`, `swing PERCENT`, `seed N`.

Edit the selected pad: `rev`, `pitch ST`, `stretch RATIO`, `rate RATIO`, `crush [BITS [DOWNSAMPLE]]`, `drive [AMOUNT]`, `lp HZ`, `hp HZ`,
`trunc MS`, `gain DB`, `norm`, `stutter [MS [REPEATS]]`, `mode one_shot|gate|loop|toggle`, `choke N`, `send a|b LEVEL`,
`pan -1..1`, `att SECONDS`, `rel SECONDS`, `name TEXT`, `clear`, `undo`, `redo`.

Effects: `fx delay LEVEL`, `fx reverb LEVEL`, `fx filter|drive|crush` (adds to the master chain), `fx clear`.

Files: `save [DIR]`, `load DIR`, `wav PATH`, `stems DIR`, `pads DIR`, `midi PATH`, `theme NAME`, `quit`.


## Workspaces (v0.2)

Switch with `F1`-`F7`, `^N` (next) or `:view NAME`. Pad keys (`1234 qwer asdf zxcv`) work everywhere. Each workspace uses only letters that are not pad keys.

| View | Keys |
|---|---|
| Perform (F1) | as above |
| Edit (F2) | arrows move the cursor (up/down are coarse), `m` add marker, `k` delete nearest, `h`/`l` previous/next marker, `i`/`u` zoom in/out, `p` audition the region, `o` put regions on pads in time order, `g` automatic markers |
| Sequence (F3) | arrows move, `Enter` toggle step, `Backspace` delete, `u`/`i` velocity down/up, `p` probability (100/75/50/25), `k` ratchet (x1-x4), `l` length, `t` toggle step view and velocity-glyph roll |
| Mixer (F4) | up/down select row, left/right level, `m` mute bus, `c` clear bus effects |
| Create (F5) | up/down parameter, left/right change, `g` kit, `p` pattern, `y` keep, `n` reject (undoes and teaches), `t` taste on/off |
| Library (F6) | up/down, `Enter` preview, `o` open as source, `a` assign to the selected pad, `/` search |
| Diagnostics (F7) | read only |

Drop a file (or folder, or zip pack) on the terminal to load it. Your terminal pastes the path; SAMPire reads it as a bracketed paste.

## More commands (v0.2)

Note: `:stems` now separates the source into stems; per-pad export is `:stemfiles DIR`; MIDI file export is `:midi-file PATH`; `:midi` controls MIDI devices.

- Editing: `mk add [SEC] | del | clear | auto | snap | apply`, `loop find | use N`, `stems [dsp|demucs]`, `like [N]`
- Pads: `pfilter lp|hp|bp HZ [Q]` / `pfilter off`, `lfo pitch|cutoff|amp|pan RATE DEPTH [shape]` / `lfo off`, `padbus 1-4`, `protect on|off`, `chromatic on|off`, `roll [DIV] [COUNT]`, `variant add|clear`, `variation none|round_robin|random`, `steal oldest|quietest`
- Song: `song add NAME [REPS] | del N | move A B | clear | on | off | list`
- Effects: `fx list`; `fx master add TYPE k=v ...`; `fx bus1 add compressor threshold_db=-20`; `fx bus2 sidechain 1`; `fx bus1 gain 0.8 | mute | name TEXT | set N k=v | rm N | clear`; `fx send_a delay time_s=0.25`; `fx send_b reverb room=0.9`; `fx return a 0.5`
- Creative: `prompt TEXT`, `provider rules|command:PROGRAM|python:mod:fn`, `keep`, `reject`, `taste [reset|on|off]`
- Library: `lib scan DIR... | info | clean | dups [near]`, `find QUERY`, `tag NAME` / `tag rm NAME`, `rate N`, `fav`, `similar [K]`, `pack PATH`
- MIDI: `midi ports | learn TARGET [project] | follow on|off | out on|off [NAME] | map`
- Files: `save`, `load DIR`, `recover [DIR]`, `wav PATH`, `lufs TARGET PATH`, `stemfiles DIR`, `busstems DIR`, `pads DIR`, `midi-file PATH`
- Other: `view NAME`, `diag`, `backend stretch rubberband|wsola`, `backend onsets aubio|builtin`, `monitor on|off`, `sample [SEC]`

## MIDI defaults

Pad notes are `36 + pad index` (bank A 36-51, B 52-67, C 68-83, D 84-99). CC7 master, CC10 pan of the selected pad, CC1 LFO depth of the selected pad,
CC16-31 the selected pad's gain, pan, pitch, fine, attack, decay, sustain, release, send A, send B, filter cutoff, filter Q, LFO rate, LFO depth, velocity sensitivity.
Pitch bend bends all playing voices (+/- 2 semitones). Program change selects a bank. `:midi learn bus1.gain` then move a control to bind it; targets are listed by `sampire.params.targets()`.
With `:chromatic on`, all MIDI notes play the chromatic pad (C4 = original pitch).
