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
