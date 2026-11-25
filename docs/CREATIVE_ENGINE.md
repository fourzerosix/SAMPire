# Creative engine

Level 1: algorithmic, seeded, offline. It only produces project edits and never runs in the audio callback.

## Kit generation (`creative.generate_kit`)

1. Analyse the source (cached on the asset): tempo, onsets, key.
2. Slice (`hybrid` by default: transients nudged toward the grid, gaps filled).
3. Describe each slice (centroid, band energy, flatness, decay, pitch clarity) and classify it.
4. Score slices (strength, silence rejection, length and percussive/melodic preference).
5. Pick `pads` slices by farthest-point sampling so the kit is varied, not just the loudest hits.
6. Order for a drum layout: kick, snare, hats, open hat, toms, perc, bass, tone.
7. Set sensible modes (hats share a choke group, open hat has a short release).
8. Leave `keep_original` of the pads untouched, then give a share of the rest a random recipe (`experimental` scales how many).

`KitParams`: `pads`, `mode`, `sensitivity`, `experimental`, `keep_original`, `prefer`, `length`, `bank`, `seed`, `tail_ms`, `ops`.

## Pattern generation (`creative.generate_pattern`)

Styles: `break`, `four_floor`, `sparse`, `euclid`, `chops`. Kick and snare form the backbone, hats carry accents and
ghost notes, other pads fill in. `density` controls how busy it is, `consistency` how much bar 1 repeats, `swing` the groove.
`vary_pattern` mutates a copy; `rearrange_chops` plays the chops in a new order (shuffle, reverse, every_other, stutter).

## Instructions (`creative.parse_instruction`)

Offline keyword and number parsing into a plan:

```
"chop it into 12 pieces, pitch some down, reverse a few, two 4-bar patterns at 95 bpm with swing"
-> {'kit': {'pads': 12, 'ops': ['reverse', 'pitch_down'], 'experimental': 0.7},
    'pattern': {'count': 2, 'bars': 4, 'style': 'break', 'density': 0.5}, 'bpm': 95.0, 'swing': 0.2}
```

Recognised: piece/pad counts, reverse, pitch up/down, stretch, stutter/glitch, lo-fi/crush, distort, vinyl, tape, granular,
gate, dark/muffled/bright, experimental/subtle, percussive/melodic, short/long, pattern counts and bar lengths,
four-on-the-floor, sparse, euclidean, swing, `N bpm`. Nothing leaves your machine.

## Determinism

Every random choice flows from `seed`. Pattern generation, kit generation, recipes, humanize and probability all derive
their generators from it. Run the same command twice and compare the files.


## Plan providers (`providers.py`)

`:prompt` and `--prompt` go through a provider that returns a plan dict. `rules` (default) is the offline parser above. `command:PROGRAM ARGS` runs your program,
sending `{"prompt": ..., "context": {bpm, key, duration, pads_filled}}` as JSON on stdin and reading a plan as JSON from stdout (60 s timeout, 100 kB cap).
`python:module:function` imports your function; it runs your code, so it needs `--allow-python-provider`. Whatever a provider returns is passed through
`validate_plan`: unknown keys are dropped, numbers are clamped, names must come from a whitelist. A plan cannot carry file paths, commands or code, so a language
model can steer SAMPire without being able to do anything SAMPire does not already do. Nothing is sent anywhere unless you configure a provider that sends it.

## Taste (`taste.py`)

In the Create workspace, `y` keeps what was generated and `n` rejects it (and undoes it). Each decision updates a small logistic model over readable features
(`label:kick`, `op:reverse`, `len:short`, `style:euclid`, ...). With taste on, slice scoring, transform choice and the `auto` pattern style lean toward what you kept.
Weights live in `~/.config/sampire/taste.json`; `:taste` shows the strongest ones and `:taste reset` clears them. With no examples it changes nothing, so seeds stay reproducible.
