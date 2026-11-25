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
