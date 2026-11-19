# GTA VI trophy rarity forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/acengrahmi/makefun-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/acengrahmi/makefun-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
the case target, as five quantiles (5/25/50/75/95%) per trophy, scored by
pinball loss on the logit scale. Entry for make.fun
`gta6-trophy-rarity-forecast-01`.

See the bounty page: https://make.fun/bounty/gta6-trophy-rarity-forecast-01

## How it works

Exophase refreshes its counts hours apart, so the history is a staircase:
flat stretches separated by refresh jumps where the tracked-player count
jumps and almost every rate falls (new players have earned far fewer
trophies). `main.py` models the two quantities behind every rate
`r = 100 * E / N`:

- **N(t)**, players tracked: log N fitted linear in log(1 + hours since
  release) over the distinct refresh snapshots, extrapolated to the last
  refresh expected before the target.
- **E_i(t)**, players who earned trophy i: grown by two flows — newly
  tracked players earning it (probability `s_i`) and already-tracked players
  earning it later (`U_i` per player per day), fitted per trophy by ridge
  regression over the refresh-to-refresh changes and shrunk toward the
  tier mean (platinum/gold/silver/bronze).

When the history is a single flat snapshot (the practice cases), the flows
cannot be fitted, so the program falls back to a prior for the first
refresh, centred on the launch-night facts stated in the competition rules
(N multiplied ~6x; the new cohort earned non-start trophies at ~1/15 of the
old cohort's rate). The saturated 100% trophy is forecast to stay at 100%.

Spreads combine the fit's parameter uncertainty, the uncertainty in N at
the target, and a floor that grows with the horizon. See `REPORT.md` for
the model, the validation numbers, and the honest limits.

## Run

```sh
python3 main.py
```

Reads every `*.in` in `MAKEFUN_INPUT_DIR` (default `/input`) and writes
`<id>.out` to `MAKEFUN_OUTPUT_DIR` (default `/output`). Python 3.13
standard library only, no network, deterministic (two runs write identical
bytes), ~0.1 s per case.

## Files

- `main.py` — the whole entry, single file.
- `REPORT.md` — model, validation, and limits.
- `dev/` — validation tooling (scorer, scenario generators, tuning):
  not part of the graded entry.
