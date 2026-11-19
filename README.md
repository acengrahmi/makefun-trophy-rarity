[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/acengrahmi/makefun-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/acengrahmi/makefun-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecasts every GTA VI trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy. Single file, `main.py`, Python 3.13 standard library only, deterministic.

## Model

Two branches, chosen by the shape of the case history:

- **Flat history (launch night).** Every read is identical (N = 638), so the next read is Exophase's first refresh (N 638 → 3,868). The new cohort has finished the opening and almost nothing else, so its earns are modelled as overdispersed counts (negative binomial, size 6) concentrated on the early story chain — Chapter 1 weight 1.0, Chapter 2 0.45, Chapter 3 0.20, Chapter 4 0.09, final mission 0.02, named missions 0.15 each, quick side activities 0.05, everything else 0.03, 100%-completion gold 0 — with 80 expected new earns in total. The forecast is the quantiles of (E + k) / 3,868, so the median trophy sits at pure dilution. Longer flat horizons compound further refreshes with decaying growth (2.0×, 1.5×, …) and the same allocation. A trophy at 100% stays at 100%.
- **Rich history (the final).** A two-flow model per trophy, dE = s·dN + U·∫N dt, fitted by weighted least squares over refresh epoch pairs and shrunk toward tier means by empirical Bayes. N(t) is projected with a saturating growth law (checkpoint log-growth rate decayed with a 2.5-day time constant), flows decay mildly over the horizon (exp(−H/12 days)), and the spread combines posterior variance, fit residuals, N uncertainty and a horizon floor (0.10 + 0.06·√days).

See `REPORT.md` for the diagnosis, the validation numbers and the honest limits.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`, 55 rows, non-decreasing quantiles in [0, 100]. Runs in well under the 8 s/case limit (~0.04 s/case on the practice cases) and is byte-identical across runs.

## License

MIT — see `LICENSE`.
