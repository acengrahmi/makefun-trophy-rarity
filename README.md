# GTA VI Trophy Rarity Forecaster — Counts, Not Rates

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/acengrahmi/makefun-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/acengrahmi/makefun-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A forecaster for the earn rate of every Grand Theft Auto VI PlayStation 5
trophy on Exophase at a case's target, written as five quantiles per trophy
and scored by pinball loss on the logit scale.

The page's rates are not the state of the game — the counts behind them are.
Every printed rate is an integer earner count over the players tracked,
rounded to 2 decimals, and the counts sum exactly to the page's
trophies-earned header. So this entry recovers the counts, collapses each
case's history into the distinct page states between Exophase refreshes,
forecasts the two totals (players tracked, trophies earned) and each
trophy's count on a log-time launch clock with the counts forced to sum to
the forecast earned total, and only then turns counts back into rates. A
history that has seen no refresh is forecast as persistence blended with
the dilution a refresh brings: a flood of newly tracked players who hold
almost nothing but the one trophy everyone earns.

`main.py` is Python 3.13, standard library only, deterministic, and runs in
milliseconds per case. `REPORT.md` records the analysis, the validation
protocol and the local numbers.

Entered on make.fun — see the bounty: https://make.fun/bounty/gta6-trophy-rarity-forecast-01
