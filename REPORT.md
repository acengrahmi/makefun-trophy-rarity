# REPORT — gta6-trophy-rarity-forecast-01, v4

## What the practice data actually contains (EDA)

Both practice cases (p00, checkpoint 2026-11-19T03:00Z; p01, 03:30Z) hold a
**single flat page state**: 7 and 8 reads respectively, every one identical —
638 players tracked, 943 trophies earned. No refresh, trend or trajectory is
observable inside either practice history; the only dynamics the contest
offers are the ones the rules narrate (the page then refreshed by 04:30Z to
3,868 players / 4,276 earned, every rate but the 100% one lower).

Structure found in that one state, all verified exactly:

- **Rates are integer counts in disguise.** For all 55 trophies,
  `rate == round(100 * count / 638, 2)` for an integer count (max deviation
  of `rate * 638 / 100` from an integer is 0.025, inside the 0.032 rounding
  bound), and the recovered counts sum to exactly 943 = `trophies_earned`.
  The generator's state is (players, counts); rates are a rounded view.
- **The distribution is a spike and a floor.** Trophy 2 ("Just Like
  Paradise") sits at 638/638 = 100%. Every other count is between 4 and 13:
  the platinum is 4, and the remaining 54 trophies share just 305 earners.
  Counts are *not* monotone in trophy id (13 increases along the order), so
  id order is not a progression ladder at this population size.
- **The published refresh is a dilution, not a rescaling.** Going 638 →
  3,868 players (+3,230) added 3,333 earners, of which 3,230 are the
  universal trophy if it stays at 100%: the other 54 trophies gained only
  ~103 earners in total (305 → 408, ×1.34) while the denominator grew
  ×6.06. New tracked players arrive holding almost nothing.

## The model (main.py)

Two regimes, one per shape of history:

- **Multiple states** (the final case's shape): collapse reads to distinct
  states; recover counts; fit players and earned totals by least squares on
  a log-time axis `ln(1 + hours since release)`; fit each trophy's count on
  the same axis with its slope shrunk by the number of states, re-anchored
  at the last count and held non-decreasing; rescale all counts so they sum
  to the forecast earned total (the page's own accounting identity); rates
  = counts / players at the end. Spread: a horizon term, a small-count term
  `0.55/sqrt(count)`, and a floor from the pooled observed logit-rate
  volatility between successive states.
- **One state** (the practice shape): no transition has been seen, so blend
  persistence with the refresh scenario above on the logit scale. The
  refresh weight follows staleness — (state age + horizon) against the
  ~4.7 h cadence the launch-night recording shows — and the flood's size
  (players ×~6, other counts ×~1.34 in aggregate, share-preserving) decays
  with the game's age. The universal trophy (count == players) is pinned at
  99.99 with a narrow spread in both regimes.

Deterministic, stdlib only, ~10 ms per case; output format verified
(55 rows, monotone quantiles, ≤ 65,536 bytes, byte-identical across runs).
A synthetic 3-state history smoke-tests the multi-state regime (platinum
median dilutes to 0.07, universal trophy stays 99.99).

## Validation protocol and local numbers

One fixed protocol, evaluated a handful of times, no parameter sweeps. Two
halves, because the practice data supports exactly two honest checks:

1. **Holdout split** — each practice case truncated to its first 4 reads,
   scored against the case's final read; compared with the house baseline
   (the RULES.txt persistence formula) and a last-value forecast:
   - model: **1.112 (p00), 1.136 (p01)**; house baseline: 0.0249;
     last-value: 0.000000.
   - Read this correctly: the held-out answer is the *same flat state*, so
     by construction it contains no refresh and persistence is unbeatable
     there. This half validates format and the no-refresh branch only. The
     model's refresh expectation is a deliberate bet this split cannot see —
     and both real practice targets did refresh.
2. **Refresh scenario** — the practice cases scored against the refresh the
   rules publish (players 3,868, earned 4,276; universal trophy takes all
   3,868; the other 408 earners allocated proportionally to current counts
   by largest remainder):
   - model: **0.102 (p00), 0.096 (p01)**; house baseline: 1.404;
     last-value: 1.493.
   - The model is ~14× better than the house baseline under the scenario,
     because its medians sit on the diluted rates instead of the stale
     ones. One principled iteration was taken on this evidence (refresh
     weight haircut 0.95 → 0.98: scenario 0.123/0.118 → 0.102/0.096), then
     iteration stopped.

Reference points: a previous entry of ours scored 0.113 on the server's
real practice answers; a careful local model was reported at ≈0.078 on its
own local scenario. My scenario estimate (≈0.099) sits between them.

## The gap to the 0.012 cutoff — honestly

The current top-10 cutoff (≈0.012 mean pinball) is **not reachable from the
practice inputs as given**, and this entry does not reach it either. The
arithmetic is unforgiving: pre-refresh counts are 4–13, so a ±1 error in
one trophy's post-refresh count is a ~0.08–0.22 logit error on that trophy
— several times the entire 0.012 budget, for one trophy. A 0.012 mean over
55 trophies × 5 quantiles requires predicting essentially every
post-refresh count *exactly*, i.e. knowing the generator's refresh
allocation rule itself, not estimating it. The practice histories contain a
single flat state — zero observed transitions — so that rule is not in the
data a program is given. Where the structure *is* learnable — the final
case's own history, which carries every refresh of the first two days —
this entry is built to exploit it (count recovery, state decomposition,
totals-and-shares accounting, learned volatility). On the practice board
its realistic expectation is the ~0.08–0.11 band, roughly 8× the cutoff;
closing the rest would need the generator's exact allocation, which I
could not find in the kit and will not pretend to have found.
