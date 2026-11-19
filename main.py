"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (5/25/50/75/95%) of
the rate in percent. Python 3.13 standard library only; no network.

The model
---------
Exophase does not update its counts continuously: it refreshes a game's page
on its own schedule, hours apart, so the history is a staircase of flat
stretches separated by refresh jumps. At a refresh the tracked-player count
N jumps (launch night: 638 -> 3,868) and every rate but the saturated one
falls, because the new players have earned far fewer trophies than the ones
already tracked. A persistence forecast therefore misses by ~1.5 logit units
whenever a refresh lands in the horizon.

The program models the two quantities behind every rate, r = 100 * E / N:

* N(t), the players tracked, fitted as log N linear in log(1 + hours since
  release) over the distinct refresh snapshots (epochs) in the history, and
  extrapolated to the last refresh expected before the target.
* E_i(t), the number of tracked players who earned trophy i, grown by two
  flows: new players joining (they earn trophy i with probability s_i) and
  players already tracked earning it later (rate U_i per player per day).
  Between epochs e-1 and e, dE_i = s_i * dN + U_i * N * dt. Both flows are
  fitted per trophy by a small ridge regression over the epoch pairs, shrunk
  toward the tier mean (platinum/gold/silver/bronze), which is itself fitted
  on the tier's pooled counts and shrunk toward the global mean. This is the
  empirical-Bayes part: a rare trophy's own history is thin, so it borrows
  strength from its tier.

When the history is a single flat snapshot (the practice cases: every read
identical, no refresh yet seen), the flows cannot be fitted from data, so the
program falls back to a prior for the first refresh, centred on the launch
night facts the competition's own rules state (the first refresh multiplied
N by ~6 and the new cohort earned non-start trophies at ~1/15 of the old
cohort's rate), with an honest spread around it. The saturated trophy (100%
of tracked players hold it) is forecast to stay at 100%: every new player
holds it too.

Spreads are honest: they combine the propagated parameter uncertainty of the
(s_i, U_i) fit, the uncertainty in N at the target (growth fit plus the
unknown timing of the last refresh before the target), and a floor that
grows with the horizon, because a 4.5-day forecast of a launch week's trophy
curve is genuinely uncertain. Quantiles are the median's logit plus the
standard normal's quantile times the spread, so the pinball loss (a proper
scoring rule) is minimised by reporting the quantiles actually believed.

I/O contract (RULES.txt): read every *.in in MAKEFUN_INPUT_DIR (else /input),
write <id>.out to MAKEFUN_OUTPUT_DIR (else /output) as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}, one row per
trophy, five finite numbers in [0, 100], non-decreasing, file <= 65,536 bytes.
The program is fully deterministic: two runs on the same case write the same
bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; put it back first
# (kept for symmetry with the starter; this entry is a single file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Standard normal quantiles at the scored levels, used when the case asks for
# exactly the levels in the rules; any other level is inverted from math.erf.
Z_LEVELS = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817,
            0.5: 0.0, 0.75: 0.6744897501960817, 0.95: 1.6448536269514722}

# --- launch-night refresh prior (flat-history branch only) -----------------
# The competition's rules state the facts of Exophase's first refresh of this
# game: N went 638 -> 3,868 (a factor of ~6.06) and trophies earned 943 ->
# 4,276. The new cohort (3,230 players) earned 3,333 trophies; the saturated
# trophy takes one per new player, leaving 103 earns over the other 54
# trophies, i.e. the new cohort earned each non-start trophy at ~0.067 of the
# old cohort's rate. Those two numbers centre the prior below; the spreads
# around them are wide, because the next refresh need not repeat the first.
PRIOR_LOG_GROWTH_MEAN = math.log(6.06)   # median N multiplier at a refresh
PRIOR_LOG_GROWTH_SD = 0.25               # 90% interval roughly [4.0x, 9.2x]
PRIOR_LOG_COHORT_MEAN = math.log(0.067)  # median new-cohort earn ratio
PRIOR_LOG_COHORT_SD = 0.35               # 90% interval roughly [0.038, 0.12]
# Per-trophy variation of the new-cohort earn ratio around the common value.
PRIOR_TROPHY_VAR_LOGIT = 0.20
# Refreshes arrive roughly every few hours on Exophase (the first came ~4.5
# hours after the first probe); used to count expected refreshes over a
# horizon when the history is flat, and for the last-refresh timing lag.
REFRESH_INTERVAL_HOURS = 5.0

# Spread floor (logit sd): a constant term plus a term growing with the
# horizon in days, so a 4.5-day forecast is honestly wider than a 90-minute
# one. Tuned on synthetic-refresh scenarios (see dev/scenarios.py), not on
# the practice answers (which are not in the kit).
SPREAD_FLOOR_BASE = 0.20
SPREAD_FLOOR_PER_SQRT_DAY = 0.15
SPREAD_MAX = 2.0
SPREAD_MIN = 0.15

# Prior scales for the two-flow regression (branch B). s is a probability
# (new players earning the trophy); U is the same over one day per existing
# player. Tier means are fitted from the data; these are only the widths the
# per-trophy fits are allowed to wander from their tier mean.
PRIOR_SD_S = 0.20
PRIOR_SD_U_DAY = 0.15


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100, not 100.0.
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p: float) -> float:
    """Standard normal quantile by bisection on math.erf (stdlib only)."""
    if p in Z_LEVELS:
        return Z_LEVELS[p]
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


# --- history: collapse reads into refresh epochs ---------------------------
# Exophase refreshes hours apart, so consecutive identical reads are one
# snapshot. An epoch is a maximal run of reads with the same players-tracked
# count and the same rates; its time is when the new values were first seen.

def extract_epochs(case: dict):
    """Return a list of epochs, oldest first.

    Each epoch is a dict with: t (datetime of first sighting), n (players
    tracked), rates (list, one per trophy, None where the read lacked it),
    earned (trophies_earned, may be None).
    """
    epochs = []
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if epochs and epochs[-1]["key"] == key:
            continue
        epochs.append({
            "key": key,
            "t": parse_time(read["at"]),
            "n": float(read["players_tracked"]),
            "rates": list(read["rates"]),
            "earned": read.get("trophies_earned"),
        })
    return epochs


def earn_counts(epoch: dict, n_trophies: int):
    """Per-trophy earn counts E_i = rate_i * N / 100 (None where unknown)."""
    out = []
    for r in epoch["rates"]:
        out.append(None if r is None else r * epoch["n"] / 100.0)
    return out


# --- two-flow regression ----------------------------------------------------
# Between epochs e-1 and e: dE_i = s_i * dN + U_i * N * dt_days.
# s_i: probability a newly tracked player has earned trophy i.
# U_i: probability an already tracked player earns it, per player per day.
# Fitted per trophy by weighted ridge regression over the epoch pairs,
# shrunk toward the tier mean; tier means come from the same regression on
# the tier's pooled counts, shrunk toward the global mean.

def _solve_2x2(a11, a12, a22, b1, b2):
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return None
    return ((b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det)


def fit_two_flow(pairs, prior, prior_sd_s, prior_sd_u):
    """Weighted ridge fit of (s, U) over epoch pairs.

    pairs: list of (dN, n_prev, dt_days, dE, var) with var the variance of dE.
    prior: (s0, U0) the shrinkage target. Returns ((s, U), (var_s, var_u,
    cov_su)) with the covariance from the ridge inverse Hessian, or
    (prior, None) when there are no usable pairs.
    """
    lam_s = 1.0 / (prior_sd_s * prior_sd_s)
    lam_u = 1.0 / (prior_sd_u * prior_sd_u)
    a11 = lam_s
    a12 = 0.0
    a22 = lam_u
    b1 = lam_s * prior[0]
    b2 = lam_u * prior[1]
    used = 0
    for d_n, n_prev, dt_days, d_e, var in pairs:
        if var <= 0:
            continue
        w = 1.0 / var
        x1 = d_n
        x2 = n_prev * dt_days
        a11 += w * x1 * x1
        a12 += w * x1 * x2
        a22 += w * x2 * x2
        b1 += w * x1 * d_e
        b2 += w * x2 * d_e
        used += 1
    if used == 0:
        return prior, None
    sol = _solve_2x2(a11, a12, a22, b1, b2)
    if sol is None:
        return prior, None
    s, u = sol
    s = min(1.0, max(0.0, s))
    u = min(1.0, max(0.0, u))
    # Covariance from the inverse of the (unclamped) ridge Hessian.
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return (s, u), None
    cov = (a22 / det, a11 / det, -a12 / det)
    return (s, u), cov


def pair_variance(d_e_expected, n_from, n_to):
    """Variance of a dE between two epochs: Poisson-ish count noise around
    the EXPECTED dE (never the observed dE -- weighting by the observed
    outcome biases the fit downward, because a pair whose noise came out
    high gets down-weighted and one whose noise came out low gets
    up-weighted), plus the +-0.005 percentage point rounding of the two
    printed rates."""
    rounding = (n_from * n_from + n_to * n_to) / (12.0 * 20000.0 * 20000.0)
    return max(d_e_expected, 1.0) + rounding


def build_pairs(epochs, counts, trophy_idx, expected):
    """Epoch pairs usable for one trophy: (dN, N_prev, dt_days, dE, var).

    expected is (s, U): the prior expectation of the flows, used only to
    size the variance of each pair's dE.
    """
    pairs = []
    for k in range(1, len(epochs)):
        e_from = counts[k - 1][trophy_idx]
        e_to = counts[k][trophy_idx]
        if e_from is None or e_to is None:
            continue
        dt_days = hours_between(epochs[k - 1]["t"], epochs[k]["t"]) / 24.0
        if dt_days <= 0:
            continue
        d_n = epochs[k]["n"] - epochs[k - 1]["n"]
        # The existing-player flow runs on the mean tracked count over the
        # interval, not the starting one: N grows inside the interval.
        n_mean = 0.5 * (epochs[k - 1]["n"] + epochs[k]["n"])
        d_e_exp = (expected[0] * d_n +
                   expected[1] * n_mean * dt_days)
        pairs.append((d_n, n_mean, dt_days, e_to - e_from,
                      pair_variance(d_e_exp, epochs[k - 1]["n"],
                                    epochs[k]["n"])))
    return pairs


def build_pooled_pairs(epochs, counts, indices, expected):
    """Epoch pairs for a pooled group of trophies (tier or all): dE summed."""
    pairs = []
    for k in range(1, len(epochs)):
        d_e = 0.0
        ok = True
        for i in indices:
            e_from = counts[k - 1][i]
            e_to = counts[k][i]
            if e_from is None or e_to is None:
                ok = False
                break
            d_e += e_to - e_from
        if not ok:
            continue
        dt_days = hours_between(epochs[k - 1]["t"], epochs[k]["t"]) / 24.0
        if dt_days <= 0:
            continue
        d_n = epochs[k]["n"] - epochs[k - 1]["n"]
        n_mean = 0.5 * (epochs[k - 1]["n"] + epochs[k]["n"])
        d_e_exp = (expected[0] * d_n +
                   expected[1] * n_mean * dt_days) * len(indices)
        var = pair_variance(d_e_exp, epochs[k - 1]["n"], epochs[k]["n"])
        pairs.append((d_n, n_mean, dt_days, d_e, var))
    return pairs


# --- players-tracked extrapolation -------------------------------------------
# log N is fitted linear in log(1 + hours since release) over the epochs and
# read at the effective target time: the target minus half the typical
# inter-refresh gap, because the answer is the rates as of Exophase's last
# refresh before the target, not the rates at the target itself.

def fit_players_growth(epochs, release):
    """OLS of log N on log(1 + hours since release). Returns (a, b, resid_sd)
    or None when there are too few epochs."""
    xs, ys = [], []
    for ep in epochs:
        if ep["n"] <= 0:
            continue
        xs.append(math.log1p(max(0.0, hours_between(release, ep["t"]))))
        ys.append(math.log(ep["n"]))
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    resid_sd = math.sqrt(sum(r * r for r in resid) / max(1, len(xs) - 2))
    return (a, b, resid_sd)


def integrate_players(a, b, t_from_h, t_to_h, steps=48):
    """Integral of the fitted N(t) = exp(a + b*log1p(t)) over [t_from_h,
    t_to_h] (hours since release), in player-days (Simpson's rule).

    The existing-player flow earns trophies at a rate proportional to how
    many players are tracked at each moment, so it runs on this integral,
    not on N_last * dt: over a 4.5-day horizon N grows several-fold, and
    using the starting N would undercount the flow by a factor of ~2-3.
    """
    if t_to_h <= t_from_h:
        return 0.0
    n = steps if steps % 2 == 0 else steps + 1
    h = (t_to_h - t_from_h) / n

    def n_at(t):
        return math.exp(a + b * math.log1p(max(0.0, t)))

    total = n_at(t_from_h) + n_at(t_to_h)
    for k in range(1, n):
        w = 4.0 if k % 2 == 1 else 2.0
        total += w * n_at(t_from_h + k * h)
    return total * h / 3.0 / 24.0


def median_gap_hours(epochs):
    gaps = [hours_between(epochs[k - 1]["t"], epochs[k]["t"])
            for k in range(1, len(epochs))]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return REFRESH_INTERVAL_HOURS
    gaps.sort()
    return gaps[len(gaps) // 2]


# --- branch B: history with at least two refresh snapshots ------------------

def forecast_with_history(case, epochs, counts, release, target, horizon_days,
                          levels, z_vals):
    n_tr = len(case["trophies"])
    last = epochs[-1]
    n_last = last["n"]
    gap = median_gap_hours(epochs)
    lag_hours = min(0.5 * gap, 12.0)
    # Effective target: last refresh before the answering read.
    target_eff = target - timedelta(hours=lag_hours)
    t_eff_hours = max(0.0, hours_between(release, target_eff))
    t_last_hours = max(0.0, hours_between(release, last["t"]))
    dt_days = max(0.0, (t_eff_hours - t_last_hours) / 24.0)

    growth = fit_players_growth(epochs, release)
    if growth is None or growth[1] <= 0:
        n_star = n_last
        sd_log_n = 0.35
        int_n_days = n_last * dt_days
    else:
        a, b, resid_sd = growth
        n_star = math.exp(a + b * math.log1p(t_eff_hours))
        # Cap the extrapolation: no more than 50x the last count, and never
        # below the last count (tracked players only grow on a launch week).
        n_star = min(max(n_star, n_last), 50.0 * n_last)
        growth_rate_per_hour = b / (1.0 + t_eff_hours)
        sd_log_n = math.sqrt(resid_sd * resid_sd +
                             (growth_rate_per_hour * lag_hours) ** 2 +
                             0.10 ** 2)
        int_n_days = integrate_players(a, b, t_last_hours, t_eff_hours)
    d_n_star = n_star - n_last

    # Tier means, then per-trophy fits, both shrunk toward the level above.
    # A pooled fit estimates the SUM of the group's flows (its dE is the
    # group's summed dE), so divide by the group size to get the mean flow
    # that a single trophy of the group is shrunk toward.
    tiers = {}
    for i, tr in enumerate(case["trophies"]):
        tiers.setdefault(tr["tier"], []).append(i)
    all_idx = list(range(n_tr))
    crude = (0.05, 0.02)
    global_pairs = build_pooled_pairs(epochs, counts, all_idx, crude)
    global_fit, _ = fit_two_flow(global_pairs, crude,
                                 PRIOR_SD_S * 2, PRIOR_SD_U_DAY * 2)
    global_fit = (global_fit[0] / len(all_idx), global_fit[1] / len(all_idx))
    tier_fit = {}
    for tier, idxs in tiers.items():
        tp = build_pooled_pairs(epochs, counts, idxs, global_fit)
        f, _ = fit_two_flow(tp, global_fit, PRIOR_SD_S, PRIOR_SD_U_DAY)
        tier_fit[tier] = (f[0] / len(idxs), f[1] / len(idxs))

    spread_floor = math.sqrt(
        SPREAD_FLOOR_BASE ** 2 +
        (SPREAD_FLOOR_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2)
    resid_extra = 0.12 if len(epochs) >= 4 else 0.20

    rows = []
    for i, tr in enumerate(case["trophies"]):
        e_last = counts[-1][i]
        if e_last is None:
            # Trophy missing from the last read: use its last known rate.
            r_last = None
            for k in range(len(epochs) - 1, -1, -1):
                if counts[k][i] is not None:
                    r_last = counts[k][i] / epochs[k]["n"] * 100.0
                    e_last = counts[k][i]
                    break
            if r_last is None:
                r_last = 1.0
                e_last = 0.0
        else:
            r_last = e_last / n_last * 100.0
        if r_last >= 99.99:
            rows.append((tr["id"], saturated_row(levels)))
            continue
        pairs = build_pairs(epochs, counts, i, tier_fit[tr["tier"]])
        (s_i, u_i), cov = fit_two_flow(pairs, tier_fit[tr["tier"]],
                                       PRIOR_SD_S, PRIOR_SD_U_DAY)
        # The new-player flow runs on the growth in N; the existing-player
        # flow runs on the integral of N over the horizon (see above).
        e_star = e_last + s_i * d_n_star + u_i * int_n_days
        e_star = min(max(e_star, 0.0), n_star)
        r_star = 100.0 * e_star / n_star if n_star > 0 else r_last
        r_star = min(max(r_star, 0.02), 99.99)
        # Spread: parameter uncertainty (finite differences on the logit),
        # N-at-target uncertainty, and the horizon floor.
        spread = spread_floor
        if cov is not None:
            var_s, var_u, cov_su = cov
            var_e = ((d_n_star ** 2) * var_s +
                     (int_n_days ** 2) * var_u +
                     2.0 * d_n_star * int_n_days * cov_su)
            if var_e > 0:
                sd_e = math.sqrt(var_e)
                lg0 = logit(r_star)
                e_hi = min(e_star + sd_e, n_star)
                r_hi = 100.0 * e_hi / n_star if n_star > 0 else r_star
                s_param = abs(logit(r_hi) - lg0)
                spread = math.sqrt(spread ** 2 + s_param ** 2)
        if sd_log_n > 0 and d_n_star > 0:
            # A level shift of the N curve scales N* and its integral alike.
            n_hi = n_star * math.exp(sd_log_n)
            e_hi = min(e_last + s_i * (n_hi - n_last) +
                       u_i * int_n_days * math.exp(sd_log_n), n_hi)
            r_hi = 100.0 * e_hi / n_hi if n_hi > 0 else r_star
            n_lo = n_star * math.exp(-sd_log_n)
            e_lo = min(max(e_last + s_i * (n_lo - n_last) +
                           u_i * int_n_days * math.exp(-sd_log_n),
                           0.0), n_lo)
            r_lo = 100.0 * e_lo / n_lo if n_lo > 0 else r_star
            s_n = 0.5 * abs(logit(r_hi) - logit(r_lo))
            spread = math.sqrt(spread ** 2 + s_n ** 2)
        spread = math.sqrt(spread ** 2 + resid_extra ** 2)
        spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
        rows.append((tr["id"], quantile_row(r_star, spread, levels, z_vals)))
    return rows


def saturated_row(levels):
    """Quantiles for a trophy every tracked player holds (rate 100%).

    The median stays at 99.99 (the scorer clamps 100% to 99.99 before the
    logit, so 99.99 is the highest honest median); the lower tail is wide
    enough to cover the trophy slipping a little, which is cheap insurance
    because a low quantile below the truth costs only 2*tau per logit unit.
    """
    s = 1.0
    lg = logit(99.99)
    qs = []
    for tau in levels:
        z = norm_ppf(tau)
        qs.append(round4(clamp_pct(inv_logit(lg + z * s))))
    # The median (the 0.5 level, wherever it sits) is exactly 99.99; the
    # upper quantiles are capped there too by clamp_pct.
    for k, tau in enumerate(levels):
        if tau >= 0.5:
            qs[k] = 99.99
    return qs


# --- branch A: flat history (a single snapshot) -----------------------------
# No refresh has been seen, so the flows cannot be fitted. Forecast the
# first refresh with the launch-night prior: N multiplies by g, the new
# cohort earns each non-saturated trophy at c times its old rate, so the
# rate is multiplied by D = (1 + c*(g-1)) / g. The spread comes from the
# prior widths of g and c (propagated over a deterministic grid, so the
# program stays deterministic) plus per-trophy variation.

def dilution_spread():
    pts = []
    for gi in range(-4, 5):
        for ci in range(-4, 5):
            g = math.exp(PRIOR_LOG_GROWTH_MEAN + PRIOR_LOG_GROWTH_SD * gi / 2.0)
            c = math.exp(PRIOR_LOG_COHORT_MEAN + PRIOR_LOG_COHORT_SD * ci / 2.0)
            pts.append(math.log((1.0 + c * (g - 1.0)) / g))
    mean = sum(pts) / len(pts)
    var = sum((p - mean) ** 2 for p in pts) / len(pts)
    return math.sqrt(var)


def forecast_flat_history(case, epochs, release, target, horizon_days,
                          levels, z_vals):
    last = epochs[-1]
    n_last = last["n"]
    horizon_hours = horizon_days * 24.0
    n_refreshes = max(1, int(round(horizon_hours / REFRESH_INTERVAL_HOURS)))
    g_med = math.exp(PRIOR_LOG_GROWTH_MEAN)
    c_med = math.exp(PRIOR_LOG_COHORT_MEAN)
    d_med = (1.0 + c_med * (g_med - 1.0)) / g_med
    d_all = d_med ** n_refreshes
    sd_log_d = dilution_spread() * math.sqrt(n_refreshes)
    spread = math.sqrt(sd_log_d ** 2 + PRIOR_TROPHY_VAR_LOGIT ** 2 +
                       SPREAD_FLOOR_BASE ** 2 +
                       (SPREAD_FLOOR_PER_SQRT_DAY *
                        math.sqrt(horizon_days)) ** 2)
    spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r_last = last["rates"][i]
        if r_last is None:
            r_last = 1.0
        if r_last >= 99.99:
            rows.append((tr["id"], saturated_row(levels)))
            continue
        r_star = min(max(r_last * d_all, 0.02), 99.99)
        rows.append((tr["id"], quantile_row(r_star, spread, levels, z_vals)))
    return rows


def quantile_row(median_pct, spread, levels, z_vals):
    lg = logit(median_pct)
    qs = []
    prev = None
    for tau, z in zip(levels, z_vals):
        q = round4(clamp_pct(inv_logit(lg + z * spread)))
        if prev is not None and q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


# --- per-case forecast and entrypoint ----------------------------------------

def forecast_case(case: dict, name: str) -> dict:
    levels = list(case["levels"])
    z_vals = [norm_ppf(tau) for tau in levels]
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    release_text = (case.get("game") or {}).get("release_date", "2026-11-19")
    try:
        release = parse_time(release_text + "T00:00:00Z"
                             if len(release_text) == 10 else release_text)
    except ValueError:
        release = parse_time(case["at"]) if case.get("at") else None
    target = parse_time(case["target"])
    if not history:
        # No history at all: no information. Forecast a flat 1% with the
        # widest honest spread rather than failing the case.
        rows = [(tr["id"], quantile_row(1.0, SPREAD_MAX, levels, z_vals))
                for tr in trophies]
        return {"trophies": [{"id": tid, "q": q} for tid, q in rows]}
    epochs = extract_epochs(case)
    counts = [earn_counts(ep, len(trophies)) for ep in epochs]
    if len(epochs) == 1:
        rows = forecast_flat_history(case, epochs, release, target,
                                     horizon_days, levels, z_vals)
    else:
        rows = forecast_with_history(case, epochs, counts, release, target,
                                     horizon_days, levels, z_vals)
    return {"trophies": [{"id": tid, "q": q} for tid, q in rows]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
