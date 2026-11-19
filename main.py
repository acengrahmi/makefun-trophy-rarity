#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase.

Single file, Python 3.13 standard library only, deterministic.

Model (see REPORT.md for the full design and validation):
  * Flat history (every read identical, e.g. launch night, N = 638):
    the next read is the documented first refresh (N 638 -> 3,868).
    Each trophy's new earns are overdispersed counts (negative
    binomial, size 6) whose mean follows a story-earliness allocation:
    the new cohort has finished the opening and almost nothing else,
    so its earns concentrate on the early story chain, not spread
    proportionally over all trophies.  The forecast is the quantiles
    of (E_i + k_i) / N_next, so the median trophy sits at pure
    dilution.  Longer flat horizons compound further refreshes with
    decaying growth (2.0x, 1.5x, ...) and the same allocation.
  * Rich history (many refresh epochs): a two-flow model per trophy,
    dE_i = s_i * dN + U_i * (integral of N dt), fitted per trophy by
    weighted least squares over refresh epoch pairs and shrunk toward
    tier means by empirical Bayes.  N(t) is projected with a
    saturating growth law, flows decay mildly over the horizon, and
    the spread combines posterior variance, fit residuals, N
    uncertainty and a horizon floor.
  * A trophy at 100% stays at 100%.

I/O contract (RULES.txt): read every *.in in $MAKEFUN_INPUT_DIR
(else /input/), write <id>.out per case to $MAKEFUN_OUTPUT_DIR
(else /output/): {"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}]}, 55 rows, non-decreasing quantiles in [0, 100].
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Flat branch constants (from the design).
NB_SIZE = 6.0          # negative-binomial size for new earns per refresh
TOTAL_NEW_EARNS = 80.0  # expected new earns K in the first refresh
FIRST_N = 3868.0       # documented refreshed players-tracked
FIRST_N0 = 638.0       # launch-night players-tracked
LOGIT_FLOOR = 0.02     # small logit spread floor (refresh-size uncertainty)
REFRESH_MINUTES = 90.0  # cadence assumed between successive refreshes

# Rich branch constants (from the design).
FLOW_DECAY_DAYS = 12.0  # flows decay as exp(-H / 12 days)
N_DECAY_DAYS = 2.5      # N log-growth rate decays with a 2.5-day constant
TIER_PSEUDO = 2.0       # pseudo-count pulling tier means to the global mean


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(x):
    """logit of a percent, clamped to the 0.01..99.99 range the page prints."""
    p = clamp(x, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def expit_pct(l):
    """Inverse of logit_pct, as a percent."""
    if l >= 0:
        p = 1.0 / (1.0 + math.exp(-l))
    else:
        e = math.exp(l)
        p = e / (1.0 + e)
    return clamp(p * 100.0, 0.01, 99.99)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def negbin_quantiles(mean, size, taus):
    """Quantiles of NB(size, mean) by PMF recurrence. Deterministic.

    Returns one integer count per tau.  mean <= 0 gives all zeros.
    """
    out = []
    if mean <= 0.0:
        return [0] * len(taus)
    p = size / (size + mean)  # P(0) = p**size
    pmf = p ** size
    cdf = pmf
    k = 0
    # Upper cap: far past the 0.9999 point for any plausible mean.
    var = mean + mean * mean / size
    kmax = int(mean + 25.0 * math.sqrt(var) + 100.0)
    for tau in taus:
        while cdf < tau and k < kmax:
            k += 1
            # P(k) = P(k-1) * (k-1+size)/k * (1-p)
            pmf = pmf * (k - 1.0 + size) / k * (1.0 - p)
            cdf += pmf
        out.append(k)
    return out


def trophy_weight(trophy):
    """Story-earliness weight for the flat branch's new-earn allocation.

    Chapter 1 1.0, Chapter 2 0.45, Chapter 3 0.20, Chapter 4 0.09,
    final mission 0.02, named missions 0.15 each, quick side
    activities 0.05, everything else 0.03, 100%-completion gold 0.
    Classification is by description text, not by id.
    """
    desc = (trophy.get("description") or "").strip().lower()
    if "100% completion" in desc:
        return 0.0
    if desc.startswith("complete chapter 1"):
        return 1.0
    if desc.startswith("complete chapter 2"):
        return 0.45
    if desc.startswith("complete chapter 3"):
        return 0.20
    if desc.startswith("complete chapter 4"):
        return 0.09
    if desc == "complete the final mission.":
        return 0.02
    # Named missions: "Complete <Title>." / "Complete the final
    # mission for <person>." -- titles carry no digits, unlike the
    # grind trophies ("Complete 250 Stunts.", "Complete 8 mission
    # strands ...", "Complete 3 Collectible Card sets.",
    # "Complete 50 Challenges.").
    if desc.startswith("complete ") and not any(ch.isdigit() for ch in desc):
        return 0.15
    quick = ("arcade game", "fishing challenge", "cocktail", "bowling",
             "meditate", "try each type", "unscheduled departure",
             "showroom", "melee drive-by")
    if any(k in desc for k in quick):
        return 0.05
    return 0.03


def last_known_rates(case):
    """Per-trophy rate from the last read that lists it (None if never)."""
    trophies = case["trophies"]
    rates = [None] * len(trophies)
    for read in case.get("history", []):
        rs = read.get("rates") or []
        for i in range(min(len(trophies), len(rs))):
            if rs[i] is not None:
                rates[i] = float(rs[i])
    return rates


def extract_epochs(case):
    """Collapse consecutive identical reads into refresh epochs.

    Returns a list of (time_days, players_tracked, rates_tuple),
    one per distinct page state, oldest first.  Identical reads are
    the same Exophase refresh seen twice, not new information.
    """
    epochs = []
    for read in case.get("history", []):
        n = read.get("players_tracked")
        rates = tuple(read.get("rates") or [])
        if n is None:
            continue
        key = (n, rates)
        if epochs and (epochs[-1][1], epochs[-1][2]) == key:
            continue
        t = parse_time(read["at"])
        epochs.append((t, n, rates))
    return epochs


def forecast_flat(case, n_last, rates_last, horizon_minutes):
    """Flat history: every read identical, so the next read is a refresh.

    First refresh: N 638 -> 3,868 (documented) or a generic 3x prior
    at any other N.  Longer horizons compound further refreshes with
    decaying growth factors 2.0x, 1.5x, 4/3x, ... (factor (k+1)/k),
    so N_final = N_first * n_refreshes, and each refresh adds the
    same allocation of K = 80 expected new earns.  New earns over
    n refreshes are NB(size 6*n, mean n*mu_i).
    """
    horizon_days = horizon_minutes / 1440.0
    n_ref = max(1, int(round(horizon_minutes / REFRESH_MINUTES)))
    if abs(n_last - FIRST_N0) < 1.0:
        n_first = FIRST_N
    else:
        n_first = n_last * 3.0
    n_final = n_first * n_ref
    trophies = case["trophies"]
    weights = [trophy_weight(t) for t in trophies]
    wsum = sum(weights) or 1.0
    rows = []
    for i, t in enumerate(trophies):
        rate = rates_last[i]
        if rate is None:
            rate = 0.0
        e_last = rate / 100.0 * n_last
        if rate >= 99.995:
            # A trophy at 100% stays at 100%; only the lower tail
            # widens slowly with horizon and N growth.
            q05 = max(99.0, 100.0 - 0.1 - 0.05 * math.sqrt(horizon_days))
            rows.append({"id": t["id"], "q": [round(q05, 4), 99.99,
                                               99.99, 99.99, 99.99]})
            continue
        mu = TOTAL_NEW_EARNS * weights[i] / wsum * n_ref
        ks = negbin_quantiles(mu, NB_SIZE * n_ref, LEVELS)
        qs = []
        for j, tau in enumerate(LEVELS):
            r = 100.0 * (e_last + ks[j]) / n_final
            # Small logit floor for refresh-size uncertainty.
            r = expit_pct(logit_pct(r) + Z[j] * LOGIT_FLOOR)
            qs.append(round(clamp(r, 0.01, 99.99), 4))
        # Keep quantiles non-decreasing after rounding.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": t["id"], "q": qs})
    return rows


def forecast_fallback(case, horizon_minutes):
    """No usable history at all: flat 1% forecast with maximum spread."""
    horizon_days = horizon_minutes / 1440.0
    sigma = 1.0 + 0.06 * math.sqrt(horizon_days)
    rows = []
    for t in case["trophies"]:
        qs = [round(expit_pct(logit_pct(1.0) + Z[j] * sigma), 4)
              for j in range(5)]
        rows.append({"id": t["id"], "q": qs})
    return rows


def _solve2(a11, a12, a22, b1, b2):
    """Solve a symmetric 2x2 system; None when (near-)singular."""
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12 * max(1.0, abs(a11 * a22)):
        return None
    return ((b1 * a22 - b2 * a12) / det,
            (a11 * b2 - a12 * b1) / det)


def _wls(pairs, ys, weights):
    """Weighted least squares of y ~ s*dN + U*A over epoch pairs.

    pairs: list of (dN, A); ys: outcome per pair; weights: per pair.
    Returns (s, U, var_s, var_U, resid_var) or None if degenerate.
    """
    a11 = a12 = a22 = b1 = b2 = 0.0
    for (dn, a), y, w in zip(pairs, ys, weights):
        a11 += w * dn * dn
        a12 += w * dn * a
        a22 += w * a * a
        b1 += w * dn * y
        b2 += w * a * y
    sol = _solve2(a11, a12, a22, b1, b2)
    if sol is None:
        return None
    s, u = sol
    m = len(pairs)
    rss = 0.0
    wsum = 0.0
    for (dn, a), y, w in zip(pairs, ys, weights):
        r = y - s * dn - u * a
        rss += w * r * r
        wsum += w
    dof = max(1, m - 2)
    sig2 = rss / dof if wsum > 0 else 0.0
    inv = _solve2(a11, a12, a22, 1.0, 0.0)
    if inv is None:
        return (s, u, float("inf"), float("inf"), sig2)
    # (X'WX)^-1 diagonal: inv of [[a11,a12],[a12,a22]] solved for
    # unit vectors gives columns; var_s = sig2 * inv11, var_U = sig2 * inv22.
    inv11 = inv[0]
    inv22 = _solve2(a11, a12, a22, 0.0, 1.0)[1]
    return (s, u, max(0.0, sig2 * inv11), max(0.0, sig2 * inv22), sig2)


def _median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.0
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _loglin_slope(ts, lns):
    """Least-squares slope of ln N against time (days)."""
    n = len(ts)
    if n < 2:
        return 0.0
    mt = sum(ts) / n
    ml = sum(lns) / n
    num = sum((t - mt) * (l - ml) for t, l in zip(ts, lns))
    den = sum((t - mt) ** 2 for t in ts)
    return num / den if den > 0 else 0.0


def forecast_rich(case, epochs, horizon_minutes):
    """Rich history: two-flow model per trophy over refresh epochs."""
    trophies = case["trophies"]
    n_tr = len(trophies)
    tiers = [t.get("tier") or "bronze" for t in trophies]
    horizon_days = horizon_minutes / 1440.0
    t0 = epochs[0][0]
    times = [(e[0] - t0).total_seconds() / 86400.0 for e in epochs]
    ns = [float(e[1]) for e in epochs]

    # Epoch pairs: dN, A = integral of N dt (trapezoid, player-days),
    # dE per trophy (skipped where a read lacks the trophy).
    pairs = []
    dE = [[] for _ in range(n_tr)]
    for j in range(len(epochs) - 1):
        dt = times[j + 1] - times[j]
        if dt <= 0:
            continue
        dn = ns[j + 1] - ns[j]
        a = 0.5 * (ns[j] + ns[j + 1]) * dt
        pairs.append((dn, a))
        r0 = epochs[j][2]
        r1 = epochs[j + 1][2]
        for i in range(n_tr):
            if i < len(r0) and i < len(r1) and r0[i] is not None \
                    and r1[i] is not None:
                dE[i].append(float(r1[i]) / 100.0 * ns[j + 1]
                             - float(r0[i]) / 100.0 * ns[j])
            else:
                dE[i].append(None)
    if len(pairs) < 1:
        return None

    # Pooled fits: per tier (summed counts / tier size) and global.
    def pooled(idxs):
        ys = []
        for k in range(len(pairs)):
            tot = 0.0
            ok = True
            for i in idxs:
                if dE[i][k] is None:
                    ok = False
                    break
                tot += dE[i][k]
            ys.append(tot if ok else 0.0)
        fit = _wls(pairs, ys, [1.0] * len(pairs))
        if fit is None:
            return (0.0, 0.0)
        return (max(0.0, fit[0]) / max(1, len(idxs)),
                max(0.0, fit[1]) / max(1, len(idxs)))

    tier_names = sorted(set(tiers))
    tier_mean = {}
    for name in tier_names:
        idxs = [i for i in range(n_tr) if tiers[i] == name]
        tier_mean[name] = pooled(idxs)
    global_mean = pooled(list(range(n_tr)))
    # Pull tier means toward the global mean by tier size.
    for name in tier_names:
        n_t = sum(1 for t in tiers if t == name)
        s_t, u_t = tier_mean[name]
        tier_mean[name] = (
            (n_t * s_t + TIER_PSEUDO * global_mean[0]) / (n_t + TIER_PSEUDO),
            (n_t * u_t + TIER_PSEUDO * global_mean[1]) / (n_t + TIER_PSEUDO))

    # Cross-trophy variance of the per-trophy fits, per parameter,
    # for shrinkage; small tiers use the global variance.
    raw = []
    for i in range(n_tr):
        ys = [v if v is not None else 0.0 for v in dE[i]]
        ok = [k for k in range(len(pairs)) if dE[i][k] is not None]
        if len(ok) >= 2:
            sub_pairs = [pairs[k] for k in ok]
            sub_ys = [ys[k] for k in ok]
            fit = _wls(sub_pairs, sub_ys, [1.0] * len(ok))
            raw.append(fit)
        else:
            raw.append(None)

    def _var(vals):
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)

    gvar_s = _var([f[0] for f in raw if f])
    gvar_u = _var([f[1] for f in raw if f])
    tier_var = {}
    for name in tier_names:
        idxs = [i for i in range(n_tr) if tiers[i] == name]
        vs = _var([raw[i][0] for i in idxs if raw[i]])
        vu = _var([raw[i][1] for i in idxs if raw[i]])
        tier_var[name] = (vs if vs is not None else gvar_s,
                          vu if vu is not None else gvar_u)

    # Second pass: weights from tier-mean expected counts (never the
    # observed outcome), then empirical-Bayes shrinkage to tier means.
    rates_last = last_known_rates(case)
    n_last = ns[-1]
    e_last = []
    for i in range(n_tr):
        r = rates_last[i]
        e_last.append(0.0 if r is None else r / 100.0 * n_last)

    post = []  # (s, U, var_s, var_U, resid_var) per trophy
    for i in range(n_tr):
        s_t, u_t = tier_mean[tiers[i]]
        vs_t, vu_t = tier_var[tiers[i]]
        fit = raw[i]
        if fit is None:
            post.append((s_t, u_t, vs_t or 0.0, vu_t or 0.0, 0.0))
            continue
        ok = [k for k in range(len(pairs)) if dE[i][k] is not None]
        sub_pairs = [pairs[k] for k in ok]
        sub_ys = [dE[i][k] for k in ok]
        wts = [1.0 / (1.0 + max(0.0, s_t * dn + u_t * a))
               for (dn, a) in sub_pairs]
        fit2 = _wls(sub_pairs, sub_ys, wts)
        if fit2 is None:
            post.append((s_t, u_t, vs_t or 0.0, vu_t or 0.0, 0.0))
            continue
        s_f, u_f, vs_f, vu_f, resid = fit2

        def shrink(fit_v, fit_var, tier_v, tier_mean_v):
            if fit_var is None or fit_var == float("inf") or fit_var <= 0:
                return tier_mean_v, (tier_v or 0.0)
            if tier_v is None or tier_v <= 0:
                return fit_v, fit_var
            w_f = 1.0 / fit_var
            w_t = 1.0 / tier_v
            return ((fit_v * w_f + tier_mean_v * w_t) / (w_f + w_t),
                    1.0 / (w_f + w_t))

        s_p, vs_p = shrink(s_f, vs_f, vs_t, s_t)
        u_p, vu_p = shrink(u_f, vu_f, vu_t, u_t)
        post.append((max(0.0, s_p), max(0.0, u_p), vs_p, vu_p, resid))

    # N(t) projection: saturating growth law.  Checkpoint log-growth
    # rate = geometric mean of the median of the last 3 epoch rates
    # and a log-linear fit over the last 8 epochs; decayed with a
    # 2.5-day time constant; integrated to the effective target
    # (target minus half the recent refresh gap).
    epoch_rates = []
    for j in range(len(epochs) - 1):
        dt = times[j + 1] - times[j]
        if dt > 0 and ns[j + 1] > ns[j]:
            epoch_rates.append(math.log(ns[j + 1] / ns[j]) / dt)
    med3 = _median(epoch_rates[-3:]) if epoch_rates else 0.0
    k8 = min(8, len(epochs))
    slope8 = _loglin_slope(times[-k8:], [math.log(n) for n in ns[-k8:]])
    r0 = math.sqrt(max(med3, 1e-4) * max(slope8, 1e-4))
    r0 = clamp(r0, 1e-3, 5.0)
    gaps = [times[j + 1] - times[j] for j in range(len(epochs) - 1)]
    gap = _median(gaps[-4:]) if gaps else 0.5 / 24.0
    h_eff = max(0.0, horizon_days - 0.5 * gap)
    n_target = n_last * math.exp(r0 * N_DECAY_DAYS *
                                 (1.0 - math.exp(-h_eff / N_DECAY_DAYS)))
    # Integral of N dt over the horizon (trapezoid, 100 steps).
    steps = 100
    int_n = 0.0
    prev_n = n_last
    prev_t = 0.0
    for st in range(1, steps + 1):
        tt = h_eff * st / steps
        nn = n_last * math.exp(r0 * N_DECAY_DAYS *
                               (1.0 - math.exp(-tt / N_DECAY_DAYS)))
        int_n += 0.5 * (prev_n + nn) * (tt - prev_t)
        prev_n, prev_t = nn, tt

    decay = math.exp(-horizon_days / FLOW_DECAY_DAYS)
    dn_proj = n_target - n_last
    hist_exposure = sum(dn * dn + a * a for (dn, a) in pairs) or 1.0
    floor = 0.10 + 0.06 * math.sqrt(horizon_days)
    rows = []
    for i, t in enumerate(trophies):
        rate = rates_last[i]
        if rate is not None and rate >= 99.995:
            q05 = max(99.0, 100.0 - 0.1 - 0.05 * math.sqrt(horizon_days))
            rows.append({"id": t["id"], "q": [round(q05, 4), 99.99,
                                               99.99, 99.99, 99.99]})
            continue
        s_p, u_p, vs_p, vu_p, resid = post[i]
        de = decay * (s_p * dn_proj + u_p * int_n)
        e_tgt = max(0.0, e_last[i] + de)
        p = clamp(e_tgt / n_target, 0.0001, 0.9999)
        # Spread: posterior variance + fit residuals (scaled by
        # projection leverage) + N uncertainty + horizon floor.
        lev = min(25.0, (dn_proj * dn_proj + int_n * int_n) / hist_exposure)
        var_de = (vs_p * dn_proj * dn_proj + vu_p * int_n * int_n
                  + resid * lev)
        var_rate = var_de / (n_target * n_target) + \
            (p * 0.08 * math.sqrt(horizon_days)) ** 2
        sig_logit = math.sqrt(max(0.0, var_rate)) / (p * (1.0 - p))
        sigma = math.sqrt(sig_logit * sig_logit + floor * floor)
        base = logit_pct(100.0 * p)
        qs = []
        for j in range(5):
            qs.append(round(clamp(expit_pct(base + Z[j] * sigma),
                                  0.01, 99.99), 4))
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": t["id"], "q": qs})
    return rows


def forecast_case(case):
    """Dispatch on history shape: flat -> flat branch, else rich."""
    horizon = float(case.get("horizon_minutes") or 0)
    history = case.get("history") or []
    if not history:
        return forecast_fallback(case, horizon)
    epochs = extract_epochs(case)
    rates_last = last_known_rates(case)
    n_last = None
    for read in reversed(history):
        if read.get("players_tracked") is not None:
            n_last = float(read["players_tracked"])
            break
    if n_last is None or n_last <= 0:
        return forecast_fallback(case, horizon)
    if len(epochs) <= 1:
        return forecast_flat(case, n_last, rates_last, horizon)
    rich = forecast_rich(case, epochs, horizon)
    if rich is None:
        return forecast_flat(case, n_last, rates_last, horizon)
    return rich


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        rows.sort(key=lambda r: r["id"])
        out = {"trophies": rows}
        out_path = os.path.join(out_dir, case["id"] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
