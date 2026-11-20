"""gta6-trophy-rarity-forecast-01: a count-based forecaster for GTA VI trophy earn rates.

The idea, from the shape of the data itself:

  * Every rate the page prints is an integer count of earners over the players
    tracked, rounded to 2 decimals: rate = round(100 * count / players, 2).
    The practice page proves it: with 638 players tracked, every one of the 55
    rates is exactly the 2-decimal rounding of an integer count over 638, and
    the counts sum exactly to the page's trophies_earned (943). So the honest
    state of the game is not the rates but the counts, and a forecast should
    be built out of counts and a player total, never out of rates alone.
  * The page only changes when Exophase refreshes it, so a case's history is a
    step function: long runs of identical reads, then a jump. Collapse the
    history into its distinct states (players, earned, counts) and model the
    states, not the reads.
  * Counts never decrease, the counts sum to trophies_earned at every state,
    and the players total grows on a saturating launch curve. Forecast the
    two totals (players, earned) on a log-time axis, forecast each trophy's
    count with its slope shrunk toward the pack, then force the counts to
    sum to the forecast earned total: the accounting identity the page
    itself obeys. Rates are counts over players at the end.
  * A history with a single state has seen no refresh, so say so: blend
    persistence with the refresh the rules describe (a flood of new tracked
    players who hold almost nothing but the trophy everyone earns, so every
    other rate is diluted), weighted by how stale the state is against the
    refresh cadence the launch-night recording shows.

Python 3.13, standard library only, deterministic: the same case always
writes the same forecast, byte for byte.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RATE_LO = 0.01    # the scorer clamps rates into [0.01, 99.99] before the logit
RATE_HI = 99.99


def parse_ts(text):
    """Seconds since the epoch for an ISO 8601 UTC stamp.

    Reads carry fractional seconds (2026-11-18T23:50:49.360Z) while case
    checkpoints do not, so parse with fromisoformat, not a fixed format.
    """
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def rate_logit(rate_pct):
    """Logit of a rate in percent, on the scorer's clamped scale."""
    return logit(clamp(rate_pct, RATE_LO, RATE_HI) / 100.0)


def recover_counts(rates, players):
    """Integer earner counts behind one read's rates (None where a rate is null)."""
    return [None if r is None else int(round(r * players / 100.0)) for r in rates]


def build_states(case):
    """Collapse a case's history into its distinct page states, oldest first.

    A state is one (players_tracked, trophies_earned, rates) the page held;
    its time is the first read that showed it. Counts are recovered per read
    and nulls are carried forward from the trophy's last known count.
    """
    states = []
    prev_key = None
    carried = [None] * len(case["trophies"])
    for read in case["history"]:
        counts = recover_counts(read["rates"], read["players_tracked"])
        for i, c in enumerate(counts):
            if c is not None:
                carried[i] = c
        key = (read["players_tracked"], read["trophies_earned"],
               tuple(read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        states.append({
            "t": parse_ts(read.get("at") or read["slot"]),
            "players": read["players_tracked"],
            "earned": read["trophies_earned"],
            "counts": list(carried),
            "rates": list(read["rates"]),
        })
    return states


def fit_line(xs, ys):
    """Least-squares (intercept, slope) of ys on xs; flat (mean, 0) if xs is degenerate."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return my - (sxy / sxx) * mx, sxy / sxx


def log_time_hours(t, anchor):
    """ln(1 + hours since the anchor): the launch curve's natural clock."""
    return math.log1p(max(0.0, (t - anchor) / 3600.0))


def forecast_multi_state(states, anchor, target_t, horizon_days):
    """Forecast (median rate %, sigma in logits) per trophy from >= 2 states.

    Totals and counts are extrapolated on the log-time axis; each trophy's
    count slope is shrunk toward zero by the number of states behind it, the
    counts are re-anchored on the last state, held non-decreasing, and then
    rescaled so they sum to the forecast earned total.
    """
    xs = [log_time_hours(s["t"], anchor) for s in states]
    x_last, x_target = xs[-1], log_time_hours(target_t, anchor)
    n_tr = len(states[0]["counts"])
    last = states[-1]

    a_p, b_p = fit_line(xs, [s["players"] for s in states])
    a_e, b_e = fit_line(xs, [s["earned"] for s in states])
    players_t = max(float(last["players"]), a_p + b_p * x_target)
    earned_t = max(float(last["earned"]), a_e + b_e * x_target)
    earned_t = min(earned_t, n_tr * players_t)

    shrink = len(states) / (len(states) + 2.0)
    counts_t = []
    for i in range(n_tr):
        ys = [s["counts"][i] for s in states]
        c_last = last["counts"][i]
        if c_last is None:
            counts_t.append(None)
            continue
        if all(y is not None for y in ys):
            _, slope = fit_line(xs, [float(y) for y in ys])
        else:
            slope = 0.0
        c = c_last + slope * shrink * (x_target - x_last)
        counts_t.append(max(float(c_last), c))

    known = [c for c in counts_t if c is not None]
    if known and sum(known) > 0:
        scale = earned_t / sum(known)
        counts_t = [None if c is None else
                    min(players_t, max(float(last["counts"][i]), c * scale))
                    for i, c in enumerate(counts_t)]

    # Observed logit-rate volatility between successive states, pooled over
    # trophies: the floor under the spread when the page has been jumpy.
    span_h = max(1.0, (states[-1]["t"] - states[0]["t"]) / 3600.0)
    diffs = []
    for prev, cur in zip(states, states[1:]):
        for i in range(n_tr):
            rp, rc = prev["rates"][i], cur["rates"][i]
            if rp is not None and rc is not None:
                diffs.append(rate_logit(rc) - rate_logit(rp))
    vol = 0.0
    if diffs:
        rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        vol = 0.9 * rms * math.sqrt((horizon_days * 24.0) / span_h)

    out = []
    for i in range(n_tr):
        c_last = last["counts"][i]
        if c_last is None or counts_t[i] is None:
            out.append((0.05, 1.5))
            continue
        if c_last >= last["players"]:
            out.append((RATE_HI, 0.10))  # everyone tracked holds it; stays so
            continue
        median = clamp(100.0 * counts_t[i] / players_t, RATE_LO, RATE_HI)
        sigma = 0.30 + 0.45 * math.sqrt(horizon_days) \
            + 0.55 / math.sqrt(max(1, c_last))
        out.append((median, clamp(max(sigma, vol), 0.05, 2.2)))
    return out


def forecast_single_state(state, anchor, at_t, target_t, horizon_days):
    """Forecast from one flat state: persistence blended with a refresh.

    The launch-night page sat unchanged for over 4 hours, then a refresh
    multiplied the tracked players about sixfold while the earners of every
    trophy but the universal one grew only modestly in total: a dilution.
    Weight that scenario by how stale the state is against a ~4.7 h refresh
    cadence, with the violence of the flood decaying over the first days.
    """
    horizon_h = max(0.0, (target_t - at_t) / 3600.0)
    age_h = max(0.0, (at_t - anchor) / 3600.0)
    decay = math.exp(-max(0.0, age_h - 3.0) / 36.0)
    state_age_h = max(0.0, (at_t - state["t"]) / 3600.0)
    p_ref = clamp((state_age_h + horizon_h) / 4.7, 0.0, 1.0) * 0.98
    g_players = 1.0 + 5.06 * decay
    g_counts = 1.0 + 0.34 * decay
    base = 0.30 + 0.35 * math.sqrt(horizon_days)

    out = []
    for i, rate in enumerate(state["rates"]):
        c = state["counts"][i]
        if rate is None or c is None:
            out.append((0.05, 1.5))
            continue
        if c >= state["players"]:
            out.append((RATE_HI, 0.10))
            continue
        r0 = clamp(rate, RATE_LO, RATE_HI)
        r1 = clamp(rate * g_counts / g_players, RATE_LO, RATE_HI)
        l0, l1 = rate_logit(r0), rate_logit(r1)
        mu = (1.0 - p_ref) * l0 + p_ref * l1
        mix = math.sqrt(p_ref * (1.0 - p_ref)) * abs(l1 - l0)
        sigma = clamp(math.sqrt(mix * mix + base * base), 0.05, 2.5)
        out.append((100.0 * inv_logit(mu), sigma))
    return out


def quantile_row(median_rate, sigma):
    """Five quantiles around a median rate, spread sigma on the logit scale."""
    centre = rate_logit(median_rate)
    row = []
    for z in Z:
        q = 100.0 * inv_logit(centre + z * sigma)
        row.append(round(clamp(q, 0.0, 100.0), 4))
    return [max(v, row[i - 1]) if i else v for i, v in enumerate(row)]


def forecast_case(case):
    """The forecast rows for one case: [{"id": n, "q": [...]}, ...] in id order."""
    trophies = case["trophies"]
    at_t = parse_ts(case["at"])
    target_t = parse_ts(case["target"])
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)
    try:
        anchor = parse_ts(case["game"]["release_date"] + "T00:00:00Z")
    except Exception:
        anchor = at_t

    states = build_states(case)
    if not states:
        medians = [(0.05, 1.5)] * len(trophies)
    elif len(states) >= 2:
        medians = forecast_multi_state(states, anchor, target_t, horizon_days)
    else:
        medians = forecast_single_state(states[0], anchor, at_t, target_t,
                                        horizon_days)

    rows = []
    for trophy, (median, sigma) in zip(trophies, medians):
        rows.append({"id": trophy["id"], "q": quantile_row(median, sigma)})
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        payload = {"trophies": forecast_case(case)}
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(json.dumps(payload, separators=(",", ":")),
                            encoding="utf-8")


if __name__ == "__main__":
    main()
