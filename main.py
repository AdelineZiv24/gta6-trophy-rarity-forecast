"""gta6-trophy-rarity-forecast-01 entry: hierarchical logit-drift forecaster.

Original model (not the starter persistence baseline):
  * History is deduped into distinct page snapshots (Exophase refreshes freeze
    the page, then it jumps), so repeated identical reads count once.
  * Each trophy's logit(rate) is modelled as drifting linearly in log-time
    x = ln(1 + hours since 2026-11-19T00:00Z). The per-trophy OLS slope is
    shrunk hierarchically toward its tier mean and the global mean slope,
    anchored at the last observed logit, with the total drift capped.
  * The predictive spread on the logit scale combines (a) a structural floor
    sigma(h) = sqrt(A^2 + (B*sqrt(h_days))^2) so flat/short histories still get
    an honest wide spread, (b) realised snapshot-to-snapshot logit volatility
    pooled across trophies, and (c) a rolling-origin backtest inside the given
    history (fit on prefixes, score later snapshots, pool errors by horizon).
    The widest of the three (with damping on the empirical terms) wins, capped.
  * Quantiles are inv_logit(mu + z_tau * sigma) with z for the case levels.

Deterministic: no randomness, no wall clock, sorted file iteration, fixed
iteration orders everywhere. Standard library only, no network.
On any internal failure a case falls back to persistence with a wide spread.
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
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc).timestamp()

# Model constants (Round 2, tuned by simulation backtests; see BACKTEST.md).
PARAMS = {
    "shrink_k": 4.0,        # pseudo-count pulling a trophy slope to its prior
    "slope_clamp": 2.5,     # |slope| cap, logit per unit log-time
    "drift_cap": 1.2,       # |total drift| cap in logit units
    "floor_a": 0.15,        # structural spread intercept (refresh-jump risk)
    "floor_b": 0.20,        # structural spread per sqrt(day)
    "emp_damp": 0.10,       # damping on empirical (vol / rolling-origin) sigma
    "sigma_cap": 1.4,
    "fallback_b": 0.9,      # fallback persistence spread per sqrt(day)
    "fallback_a": 0.4,
    "recent_frac": 0.30,    # fit slopes on the last this-fraction of log-time span
    "drift_scale": 0.32,    # global damping on extrapolated drift
}


def clamp_pct(x):
    if x is None:
        return 0.01
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.01
    if not math.isfinite(v):
        return 0.01
    return min(99.99, max(0.01, v))


def logit_pct(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        v = 1.0 / (1.0 + math.exp(-lg))
    else:
        e = math.exp(lg)
        v = e / (1.0 + e)
    return 100.0 * v


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """Parse an ISO-8601 UTC timestamp to epoch seconds; None on failure."""
    if not isinstance(s, str):
        return None
    txt = s.strip()
    try:
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def x_of(ts):
    """Log-time coordinate: ln(1 + hours since the launch epoch)."""
    hours = (ts - EPOCH) / 3600.0
    if hours <= -0.999:
        hours = -0.999
    return math.log1p(hours)


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs; None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for i in range(n):
        dx = xs[i] - mx
        sxx += dx * dx
        sxy += dx * (ys[i] - my)
    if sxx <= 1e-12:
        return None
    return sxy / sxx


def dedupe_snapshots(history, n_trophies):
    """Collapse consecutive identical reads into distinct page snapshots.

    Returns a list of (timestamp, players, rates-tuple). A snapshot's
    timestamp is the latest read still showing that state.
    """
    snaps = []
    prev_key = None
    for read in history:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        tup = tuple(rates[i] if i < len(rates) else None for i in range(n_trophies))
        players = read.get("players_tracked")
        key = (players, tup)
        ts = parse_time(read.get("slot")) or parse_time(read.get("at"))
        if ts is None:
            ts = snaps[-1][0] if snaps else EPOCH
        if key == prev_key and snaps:
            snaps[-1] = (ts, snaps[-1][1], snaps[-1][2])
        else:
            snaps.append((ts, players, tup))
            prev_key = key
    return snaps


def fallback_forecast(case):
    """Persistence median with a wide structural spread. Never raises."""
    horizon_days = max(0.0, float(case.get("horizon_minutes") or 0) / 1440.0)
    sigma = math.sqrt(PARAMS["fallback_a"] ** 2 + (PARAMS["fallback_b"] * math.sqrt(horizon_days)) ** 2)
    sigma = min(sigma, PARAMS["sigma_cap"])
    history = case.get("history") or []
    last_rates = None
    for read in reversed(history):
        if isinstance(read, dict) and isinstance(read.get("rates"), list):
            last_rates = read["rates"]
            break
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        rate = last_rates[i] if last_rates is not None and i < len(last_rates) else None
        lg = logit_pct(rate if rate is not None else 0.01)
        qs = [round4(min(100.0, max(0.0, inv_logit_pct(lg + z * sigma)))) for z in Z]
        qs = sorted(qs)
        rows.append({"id": trophy.get("id"), "q": qs})
    return {"trophies": rows}


def model_forecast(case):
    trophies = case["trophies"]
    n = len(trophies)
    history = case.get("history") or []
    horizon_minutes = float(case.get("horizon_minutes") or 0)
    horizon_days = max(0.0, horizon_minutes / 1440.0)

    at_ts = parse_time(case.get("at"))
    target_ts = parse_time(case.get("target"))
    if target_ts is None and at_ts is not None:
        target_ts = at_ts + horizon_minutes * 60.0
    if target_ts is None:
        target_ts = EPOCH + horizon_minutes * 60.0
    x_target = x_of(target_ts)

    snaps = dedupe_snapshots(history, n)
    if not snaps:
        return fallback_forecast(case)
    snap_x = [x_of(s[0]) for s in snaps]
    x_last = snap_x[-1]
    dx = max(0.0, x_target - x_last)

    # Per-trophy logit series over distinct snapshots.
    series = []  # (xs, ys) per trophy, only non-null reads
    last_logit = []
    for i in range(n):
        xs = []
        ys = []
        for j in range(len(snaps)):
            r = snaps[j][2][i]
            if r is not None:
                xs.append(snap_x[j])
                ys.append(logit_pct(r))
        series.append((xs, ys))
        last_logit.append(ys[-1] if ys else logit_pct(0.01))

    # Raw slopes + hierarchical priors (tier mean, global mean). Slopes are fit
    # on the recent part of the log-time span (adoption decelerates, so the
    # early steep phase must not dominate the extrapolation).
    rf = PARAMS["recent_frac"]
    x_cut = snap_x[-1] - rf * (snap_x[-1] - snap_x[0]) if rf < 1.0 else None
    raw_slope = []
    slope_counts = []
    for i in range(n):
        xs, ys = series[i]
        if x_cut is not None:
            pts = [(a, b) for a, b in zip(xs, ys) if a >= x_cut]
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
        slope_counts.append(len(xs))
        if len(xs) >= 2 and xs[-1] - xs[0] > 0.02:
            raw_slope.append(ols_slope(xs, ys))
        else:
            raw_slope.append(None)
    valid = [s for s in raw_slope if s is not None]
    global_mean = sum(valid) / len(valid) if valid else 0.0
    tier_sum = {}
    tier_cnt = {}
    for i in range(n):
        if raw_slope[i] is not None:
            t = trophies[i].get("tier")
            tier_sum[t] = tier_sum.get(t, 0.0) + raw_slope[i]
            tier_cnt[t] = tier_cnt.get(t, 0) + 1
    k = PARAMS["shrink_k"]
    slopes = []
    for i in range(n):
        t = trophies[i].get("tier")
        if t in tier_cnt:
            prior = 0.5 * (tier_sum[t] / tier_cnt[t]) + 0.5 * global_mean
        else:
            prior = global_mean
        s = raw_slope[i]
        if s is None:
            slopes.append(prior if valid else 0.0)
        else:
            cnt = slope_counts[i]
            w = cnt / (cnt + k)
            slopes.append(w * s + (1.0 - w) * prior)
    sc = PARAMS["slope_clamp"]
    slopes = [min(sc, max(-sc, s)) for s in slopes]

    # Empirical sigma (b): pooled snapshot-to-snapshot logit volatility.
    vol_sigma = 0.0
    sq = 0.0
    cnt = 0
    for i in range(n):
        xs, ys = series[i]
        for j in range(1, len(xs)):
            dt_days = (xs[j] - xs[j - 1])  # in log-time units; convert below
            # Use real elapsed days between the two snapshots instead.
            cnt += 1
            sq += (ys[j] - ys[j - 1]) ** 2
    # Recompute with real time gaps (log-time gaps distort short horizons).
    sq = 0.0
    cnt = 0
    for i in range(n):
        idx = [j for j in range(len(snaps)) if snaps[j][2][i] is not None]
        for a in range(1, len(idx)):
            j0, j1 = idx[a - 1], idx[a]
            dt_days = (snaps[j1][0] - snaps[j0][0]) / 86400.0
            if dt_days <= 0:
                continue
            d = logit_pct(snaps[j1][2][i]) - logit_pct(snaps[j0][2][i])
            if abs(d) > 3.0:
                d = math.copysign(3.0, d)
            sq += (d * d) / dt_days
            cnt += 1
    if cnt >= 20:
        vol_per_sqrt_day = math.sqrt(sq / cnt)
        vol_sigma = vol_per_sqrt_day * math.sqrt(horizon_days)

    # Empirical sigma (c): rolling-origin backtest inside the history.
    ro_sigma = 0.0
    if len(snaps) >= 4 and horizon_days > 0:
        errs = []  # (horizon_days, logit error) pooled over trophies/origins
        # Origins: every snapshot except the last few; targets: later snaps.
        for oi in range(1, len(snaps) - 1):
            for ti in range(oi + 1, len(snaps)):
                h_days = (snaps[ti][0] - snaps[oi][0]) / 86400.0
                if h_days <= 0:
                    continue
                xo = snap_x[oi]
                xt = snap_x[ti]
                for i in range(n):
                    r_o = snaps[oi][2][i]
                    r_t = snaps[ti][2][i]
                    if r_o is None or r_t is None:
                        continue
                    pred = logit_pct(r_o) + slopes[i] * (xt - xo) * PARAMS["drift_scale"]
                    pred = logit_pct(r_o) + min(PARAMS["drift_cap"], max(-PARAMS["drift_cap"], pred - logit_pct(r_o)))
                    errs.append((h_days, logit_pct(r_t) - pred))
        if len(errs) >= 40:
            # Fit sigma(h) = c * sqrt(h): c^2 = mean(err^2 / h).
            c2 = sum((e * e) / h for h, e in errs) / len(errs)
            ro_sigma = math.sqrt(c2) * math.sqrt(horizon_days)

    floor_sigma = math.sqrt(PARAMS["floor_a"] ** 2 + (PARAMS["floor_b"] * math.sqrt(horizon_days)) ** 2)
    sigma = max(floor_sigma, PARAMS["emp_damp"] * vol_sigma, PARAMS["emp_damp"] * ro_sigma)
    sigma = min(sigma, PARAMS["sigma_cap"])

    rows = []
    for i in range(n):
        drift = slopes[i] * dx * PARAMS["drift_scale"]
        drift = min(PARAMS["drift_cap"], max(-PARAMS["drift_cap"], drift))
        mu = last_logit[i] + drift
        qs = [round4(min(100.0, max(0.0, inv_logit_pct(mu + z * sigma)))) for z in Z]
        qs = sorted(qs)
        rows.append({"id": trophies[i].get("id"), "q": qs})
    return {"trophies": rows}


def forecast_case(case):
    try:
        out = model_forecast(case)
        # Validate shape before accepting.
        trophies = case.get("trophies") or []
        rows = out.get("trophies")
        if not isinstance(rows, list) or len(rows) != len(trophies):
            raise ValueError("bad shape")
        for row in rows:
            q = row["q"]
            if len(q) != 5 or any((not isinstance(v, (int, float))) or not math.isfinite(v) or v < 0 or v > 100 for v in q):
                raise ValueError("bad q")
            if any(q[j] > q[j + 1] for j in range(4)):
                raise ValueError("not monotone")
        return out
    except Exception:
        return fallback_forecast(case)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
