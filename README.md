# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/AdelineZiv24/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/AdelineZiv24/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Built for the make.fun bounty [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

`main.py` (Python 3.13, standard library only, deterministic, no network) reads
one case file — every Exophase read of GTA VI's PS5 trophy page recorded before
the case's checkpoint — and writes, for each of the game's trophies, five
quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate at the case's target,
scored by pinball loss on the logit scale.

## How it forecasts

The recorded page only changes at Exophase refreshes, so the history is first
reduced to its distinct snapshots. For each trophy the model fits a slope of
logit(rate) against log-time — x = ln(1 + hours since 2026-11-19T00:00Z) — over
the most recent 30% of the observed span, shrinks that slope hierarchically
toward the mean slope of the trophy's tier and of the whole field, and anchors
the median at the last observed logit plus the damped, capped drift
(drift scale 0.32, cap ±1.2 logit). Quantiles are inv_logit(μ + z_τ·σ) with σ
from a structural floor, σ = √(0.15² + (0.20·√days)²), lightly hedged by the
history's own realised volatility and rolling-origin backtests, capped at 1.4
(about 0.45 at the final's 4.52-day horizon). Any degenerate history — flat,
single-read, missing — falls back to persistence with a wide, honest spread.

The constants were tuned on a simulation study of 200 synthetic launch worlds
per suite (saturating per-trophy adoption in log-time, a tracked-player pool
growing from hundreds to tens of thousands, the page frozen between refreshes
every 2–6 hours), cut at 2.2 days with the final's 6,510-minute horizon and
scored with the contest's own pinball loss: the shipped model averages 0.111
against the persistence baseline's 0.242 and the fitted-curve baseline's 0.262
on a held-out seed suite, beating both at every one of the five quantile
levels.

## Running it locally

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

with the case files (`*.in`) in `input/`. Each run writes one `<id>.out` per
case, byte-identical across runs.
