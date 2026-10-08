---
title : "Industrial energy demand forecasting"
date : 2026-10-02
categories : [data-analytics]
tags : [time series, forecasting, Streamlit, Python]
header :
  image : ""
excerpt : "Time series, forecasting, energy"
---
[Source code](https://github.com/achafi/energy_demand_forcast)

# Industrial energy demand forecasting
*The objective of this project is to forecast a plant's electricity demand for the next day — all 96 fifteen-minute slots — using only information available at the moment the forecast is issued, and then wrap the whole thing in an application: explore the forecast, freeze it as a nomination, monitor the day against actuals, and simulate battery planning on top.*

## Introduction

Day-ahead forecasting looks trivial until you take leakage seriously. The moment you let "today's" measurements sneak into a feature that would not exist at midnight, or tune on a test period you already stared at during EDA, your metrics become fiction. This project is built around that problem: every notebook carries cutoff and leakage assertions, chronological expanding-window folds, and explicit statements about what has and has not been inspected.

The primary dataset is the **Steel Industry Energy Consumption** dataset from UCI (V E, Shin & Cho, 2021, CC BY 4.0): 15-minute sampling, `Usage_kWh` as the target. Later milestones extend the same methodology to an EWELD manufacturing customer (segment U144) and a Korean plant nomination demo.

## Notebooks

| # | Notebook | What it does |
|---|----------|--------------|
| 01 | `01_steel_energy_eda` | Data quality, demand distributions, calendar patterns, feature relationships, leakage considerations, chronological modeling handoff |
| 02 | `02_day_ahead_forecasting` | Midnight-cutoff day-ahead experiment: baselines vs four gradient-boosting candidates |
| 03 | `03_eweld_manufacturing_review` | Audit of 245 manufacturing customers, quality-based shortlist (U144 chosen) |
| 04 | `04_u144_energy_eda` | 172,320 readings (2016–2021) for the selected basic-metals customer |
| 05 | `05_u144_day_ahead_forecasting` | 14:00-previous-day forecasts, quarterly folds, 2020 chronological test |
| 06 | `06_korean_day_ahead_1400` | Nomination-demo timing: tomorrow's 96 slots issued at 14:00 today |

## Modeling approach

For the day-ahead experiment: forecast all 96 fifteen-minute slots at midnight using only earlier readings, requiring 28 complete days of history. Previous-day and previous-week baselines compete against four gradient-boosting candidates (with a holiday-feature comparison). Selection uses **equal-weight mean monthly MAE across four expanding-window folds** — train through June / validate July, through July / validate August, through August / validate September, through September / validate October — then refit through October and evaluate on November–December.

The notebook includes cutoff/leakage assertions, interval and daily error metrics, peak-demand diagnostics and interactive Plotly charts, and is candid that the November–December period had already been inspected in earlier experiments, so it is not an untouched holdout.

For the Korean nomination variant, the latest usable reading at issuance is labeled 13:45; two-day and weekly historical profiles, trailing statistics ending at issuance and calendar features feed the model, incomplete target days are excluded from training, and four July–October validation folds select the method.

Everything lands in `artifacts/`: `validation_metrics.csv`, per-fold boundaries and predictions, `test_metrics.csv`, the fitted `evaluation_hgb.joblib`, a 96-slot demo forecast for January 1, and a `metadata.json` recording features, timing assumptions, dataset checksum and environment versions.

## Streamlit forecast explorer

```bash
uv sync --locked
uv run --locked streamlit run app.py
```

The app has three tabs:

1. **Forecast & nomination** — a sidebar calendar selects an evaluated day (Nov 1 – Dec 31, 2018); choose the ML forecast or the weekly baseline, optionally compare both, and reveal actual demand for historical comparison. Daily energy, peak interval energy, peak time and the 96-row table update automatically. Saving the day's profile freezes the first nomination: changing methods or restarting the app does not overwrite it.
2. **Day monitoring** — advance a simulated clock from 00:00 to end of day; only completed slots reveal actuals (the 00:00 reading appears at 00:15). The view shows interval and cumulative comparisons, signed deviation, and the largest completed-interval deviation — positive meaning consumption above the nomination.
3. **Battery planning** — an optional simulation with synchronized demand, battery operation, cumulative cost and stored-energy plots.

Actual demand stays hidden by default in the forecast view, no nomination is ever submitted to a provider, and financial settlement is deliberately not estimated because real settlement prices and contract terms are unavailable.

## Battery simulation

The battery tab compares a fixed timer against a forecast-based rule (`rule_battery`): charge in the cheapest remaining slots before 21:00 (capped by capacity, power and optional grid headroom), discharge from the first expensive slot, and cycle only when the price spread beats round-trip efficiency. Both strategies must restore starting storage after 21:00.

The benchmark on 61 independent daily scenarios (43 meeting the common ending-energy requirement) is a nice honest result:

| Strategy | Total bill |
|---|---:|
| No battery | €20,717.72 |
| Fixed timer | €19,480.61 |
| Forecast-based control | €19,551.32 |
| Perfect foresight | €19,427.45 |

Under this illustrative tariff, the fixed timer captures most of the available savings — so the battery demo does **not** manufacture a strong forecasting use case, which is exactly the kind of negative result that usually gets quietly omitted.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13, managed with uv (`uv.lock` pinned) |
| Notebooks | Jupyter, executed with `uv run --locked` |
| Modeling | scikit-learn (HistGradientBoosting), joblib artifacts |
| App | Streamlit + Plotly (zoom, pan, hover) |
| Tests | `python -m unittest discover -s tests` |
| Data | UCI Steel Industry Energy Consumption (CC BY 4.0), EWELD segments |

## Limitations

- Results are historical, from one facility and one year; meter timing and reporting delay are assumed, not verified, and must be confirmed before live use.
- CO₂ units in the source metadata are ambiguous, and interval-boundary semantics are undocumented — calendar grouping uses recorded timestamps for that reason.
- The saved models are notebook-defined feature builders, not a standalone deployment pipeline.

## Conclusion

What makes this project worth showing is not a single MAE number — it is the discipline around it: leakage assertions inside the notebooks, expanding-window folds instead of random splits, artifacts that record their own provenance, and a battery experiment whose conclusion cut against the obvious story. Forecasting is easy to fake and hard to do honestly; this is my attempt at the second one.

[Source code](https://github.com/achafi/energy_demand_forcast)
