---
title : "Compressor anomaly detection on industrial telemetry"
date : 2026-10-07
tags : [anomaly detection, LightGBM, predictive maintenance, Python, Streamlit]
header :
  image : ""
excerpt : "Anomaly detection, LightGBM, predictive maintenance"
---
[Source code](https://github.com/achafi/compressor_anomaly_detection)

# MetroPT-3 compressor anomaly detection
*The objective of this project is to detect unusual behaviour in a metro train's air compressor by learning what its pressure **should** be given the other sensors, and flagging the moments where the prediction misses by an unusual amount — a residual-based anomaly signal over 1.5 million rows of real industrial telemetry, calibrated out-of-fold instead of by hand-picked limits.*

## Introduction

When a compressor develops a fault such as an air leak, its sensors stop behaving the way they normally do. The naive fix — "TP2 must be between X and Y" — fails immediately, because expected values change with the machine's operating mode: the same reading can be perfectly normal in one state and wrong in another.

So this project takes a different route:

1. Learn what `TP2` (compressor pressure) *should* be, given the other sensors.
2. Compare that prediction with the real reading.
3. Flag the moments where the error is unusually large.

The sensors are physically coupled — pressures, valve states and motor current move together — so a regression model captures that normal coupling, and "anomaly" becomes a model error rather than a fixed physical limit. It is a typical predictive-maintenance workflow: turn raw telemetry into review signals an engineer can look at.

## Data

The [MetroPT-3 dataset from UCI](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset): about **1.5 million sensor readings** (pressures, motor current, oil temperature, valve states) recorded from the compressor of a metro train between February and September 2020.

| Item | Value |
|---|---|
| Raw rows | 1,516,948 |
| Nominal sampling | 10 seconds |
| Missing values | none in the raw CSV |
| 5-minute bins after aggregation | 50,782 |
| Training bins (default filter/window) | 1,141 |
| Test bins (21–28 Feb, filtered) | 474 |

## Pipeline

1. **Data input** — read from `data/raw/`; if the CSV is missing, the app downloads the UCI archive automatically (~208 MB).
2. **Preprocessing** — parse and sort timestamps, then aggregate into fixed time bins (1/5/10/15 min, 5 by default): means for continuous sensors, 0/1 conversion at a 0.5 cut for state signals. Empty bins dropped; the raw CSV is never modified.
3. **Operating-mode filter** — by default only *active-offloaded* bins are kept (`COMP=1`, `DV_eletric=0`, motor current 3.0–5.5 A), so training does not mix in shutdown states. Features are the 8 other sensors; missing values are filled with medians learned from training rows only.
4. **Regression** — a **LightGBM** regressor predicts `TP2` (default `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, adjustable live in the sidebar), validated with an expanding-window `TimeSeriesSplit(5)` plus a final untouched test week.
5. **Anomaly scoring** — threshold from out-of-fold residuals; every bin in the recording is scored, and consecutive flagged bins are grouped into "runs".
6. **Output** — dashboard metrics, plots, the top 20 anomalies, the run list, and one-click CSV export.

## The threshold: calibrated out-of-fold, not in-sample

For each bin, `residual = actual TP2 − predicted TP2`. When the model works normally the residual is small; a large residual means TP2 moved in a way the rest of the system does not explain.

The trap in anomaly detection is calibrating the threshold on training residuals — those are far too optimistic. Instead the project collects **out-of-fold** predictions from the expanding-window `TimeSeriesSplit`: every validation row is predicted by a model that never saw it, and the threshold becomes the **99th percentile of the absolute OOF residuals**:

```text
threshold = quantile(|OOF residual|, 0.99)   →  0.5154 bar (default settings)
alternative (OOF mean + 3σ)                  →  0.3763 bar
raw_anomaly_score = |residual| / threshold
is_anomaly        = raw_anomaly_score >= 1
```

The quantile is a slider (0.90–0.999), so stricter or looser sensitivity is one move instead of re-editing hard-coded bounds. The final test week is never used to pick the threshold.

## Results

Model quality with defaults (`random_state=42`, reproducible):

| Split | MAE | RMSE | R² |
|---|---|---|---|
| Train (in-sample) | 0.0200 | 0.0343 | 0.9995 |
| CV folds (5, mean ± std) | 0.0724 ± 0.0237 | 0.1201 ± 0.0365 | 0.9935 ± 0.0037 |
| OOF (pooled, 950 rows) | 0.0724 | 0.1245 | 0.9934 |
| Test (untouched week, 474 bins) | 0.0510 | 0.0878 | 0.9967 |

Anomaly summary: threshold **0.5154 bar**, **50,782 bins scored**, **7,089 flagged (13.96%)**, **3,888 runs**, maximum raw score 11.76.

Feature importance (split count): `H1` (2732), `Motor_current` (2024), `TP3` (1913), `DV_pressure` (1639), `Reservoirs` (692); the three binary state signals get 0 splits.

**Reading these numbers honestly** — of the 7,089 flagged bins, 6,956 (≈98%) fall *outside* the active-offloaded training mode: a 0.9% flag rate inside it versus 19.1% outside. Those flags are distribution shift, not confirmed faults, and the app says so in its own caption. There are no labels in this project, so there are no precision/recall numbers to brag about.

## The Streamlit dashboard

```bash
uv sync
uv run streamlit run app.py
```

| Tab | What it shows |
|---|---|
| Overview | Row/bin counts, threshold, flagged bins, column dictionary, data coverage |
| Exploration | Interactive signal plots and the Pearson correlation matrix |
| Model | Per-fold CV metrics, train/OOF/test metrics, actual-vs-predicted scatter, residual histogram, feature importance |
| Anomalies | Score timeline, actual vs predicted TP2 with flagged points, top 20 anomalies, runs, CSV download |

Things worth trying: drop the threshold quantile to `0.90` and watch flagged bins jump; disable the operating-mode filter and see the model trained on every state; change the aggregation window from 5 min to 1 min. Two companion notebooks cover exploration and the regression baseline.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python ≥ 3.10 (developed on 3.13), environment with uv |
| Data | pandas, NumPy |
| Modeling | LightGBM, scikit-learn (`TimeSeriesSplit`, MAE/RMSE/R²) |
| Dashboard | Streamlit, Plotly |
| Notebooks | Jupyter (nbclient/nbformat), committed without outputs |
| Data source | UCI MetroPT-3 (CC BY 4.0) |

## Limitations

- No labels are used: no precision, recall or F1 — UCI publishes known failure windows, but this project does not compare against them.
- The threshold comes from 950 OOF bins over 1–20 February in a single operating mode; the notebook itself calls the confidence "limited to moderate".
- Scores outside the training mode are suspect — they may be distribution shift rather than faults.
- Contemporaneous estimation, not forecasting: features and target share the same time bin.
- No tests, CI or persisted model artifacts yet; the app retrains on every run.

## Conclusion

Fixed thresholds fail across operating modes, single-channel limits ignore the rest of the machine, and in-sample thresholds are quietly wrong. Replacing all three with an OOF-calibrated residual score — plus a documented, leakage-conscious validation setup and an honest limitations section — is what this project is really about. It is an end-to-end predictive-maintenance workflow on real telemetry, with the caveats written down instead of buried.

[Source code](https://github.com/achafi/compressor_anomaly_detection)
