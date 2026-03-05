# CLAUDE.md — NYISO Energy Market Analysis

## Project Overview

NYISO power markets analysis portfolio demonstrating data engineering, ML forecasting, optimization, and visualization. Built to showcase energy data science skills for power markets analyst roles (e.g., Modo Energy).

**GitHub:** https://github.com/braintickle/NYISO-analysis
**Environment:** `conda activate nyiso` | Python 3.11

## Repository Structure

```
nyiso-analysis/
├── src/
│   ├── nyiso_client.py        # NYISO public CSV API fetcher, caching, retry
│   ├── clean.py               # Cleaning, typing, outlier flagging
│   └── generate_synthetic.py  # Synthetic data for Streamlit Cloud deploy
├── notebooks/
│   ├── 01_eda.ipynb           # EDA: load, LMP, fuel mix, DA/RT spread
│   ├── 02_forecasting.ipynb   # Load + LMP forecasting (SARIMA, Prophet, XGBoost)
│   └── 03_bess_optimization.ipynb  # BESS dispatch LP optimizer
├── app/
│   └── app.py                 # Streamlit dashboard
├── data/
│   ├── raw/                   # Cached monthly parquets (gitignored)
│   └── processed/             # Clean analysis-ready parquets (gitignored)
├── requirements.txt
└── README.md
```

## Data Pipeline

Four datasets from NYISO public CSV API (no key required):

| Dataset        | NYISO endpoint | Resolution | File                          |
|----------------|---------------|------------|-------------------------------|
| load_actual    | pal           | 5-min      | data/processed/load_actual.parquet |
| lmp_dayahead   | damlbmp       | hourly     | data/processed/lmp_dayahead.parquet |
| lmp_realtime   | rtlbmp        | hourly     | data/processed/lmp_realtime.parquet |
| fuel_mix       | rtfuelmix     | 5-min      | data/processed/fuel_mix.parquet |

Also generated:
- `data/processed/lmp_forecast_2025.parquet` — XGBoost DA LMP forecasts
- `data/processed/system_load.parquet`

**Date range:** 2024–2025 full years
**Zones:** All 11 NYISO zones. Primary analysis on N.Y.C. (Zone J) and LONGIL (Zone K).
**Query engine:** DuckDB for all parquet access (predicate pushdown for memory efficiency).

## Key Packages

pandas, numpy, duckdb, xgboost, pulp, prophet, statsmodels, plotly, streamlit, mlflow, shap, scikit-learn, holidays, openmeteo-requests

## Project Results Summary

### 01 — EDA
- DA vs RT LMP spread: mean $0.81/MWh, std $50.93/MWh (NYC)
- Hour 9: most negative spread (RT < DA) — morning demand overestimated
- Hour 17: most positive spread (RT > DA) — evening peak underestimated
- Natural gas sets marginal price majority of hours
- NYC LMP highest due to transmission-constrained load pocket

### 02 — Forecasting
**Load (NYC + LONGIL):**
- XGBoost MAPE: 2.31% (industry benchmark 2–4%)
- SARIMA: 39.4%, Prophet: 79.9% (baselines)
- Train: 2024, Test: 2025 (temporal split, no leakage)
- Most important feature: `lag_168h` (same hour last week)

**LMP (NYC DA):**
- XGBoost MAPE: 12.89%
- Mean error: +5.19 $/MWh (systematic underprediction)
- Max error: +229 $/MWh (extreme spikes missed)

**Feature set (20 features):**
- Calendar: hour, dow, month, quarter, week_of_year, is_weekend, is_holiday
- Weather: temp_f, feels_like_f, humidity_pct, wind_speed_kmh
- Degree days: HDD, CDD (balance point 65°F, IPMVP standard)
- Lags: load_lag_24h, load_lag_48h, load_lag_168h
- Rolling: load_roll_mean_24h, load_roll_std_24h
- LMP-specific: lmp_lag_24h/48h/168h, lmp_roll_mean/std_24h, lmp_spike_24h

**MLflow:** experiment tracking at `../mlruns`

### 03 — BESS Optimization
**Battery specs:** 100 MW / 400 MWh (4h duration), 85% RTE, 5–95% SOC, charge derating above 80%

| Strategy | Annual Revenue |
|----------|---------------|
| Naive dispatch | $5,572,918 |
| LP perfect foresight (energy only) | $9,062,076 (+62.6%) |
| LP perfect foresight + ICAP | $14,614,618 |
| LP forecasted prices + ICAP | $14,064,097 |

- Forecast error penalty: $550,522 (7.6% of perfect)
- Basis P&L (RT vs DA settlement): $681,199
- Optimizer: PuLP LP with CBC solver, Zone J, full year 2025
- ICAP: real 2025 NYC Zone J strip auction prices, dynamic commitment balancing volatility signal vs clearing price

## Active Project: 04 — Live Dashboard

**Goal:** Real-time NYISO dashboard that turns analysis into a product.

### Architecture

```
NYISO Public CSV API
      ↓ (EventBridge cron: 5min for load/fuel, 1hr for LMP)
AWS Lambda
  → fetch → clean → compute features → XGBoost inference
      ↓
Supabase Postgres (free tier, 500MB)
  tables: load_actual, lmp_dayahead, lmp_realtime, fuel_mix,
          load_forecast, lmp_forecast, bess_dispatch
      ↓
Streamlit on AWS ECS Fargate
  → queries Postgres → renders dashboard
```

### Cloud Data Layer — Supabase Postgres
- Migrate from local DuckDB/parquet to Supabase Postgres
- Swap DuckDB queries for psycopg2/SQLAlchemy against Postgres
- Schema mirrors existing parquet structure
- Connection via `DATABASE_URL` env var (secrets in AWS Secrets Manager)

### Ingestion — AWS Lambda + EventBridge
- Refactor `src/nyiso_client.py` into Lambda handler
- Schedule: every 5 min (load_actual, fuel_mix), every 1 hr (lmp_dayahead, lmp_realtime)
- Lambda computes lag/rolling features from recent Postgres rows
- Trained XGBoost model artifact packaged with Lambda (static; retrain locally, redeploy)
- Lambda writes forecast rows to `load_forecast` / `lmp_forecast` tables
- BESS dispatch recommendation generated from latest DA LMP forecast

### Deployment — AWS ECS Fargate
- Dockerized Streamlit app
- Container image pushed to ECR
- Fargate service with public URL

### Dashboard Components
1. Current actual load vs XGBoost forecast
2. Live LMP by zone (heatmap)
3. DA/RT spread tracker
4. BESS dispatch recommendation for today ("based on today's DA prices, optimal schedule is...")
5. YTD revenue tracker vs naive benchmark

## Conventions

- All timestamps are Eastern (NYISO native timezone)
- DuckDB for local/notebook parquet queries; Postgres for cloud/dashboard
- Parquet files are gitignored; `src/nyiso_client.py` regenerates locally
- Notebooks use Plotly for interactive charts (nbviewer for rendering)
- Lambda secrets (DB connection string) stored in AWS Secrets Manager
- Docker image tagged with git SHA for traceability

## TODO

### Project 04 — Live Dashboard (current)
- [ ] Design Supabase Postgres schema from existing parquet structure
- [ ] Write migration script: load processed parquets → Postgres tables
- [ ] Refactor `nyiso_client.py` into AWS Lambda handler
- [ ] Package trained XGBoost models as Lambda layer / bundled artifact
- [ ] Build feature computation logic for Lambda (lags, rolling stats from Postgres)
- [ ] Set up EventBridge cron rules (5min load/fuel, 1hr LMP)
- [ ] Build BESS dispatch recommendation logic for daily DA prices
- [ ] Dockerize Streamlit app
- [ ] Deploy to ECS Fargate with public URL
- [ ] Build dashboard components (load vs forecast, LMP heatmap, spread tracker, BESS dispatch, YTD revenue)

### Forecasting Improvements
- [ ] Add natural gas futures prices to LMP features
- [ ] Add solar generation feature for duck curve

### Other
- [ ] Fix Plotly rendering in GitHub notebook viewer
- [ ] Project 05: Electrification structural break detection (heat pumps, EVs)
