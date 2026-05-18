# CLAUDE.md — NYISO Energy Market Analysis

# Working style: Always explain code after writing it. Walk through the logic, 
# design decisions, and tradeoffs. I need to understand everything in this 
# codebase well enough to explain it in a technical interview.

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
│   ├── migrate_to_postgres.py # One-time migration: parquets → Supabase Postgres
│   └── generate_synthetic.py  # Synthetic data for Streamlit Cloud deploy
├── lambda/                    # AWS Lambda functions (Project 04)
│   ├── handler.py             # Entry points: ingest_load_fuel, ingest_lmp, run_bess_dispatch
│   ├── ingest.py              # NYISO fetch + clean + Postgres insert
│   ├── features.py            # Feature engineering (lags, rolling, weather, calendar)
│   ├── inference.py           # XGBoost model loading + prediction + forecast writer
│   ├── bess_dispatch.py       # PuLP LP optimizer + Postgres writer
│   └── requirements.txt       # Lambda-specific dependencies
├── scripts/
│   └── export_models.py       # Train XGBoost models + save .joblib artifacts to models/
├── models/                    # Trained model artifacts (gitignored); run export_models.py
│   ├── load_model_nyc.joblib
│   ├── load_model_longil.joblib
│   └── lmp_model_nyc.joblib
├── sql/
│   └── schema.sql             # Supabase Postgres DDL (8 tables)
├── notebooks/
│   ├── 01_eda.ipynb           # EDA: load, LMP, fuel mix, DA/RT spread
│   ├── 02_forecasting.ipynb   # Load + LMP forecasting (SARIMA, Prophet, XGBoost)
│   └── 03_bess_optimization.ipynb  # BESS dispatch LP optimizer
├── app/
│   └── app.py                 # Streamlit dashboard (Postgres-first, parquet fallback)
├── data/
│   ├── raw/                   # Cached monthly parquets (gitignored)
│   └── processed/             # Clean analysis-ready parquets (gitignored)
├── Dockerfile                 # Streamlit app for ECS Fargate
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

**LMP (NYC DA) — static model:**
- XGBoost MAPE: 12.89% (static, trained on 2024 only)
- Mean error: +5.19 $/MWh (systematic underprediction)
- Max error: +229 $/MWh (extreme spikes missed)

**LMP rolling day-ahead simulation (Section 15):**
- R1 (no gas, retrain every 30 days): MAPE=11.60% (-1.76pp vs static)
- R2 (+gas_price_lag1d): MAPE=12.30% — gas still hurts (+0.70pp vs R1)
- Gas degradation shrinks from +4.71pp (static) → +0.70pp (rolling): covariate shift
  partially mitigated, but Jan–Mar still affected (trailing window anchored in 2024 regime)
- Production model: R1 rolling, no gas features

**Gas ablation (Section 14):**
- S2 +price_level: 18.07% — covariate shift (2024 mean $2.25 → 2025 mean $3.54/MMBtu)
- S3 +pct_change: 13.34% — stationary but negligible signal (existing lags cover it)
- Conclusion: gas features do not improve hourly LMP MAPE given existing price history lags

**Feature set (19 LMP features):**
- Calendar: hour, dow, month, quarter, week_of_year, is_weekend, is_holiday
- Weather: temp_f, feels_like_f, humidity_pct, wind_speed_kmh
- Degree days: HDD, CDD (balance point 65°F, IPMVP standard)
- LMP lags: lmp_lag_24h, lmp_lag_48h, lmp_lag_168h
- LMP rolling: lmp_roll_mean_24h, lmp_roll_std_24h, lmp_spike_24h

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
- [x] Design Supabase Postgres schema from existing parquet structure (`sql/schema.sql`)
- [x] Write migration script: load processed parquets → Postgres tables (`src/migrate_to_postgres.py`)
- [x] Refactor `nyiso_client.py` into AWS Lambda handler (`lambda/handler.py`, `lambda/ingest.py`)
- [x] Package trained XGBoost models as Lambda artifact (`scripts/export_models.py` → `models/*.joblib`)
- [x] Build feature computation logic for Lambda (`lambda/features.py`)
- [x] Build BESS dispatch recommendation logic for daily DA prices (`lambda/bess_dispatch.py`)
- [x] Dockerize Streamlit app (`Dockerfile`)
- [x] Build dashboard components (load vs forecast, LMP heatmap, spread tracker, BESS dispatch) in `app/app.py`
- [ ] **Run `scripts/export_models.py` to generate .joblib model artifacts**
- [ ] **Run `src/migrate_to_postgres.py` to populate Supabase (needs DATABASE_URL)**
- [ ] Set up EventBridge cron rules (5min load/fuel, 1hr LMP) in AWS Console / CDK
- [ ] Deploy Lambda to AWS (zip lambda/ + models/ → upload)
- [ ] Deploy to ECS Fargate with public URL

### Forecasting Improvements
- [ ] Add natural gas futures prices to LMP features
- [ ] Add solar generation feature for duck curve

### Other
- [ ] Fix Plotly rendering in GitHub notebook viewer
- [ ] Project 05: Electrification structural break detection (heat pumps, EVs)
