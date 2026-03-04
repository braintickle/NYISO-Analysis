# NYISO Energy Market Analysis

## Overview

A **production-style energy data science portfolio** built on live NYISO market data. This project covers the full analytics stack — from raw data ingestion to load forecasting to battery storage optimization — using real market prices, capacity auction results, and weather data.

Built as a ground-up learning project to understand NYISO market structure, power price dynamics, and battery storage economics.

---

## Results

### Project 01 — Data Pipeline & EDA

| Metric | Value |
|--------|-------|
| Date range | 2024–2025 (configurable) |
| Datasets | Actual Load, Day-Ahead LMP, Real-Time LMP, Fuel Mix |
| Zones covered | All 11 NYISO zones |
| Storage format | Parquet (5–10x smaller than CSV) |
| Dashboard | Streamlit app with zone selector, KPI cards, 4 interactive charts |

**Key finding:** DA vs RT LMP spread in NYC has a near-zero mean ($0.81/MWh) but standard deviation of $51/MWh — BESS value comes from capturing tail events, not average spreads. Summer 2025 showed significantly larger spread volatility than 2024, consistent with more extreme heat events.

---

### Project 02 — Day-Ahead Load Forecasting

| Model | NYC MAPE | Notes |
|-------|----------|-------|
| SARIMA | 39.4% | Statistical baseline, no weather features |
| Prophet | 79.9% | Trend extrapolation failure across year-long horizon |
| XGBoost | **2.31%** | Weather + lag features, within industry benchmark |

**Key finding:** `lag_168h` (same hour last week) was the most important feature — load is highly autocorrelated at weekly intervals. HDD/CDD features (from IPMVP M&V methodology) were the most important weather variables. SARIMA and Prophet failures demonstrate why explicit feature engineering outperforms automatic decomposition for energy data.

**Next steps:**
- Add natural gas futures prices as a feature (gas sets marginal price ~60-70% of hours)
- Add solar generation to capture duck curve midday price suppression
- Replace perfect price foresight in Project 03 with forecasted prices from this model — evaluate how forecast error propagates to BESS revenue

---

### Project 03 — BESS Dispatch Optimization

**Setup:** 100MW / 400MWh battery in NYC Zone J, optimized across full year 2025 using real NYISO day-ahead LMP and ICAP strip auction prices.

#### Energy Arbitrage Only (LP Optimization)

| Strategy | Annual Revenue |
|----------|---------------|
| Naive (charge 4 cheapest, discharge 4 most expensive hours) | $5,572,918 |
| LP Optimal | $9,062,076 |
| **LP improvement over naive** | **62.6%** |

#### Co-Optimized: Energy Arbitrage + ICAP

| Revenue Stream | DA Plan | RT Actual |
|---------------|---------|-----------|
| Energy arbitrage | — | — |
| ICAP (NYC Zone J) | $2,132,638 | $2,132,638 |
| **Total** | — | **$11,194,714** |
| Basis P&L (RT vs DA) | — | +$681,199 |
| Basis P&L as % of DA energy | — | 9.1% |

**Key findings:**
- LP optimization outperforms naive dispatch by 62.6% — quantifies the value of constrained optimization over heuristics
- ICAP contributed ~19% of total revenue using real 2025 NYC Zone J strip auction prices ($6.15–$13.89/kW-month)
- NYC ICAP prices are 3x rest-of-state — location premium from transmission-constrained load pocket
- Positive basis P&L of $681K means RT prices were favorable vs DA plan — summer 2025 RT spikes worked in the battery's favor
- Dynamic ICAP commitment model balances two competing signals: price volatility (high vol → reduce ICAP, keep MW for arbitrage) and ICAP clearing price (high price → increase ICAP commitment)

**Model assumptions and limitations:**
- DA energy optimization uses perfect price foresight — real operations use forecasted prices
- Charge rate degradation modeled as linearized CC/CV approximation above 80% SOC
- Ancillary services (regulation, spinning reserve) not included — would add meaningful revenue
- Battery cycle degradation costs not modeled
- ICAP co-optimization enforces SOC reservation during peak hours (2–6PM) to honor capacity obligations

**Next step:** Replace perfect DA price foresight with XGBoost forecasted prices from Project 02 to evaluate how forecast error propagates into dispatch decisions and annual revenue — a realistic backtest of the full pipeline.

---

## Project Structure

```
nyiso-analysis/
│
├── src/
│   ├── nyiso_client.py        # NYISO API fetcher with caching + retry logic
│   ├── clean.py               # Data cleaning, typing, outlier flagging
│   └── generate_synthetic.py  # Realistic synthetic data for offline dev/testing
│
├── notebooks/
│   ├── 01_eda.ipynb           # EDA — load patterns, LMP analysis, DA/RT spread
│   ├── 02_forecasting.ipynb   # Load forecasting — SARIMA vs Prophet vs XGBoost
│   └── 03_bess_optimization.ipynb  # BESS dispatch optimizer with ICAP co-optimization
│
├── app/
│   └── app.py                 # Streamlit dashboard (live demo uses synthetic data)
│
├── data/
│   ├── raw/                   # Monthly parquet cache (gitignored)
│   └── processed/             # Clean, analysis-ready parquets (gitignored)
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/braintickle/NYISO-analysis.git
cd NYISO-analysis
pip install -r requirements.txt

# 2. Run EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# 3. Run load forecasting
jupyter notebook notebooks/02_forecasting.ipynb

# 4. Run BESS optimization
jupyter notebook notebooks/03_bess_optimization.ipynb

# 5. Launch dashboard (uses synthetic data in deployed version)
streamlit run app/app.py
```

> **No API key required.** NYISO's CSV API is fully public.

---

## Data Sources

| Dataset | NYISO Name | Frequency | Description |
|---------|-----------|-----------|-------------|
| Actual Load | `pal` | 5-min | MW consumed per zone |
| Day-Ahead LMP | `damlbmp` | Hourly | Day-ahead electricity price ($/MWh) |
| Real-Time LMP | `rtlbmp` | Hourly | Real-time electricity price ($/MWh) |
| Fuel Mix | `rtfuelmix` | 5-min | Generation MW by fuel type |
| Weather | Open-Meteo | Hourly | Temperature, humidity, wind (NYC Central Park) |
| ICAP Prices | NYISO Strip Auctions | Monthly | Capacity market clearing prices ($/kW-month) |

All NYISO data fetched from `https://mis.nyiso.com/public/csv/`. Weather from Open-Meteo (no API key required).

---

## Tech Stack

`Python 3.11` · `pandas` · `numpy` · `duckdb` · `requests` · `pyarrow` · `plotly` · `streamlit` · `xgboost` · `statsmodels` · `prophet` · `pulp` · `scikit-learn` · `shap` · `mlflow` · `holidays`