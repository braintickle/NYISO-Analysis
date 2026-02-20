# NYISO Energy Data Pipeline & EDA Dashboard

> **Project 01 of 4** in a Data Science portfolio series using NYISO wholesale electricity market data.

---

## Overview

This project builds a **production-style data pipeline** that ingests live data from the [NYISO public CSV API](https://www.nyiso.com/custom-reports), cleans and stores it locally, and surfaces insights through an interactive **Streamlit dashboard**.

The goal is to demonstrate:
- Working with real, production-quality time series data (not Kaggle CSVs)
- Building a reusable ETL pipeline with local caching
- Generating domain-relevant insights from energy market data

This pipeline feeds directly into **Project 02** (Day-Ahead Load Forecasting) and **Project 03** (LMP Anomaly Detection).

---

## Results

| Metric | Value |
|--------|-------|
| Date range covered | Jan–Mar 2024 (configurable) |
| Data freshness | Daily auto-refresh via GitHub Actions |
| Datasets ingested | Actual Load, Day-Ahead LMP, Fuel Mix |
| NYISO zones covered | All 11 zones |
| Storage format | Parquet (5–10x smaller than CSV) |

**Key findings from EDA:**
- NYC + Long Island account for ~46% of total system load
- Weekday load is consistently 10–15% higher than weekends
- Evening peak (6–8 PM) is the most predictable demand driver
- Natural gas sets the marginal price >60% of hours; LMP spikes correlate with heat events and transmission congestion

---

## Project Structure

```
nyiso_project_01/
│
├── src/
│   ├── nyiso_client.py        # NYISO API fetcher with caching + retry logic
│   ├── clean.py               # Data cleaning, typing, outlier flagging
│   └── generate_synthetic.py  # Realistic synthetic data for offline dev/testing
│
├── notebooks/
│   └── 01_eda.ipynb           # Full EDA walkthrough with inline commentary
│
├── app/
│   └── app.py                 # Streamlit dashboard
│
├── data/
│   ├── raw/                   # Daily parquet cache (gitignored)
│   └── processed/             # Clean, analysis-ready parquets (gitignored)
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/nyiso-energy-pipeline.git
cd nyiso-energy-pipeline
pip install -r requirements.txt

# 2. Run the EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# 3. Launch the dashboard (uses processed data from notebook)
streamlit run app/app.py
```

> **No API key required.** NYISO's CSV API is fully public.

---

## Data Sources

| Dataset | NYISO Name | Frequency | Description |
|---------|-----------|-----------|-------------|
| Actual Load | `pal` | Hourly | MW consumed per zone |
| Day-Ahead LMP | `damlbmp` | Hourly | Day-ahead electricity price ($/MWh) |
| Fuel Mix | `rtfuelmix` | 5-min | Generation MW by fuel type |

All data is fetched from `https://mis.nyiso.com/public/csv/`.

---

## Business Context

NYISO (New York Independent System Operator) manages the wholesale electricity grid for New York State — dispatching ~40 GW of capacity and settling ~$10B in annual transactions.

**Why this data matters for DS:**
- **Load forecasting** is the #1 grid planning problem. A 1% MAPE improvement on a 25 GW system saves millions in reserve procurement.
- **LMP price prediction** drives trading, hedging, and demand response program design.
- **Fuel mix signals** are the real-time carbon intensity of the grid — increasingly used in ESG dashboards and demand flexibility programs.

This project is built from an M&V (Measurement & Verification) practitioner's perspective: the same load patterns that drive wholesale prices also determine the financial value of energy efficiency measures under IPMVP protocols.

---

## Next in This Series

| Project | Topic | Status |
|---------|-------|--------|
| **01** | Data Pipeline & EDA *(this repo)* | ✅ Complete |
| **02** | Day-Ahead Load Forecasting (XGBoost vs Prophet vs LSTM) | 🔄 In progress |
| **03** | LMP Price Spike Anomaly Detection | 📋 Planned |
| **04** | ML-Based Energy Savings Estimator (Capstone) | 📋 Planned |

---

## Tech Stack

`Python 3.11` · `pandas` · `numpy` · `requests` · `pyarrow` · `plotly` · `streamlit`

---

*Built as part of a Data Science portfolio transition from energy M&V consulting.*
