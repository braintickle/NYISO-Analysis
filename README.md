# NYISO Energy Data Analysis

## Overview

This project builds a **production-style data pipeline** that ingests live data from the [NYISO public CSV API](https://www.nyiso.com/custom-reports), cleans and stores it locally, and surfaces insights through an interactive **Streamlit dashboard**.

---

## Results

| Metric | Value |
|--------|-------|
| Date range covered | Jan–Mar 2024 (configurable) |
| Data freshness | Daily auto-refresh via GitHub Actions |
| Datasets ingested | Actual Load, Day-Ahead LMP, Fuel Mix |
| NYISO zones covered | All 11 zones |
| Storage format | Parquet (5–10x smaller than CSV) |


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

## Tech Stack

`Python 3.11` · `pandas` · `numpy` · `requests` · `pyarrow` · `plotly` · `streamlit`

---
