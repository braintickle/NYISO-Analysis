"""
export_models.py — Train XGBoost models and save as .ubj artifacts.

Reads from data/processed/ parquets (2024-2025) and Postgres (2026 holdout).
Outputs to models/ in the repo root.

Usage:
    conda activate nyiso
    python scripts/export_models.py

Outputs:
    models/load_model_nyc.ubj       — XGBoost load model for N.Y.C.  (train 2024, test 2025)
    models/load_model_longil.ubj    — XGBoost load model for LONGIL   (train 2024, test 2025)
    models/lmp_r1_production.ubj    — XGBoost LMP model for N.Y.C.   (train 2024+2025, test 2026)
"""

import logging
import os
import pathlib
import sys

import holidays
import numpy as np
import pandas as pd
import psycopg2
import requests
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT  = pathlib.Path(__file__).parent.parent
DATA_DIR   = REPO_ROOT / "data" / "processed"
MODEL_DIR  = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load model: train 2024, test 2025 (parquet covers both, MAPE 2.31% — no retrain needed)
LOAD_TRAIN_YEAR = 2024
LOAD_TEST_YEAR  = 2025

# LMP model: train 2024+2025 (covariate shift fix), holdout 2026 from Postgres
LMP_TRAIN_YEARS = [2024, 2025]
LMP_TEST_YEAR   = 2026

# ── Feature definitions (must match lambda/features.py exactly) ──────────────

LOAD_FEATURE_COLS = [
    "hour", "dow", "month", "quarter", "week_of_year",
    "is_weekend", "is_holiday",
    "temp_f", "feels_like_f", "humidity_pct", "wind_speed_kmh",
    "HDD", "CDD", "HDD_feels", "CDD_feels",
    "load_lag_24h", "load_lag_48h", "load_lag_168h",
    "load_roll_mean_24h", "load_roll_std_24h",
]

LMP_FEATURE_COLS = [
    "hour", "dow", "month", "quarter", "week_of_year",
    "is_weekend", "is_holiday",
    "temp_f", "feels_like_f", "humidity_pct", "wind_speed_kmh",
    "HDD", "CDD",
    "lmp_lag_24h", "lmp_lag_48h", "lmp_lag_168h",
    "lmp_roll_mean_24h", "lmp_roll_std_24h",
    "lmp_spike_24h",
]

BALANCE_LOAD = 62.0
BALANCE_LMP  = 65.0

WEATHER_PARAMS = {
    "latitude":         40.7829,
    "longitude":        -73.9654,
    "hourly":           "temperature_2m,relative_humidity_2m,wind_speed_10m,apparent_temperature",
    "timezone":         "America/New_York",
    "temperature_unit": "fahrenheit",
}


# ── Postgres helpers ──────────────────────────────────────────────────────────

def _load_db_url() -> str:
    """Read DATABASE_URL from env or .env file."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("DATABASE_URL not set. Add it to .env or export it.")


def fetch_2026_lmp_from_postgres() -> pd.DataFrame:
    """Fetch 2026 DA LMP for N.Y.C. from Postgres (not in local parquets)."""
    db_url = _load_db_url()
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, lmp_total
                FROM lmp_dayahead
                WHERE zone = 'N.Y.C.'
                  AND timestamp >= '2026-01-01'
                ORDER BY timestamp
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("No 2026 LMP data found in Postgres. Run the ingest Lambda first.")

    df = pd.DataFrame(rows, columns=["timestamp", "lmp_total"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Normalize timezone: Postgres returns UTC TIMESTAMPTZ
    ts = df["timestamp"]
    if ts.dt.tz is None:
        df["timestamp"] = ts.dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    else:
        df["timestamp"] = ts.dt.tz_convert("US/Eastern")
    df["zone"] = "N.Y.C."
    logger.info(f"  2026 LMP from Postgres: {len(df):,} rows  "
                f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})  "
                f"avg=${df['lmp_total'].mean():.2f}/MWh")
    return df


# ── Weather fetchers ──────────────────────────────────────────────────────────

def _parse_weather(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="shift_forward"
    )
    df = df.dropna(subset=["timestamp"]).drop(columns=["time"])
    return df.rename(columns={
        "temperature_2m":       "temp_f",
        "apparent_temperature": "feels_like_f",
        "relative_humidity_2m": "humidity_pct",
        "wind_speed_10m":       "wind_speed_kmh",
    })


def fetch_weather_archive(start_date: str, end_date: str) -> pd.DataFrame:
    """ERA5 reanalysis — used for training data (actual weather conditions)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    r = requests.get(url, params={**WEATHER_PARAMS, "start_date": start_date, "end_date": end_date}, timeout=60)
    r.raise_for_status()
    return _parse_weather(r.json())


def fetch_weather_forecast_hist(start_date: str, end_date: str) -> pd.DataFrame:
    """Historical NWP forecast — used for test/holdout data to avoid leakage.

    At inference time the model sees NWP forecasts, not ERA5 reanalysis.
    Evaluating on historical NWP gives a realistic picture of production accuracy.
    """
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    r = requests.get(url, params={**WEATHER_PARAMS, "start_date": start_date, "end_date": end_date}, timeout=60)
    r.raise_for_status()
    return _parse_weather(r.json())


# ── Feature engineering ───────────────────────────────────────────────────────

def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    us_holidays = holidays.US(years=df["timestamp"].dt.year.unique().tolist())
    df = df.copy()
    df["hour"]         = df["timestamp"].dt.hour
    df["dow"]          = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["quarter"]      = df["timestamp"].dt.quarter
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["dow"] >= 5).astype(int)
    df["is_holiday"]   = df["timestamp"].dt.date.apply(lambda d: 1 if d in us_holidays else 0)
    return df


def _merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    w["ts_hour"] = w["timestamp"].dt.floor("h")
    w = w.drop_duplicates(subset=["ts_hour"])
    df = df.copy()
    df["ts_hour"] = df["timestamp"].dt.floor("h")
    return df.merge(
        w[["ts_hour", "temp_f", "feels_like_f", "humidity_pct", "wind_speed_kmh"]],
        on="ts_hour", how="left"
    ).drop(columns=["ts_hour"])


def engineer_load_features(df_load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = _merge_weather(df_load, weather)
    df = _add_calendar(df)

    df["HDD"]       = (BALANCE_LOAD - df["temp_f"]).clip(lower=0)
    df["CDD"]       = (df["temp_f"] - BALANCE_LOAD).clip(lower=0)
    df["HDD_feels"] = (BALANCE_LOAD - df["feels_like_f"]).clip(lower=0)
    df["CDD_feels"] = (df["feels_like_f"] - BALANCE_LOAD).clip(lower=0)

    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)
    for zone in df["zone"].unique():
        mask = df["zone"] == zone
        load = df.loc[mask, "load_mw"]
        df.loc[mask, "load_lag_24h"]       = load.shift(24)
        df.loc[mask, "load_lag_48h"]       = load.shift(48)
        df.loc[mask, "load_lag_168h"]      = load.shift(168)
        df.loc[mask, "load_roll_mean_24h"] = load.shift(1).rolling(24).mean()
        df.loc[mask, "load_roll_std_24h"]  = load.shift(1).rolling(24).std()

    df = df.dropna(subset=LOAD_FEATURE_COLS).reset_index(drop=True)
    logger.info(f"  Load feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def engineer_lmp_features(df_lmp: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = _merge_weather(df_lmp, weather)
    df = _add_calendar(df)

    df["HDD"] = (BALANCE_LMP - df["temp_f"]).clip(lower=0)
    df["CDD"] = (df["temp_f"] - BALANCE_LMP).clip(lower=0)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["lmp_lag_24h"]       = df["lmp_total"].shift(24)
    df["lmp_lag_48h"]       = df["lmp_total"].shift(48)
    df["lmp_lag_168h"]      = df["lmp_total"].shift(168)
    df["lmp_roll_mean_24h"] = df["lmp_total"].shift(1).rolling(24).mean()
    df["lmp_roll_std_24h"]  = df["lmp_total"].shift(1).rolling(24).std()
    df["lmp_spike_24h"]     = (df["lmp_lag_24h"] > df["lmp_lag_24h"].quantile(0.90)).astype(int)

    df = df.dropna(subset=LMP_FEATURE_COLS).reset_index(drop=True)
    logger.info(f"  LMP feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── XGBoost params ─────────────────────────────────────────────────────────────

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20,
    eval_metric="rmse",
    verbosity=1,
)


# ── Model training ─────────────────────────────────────────────────────────────

def train_load_models(df_features: pd.DataFrame) -> dict:
    """Train one XGBoost model per zone (NYC + LONGIL). Train 2024, test 2025."""
    models = {}
    for zone in ["N.Y.C.", "LONGIL"]:
        logger.info(f"  Training load model for {zone} ...")
        zone_df = df_features[df_features["zone"] == zone]
        train = zone_df[zone_df["timestamp"].dt.year == LOAD_TRAIN_YEAR]
        test  = zone_df[zone_df["timestamp"].dt.year == LOAD_TEST_YEAR]

        X_train, y_train = train[LOAD_FEATURE_COLS], train["load_mw"]
        X_test,  y_test  = test[LOAD_FEATURE_COLS],  test["load_mw"]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        mape  = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100
        logger.info(f"    {zone} MAPE: {mape:.2f}%  (n_test={len(X_test):,})")
        models[zone] = model

    return models


def train_lmp_model(df_lmp_features: pd.DataFrame) -> XGBRegressor:
    """
    Train LMP model on 2024+2025 data, evaluate on 2026 holdout.

    Shows "before" MAPE (current 2024-only model on 2026 data) vs "after" MAPE,
    quantifying the improvement from adding 2025 training data.

    The covariate shift: 2024 avg LMP=$39/MWh → 2025=$65/MWh → 2026=$92/MWh.
    A 2024-only model chronically underpredicts because price lags and degree-day
    features from 2025-2026 are out of the model's learned distribution.
    """
    import xgboost as xgb

    logger.info("  Training LMP model (2024+2025 → 2026 holdout) ...")

    train = df_lmp_features[df_lmp_features["timestamp"].dt.year.isin(LMP_TRAIN_YEARS)]
    test  = df_lmp_features[df_lmp_features["timestamp"].dt.year == LMP_TEST_YEAR]

    logger.info(f"    Train rows: {len(train):,}  ({', '.join(str(y) for y in LMP_TRAIN_YEARS)})")
    logger.info(f"    Test rows:  {len(test):,}  (2026 holdout)")
    logger.info(f"    Train avg LMP: ${train['lmp_total'].mean():.2f}/MWh")
    logger.info(f"    Test  avg LMP: ${test['lmp_total'].mean():.2f}/MWh")

    X_train, y_train = train[LMP_FEATURE_COLS], train["lmp_total"]
    X_test,  y_test  = test[LMP_FEATURE_COLS],  test["lmp_total"]

    # ── "Before" MAPE: current 2024-only model on 2026 holdout ──────────────
    old_model_path = MODEL_DIR / "lmp_r1_production.ubj"
    if old_model_path.exists():
        old_booster = xgb.Booster()
        old_booster.load_model(str(old_model_path))
        old_preds = old_booster.predict(xgb.DMatrix(X_test))
        old_mape = np.mean(np.abs((y_test.values - old_preds) / y_test.values)) * 100
        old_mae  = np.mean(np.abs(y_test.values - old_preds))
        logger.info(f"    BEFORE (2024-only model) on 2026 holdout:")
        logger.info(f"      MAPE={old_mape:.2f}%  MAE=${old_mae:.2f}/MWh")
        logger.info(f"      avg_pred=${np.mean(old_preds):.2f}  avg_actual=${np.mean(y_test.values):.2f}")
    else:
        logger.warning("    No existing model found — skipping 'before' MAPE comparison.")

    # ── Train new model on 2024+2025 ────────────────────────────────────────
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    new_preds = model.predict(X_test)
    new_mape  = np.mean(np.abs((y_test.values - new_preds) / y_test.values)) * 100
    new_mae   = np.mean(np.abs(y_test.values - new_preds))
    logger.info(f"    AFTER (2024+2025 model) on 2026 holdout:")
    logger.info(f"      MAPE={new_mape:.2f}%  MAE=${new_mae:.2f}/MWh")
    logger.info(f"      avg_pred=${np.mean(new_preds):.2f}  avg_actual=${np.mean(y_test.values):.2f}")

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load parquet data ─────────────────────────────────────────────────────
    logger.info("Loading parquet files ...")
    df_load = pd.read_parquet(DATA_DIR / "load_actual.parquet")
    df_lmp  = pd.read_parquet(DATA_DIR / "lmp_dayahead.parquet")

    df_lmp  = df_lmp.rename(columns={"PTID": "ptid"})
    df_load = df_load.rename(columns={"PTID": "ptid"})

    # Hourly resample for load (5-min → hourly mean)
    df_load_hr = (
        df_load.groupby(["zone", pd.Grouper(key="timestamp", freq="h")])["load_mw"]
        .mean().reset_index()
    )
    df_load_hr = df_load_hr[df_load_hr["zone"].isin(["N.Y.C.", "LONGIL"])]
    df_lmp_nyc = df_lmp[df_lmp["zone"] == "N.Y.C."].copy()

    logger.info(f"  load_actual (hourly, parquet): {len(df_load_hr):,} rows  "
                f"years={sorted(df_load_hr['timestamp'].dt.year.unique().tolist())}")
    logger.info(f"  lmp_dayahead NYC (parquet):    {len(df_lmp_nyc):,} rows  "
                f"years={sorted(df_lmp_nyc['timestamp'].dt.year.unique().tolist())}")

    # ── Fetch 2026 LMP from Postgres for holdout ──────────────────────────────
    logger.info("Fetching 2026 LMP from Postgres ...")
    df_lmp_2026 = fetch_2026_lmp_from_postgres()

    # Merge parquet (2024-2025) + Postgres (2026)
    df_lmp_full = pd.concat([df_lmp_nyc, df_lmp_2026], ignore_index=True)
    df_lmp_full = (df_lmp_full
                   .drop_duplicates(subset=["timestamp"])
                   .sort_values("timestamp")
                   .reset_index(drop=True))
    logger.info(f"  LMP full dataset: {len(df_lmp_full):,} rows  "
                f"years={sorted(df_lmp_full['timestamp'].dt.year.unique().tolist())}")

    # ── Fetch weather ──────────────────────────────────────────────────────────
    logger.info("Fetching weather data ...")

    # Training weather: ERA5 reanalysis (actual conditions, no leakage concern for train set)
    weather_2024 = fetch_weather_archive("2024-01-01", "2024-12-31")
    logger.info("  2024 archive fetched")

    weather_2025 = fetch_weather_archive("2025-01-01", "2025-12-31")
    logger.info("  2025 archive fetched")

    # Holdout weather: historical NWP forecast (simulates real inference conditions — avoids leakage)
    # ERA5 archive lags ~5 days, so use NWP for the recent 2026 period
    weather_2026 = fetch_weather_forecast_hist("2026-01-01", "2026-05-18")
    logger.info("  2026 historical NWP fetched")

    weather_load = pd.concat([weather_2024, weather_2025], ignore_index=True)
    weather_lmp  = pd.concat([weather_2024, weather_2025, weather_2026], ignore_index=True)

    # ── Load model (train 2024, test 2025 — unchanged) ────────────────────────
    logger.info("Engineering load features ...")
    df_load_feat = engineer_load_features(df_load_hr, weather_load)

    logger.info("Training load models ...")
    load_models = train_load_models(df_load_feat)

    load_models["N.Y.C."].get_booster().save_model(str(MODEL_DIR / "load_model_nyc.ubj"))
    logger.info(f"  Saved: {MODEL_DIR / 'load_model_nyc.ubj'}")
    load_models["LONGIL"].get_booster().save_model(str(MODEL_DIR / "load_model_longil.ubj"))
    logger.info(f"  Saved: {MODEL_DIR / 'load_model_longil.ubj'}")

    # ── LMP model (train 2024+2025, holdout 2026) ────────────────────────────
    logger.info("Engineering LMP features ...")
    df_lmp_feat = engineer_lmp_features(df_lmp_full, weather_lmp)

    logger.info("Training LMP model ...")
    lmp_model = train_lmp_model(df_lmp_feat)

    lmp_model.get_booster().save_model(str(MODEL_DIR / "lmp_r1_production.ubj"))
    logger.info(f"  Saved: {MODEL_DIR / 'lmp_r1_production.ubj'}")

    logger.info("=== export_models.py complete ===")
    logger.info(f"  Artifacts in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
