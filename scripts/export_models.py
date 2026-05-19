"""
export_models.py — Train XGBoost models and save as .joblib artifacts.

Reads from data/processed/ parquets (same as notebook 02).
Outputs to models/ in the repo root.

Usage:
    conda activate nyiso
    python scripts/export_models.py

Outputs:
    models/load_model_nyc.joblib    — XGBoost load model for N.Y.C.
    models/load_model_longil.joblib — XGBoost load model for LONGIL
    models/lmp_model_nyc.joblib     — XGBoost LMP model for N.Y.C. (rolling R1)
"""

import logging
import pathlib
import sys

import holidays
import joblib
import numpy as np
import pandas as pd
import requests
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT  = pathlib.Path(__file__).parent.parent
DATA_DIR   = REPO_ROOT / "data" / "processed"
MODEL_DIR  = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_YEAR = 2024
TEST_YEAR  = 2025

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
    url = "https://archive-api.open-meteo.com/v1/archive"
    r = requests.get(url, params={**WEATHER_PARAMS, "start_date": start_date, "end_date": end_date}, timeout=60)
    r.raise_for_status()
    return _parse_weather(r.json())


def fetch_weather_forecast_hist(start_date: str, end_date: str) -> pd.DataFrame:
    """Historical NWP forecast — eliminates leakage on test period."""
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
    df["lmp_lag_24h"]      = df["lmp_total"].shift(24)
    df["lmp_lag_48h"]      = df["lmp_total"].shift(48)
    df["lmp_lag_168h"]     = df["lmp_total"].shift(168)
    df["lmp_roll_mean_24h"] = df["lmp_total"].shift(1).rolling(24).mean()
    df["lmp_roll_std_24h"]  = df["lmp_total"].shift(1).rolling(24).std()
    df["lmp_spike_24h"]    = (df["lmp_lag_24h"] > df["lmp_lag_24h"].quantile(0.90)).astype(int)

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
    """Train one XGBoost model per zone (NYC + LONGIL). Returns {zone: model}."""
    models = {}
    for zone in ["N.Y.C.", "LONGIL"]:
        logger.info(f"  Training load model for {zone} ...")
        zone_df = df_features[df_features["zone"] == zone]
        train = zone_df[zone_df["timestamp"].dt.year == TRAIN_YEAR]
        test  = zone_df[zone_df["timestamp"].dt.year == TEST_YEAR]

        X_train, y_train = train[LOAD_FEATURE_COLS], train["load_mw"]
        X_test,  y_test  = test[LOAD_FEATURE_COLS],  test["load_mw"]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        mape  = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100
        logger.info(f"    {zone} MAPE: {mape:.2f}%  (n_test={len(X_test):,})")
        models[zone] = model

    return models


def train_lmp_model(df_lmp_features: pd.DataFrame):
    """
    Train rolling R1 LMP model (no gas features, retrain every 30 days).

    For the static export, we train on 2024 and evaluate on 2025 — the Lambda
    will use this artifact until a retrain is triggered manually.
    """
    logger.info("  Training LMP model (rolling R1, no gas features) ...")
    train = df_lmp_features[df_lmp_features["timestamp"].dt.year == TRAIN_YEAR]
    test  = df_lmp_features[df_lmp_features["timestamp"].dt.year == TEST_YEAR]

    X_train, y_train = train[LMP_FEATURE_COLS], train["lmp_total"]
    X_test,  y_test  = test[LMP_FEATURE_COLS],  test["lmp_total"]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mape  = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100
    mae   = np.mean(np.abs(y_test.values - preds))
    logger.info(f"    NYC LMP MAPE: {mape:.2f}%  MAE: {mae:.2f} $/MWh  (n_test={len(X_test):,})")

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading parquet files ...")
    df_load = pd.read_parquet(DATA_DIR / "load_actual.parquet")
    df_lmp  = pd.read_parquet(DATA_DIR / "lmp_dayahead.parquet")

    # Normalize column names
    df_lmp = df_lmp.rename(columns={"PTID": "ptid"})
    df_load = df_load.rename(columns={"PTID": "ptid"})

    # Hourly resample for load (5-min → hourly mean)
    df_load_hr = (
        df_load.groupby(["zone", pd.Grouper(key="timestamp", freq="h")])["load_mw"]
        .mean().reset_index()
    )
    # Keep only NYC + LONGIL for load model
    df_load_hr = df_load_hr[df_load_hr["zone"].isin(["N.Y.C.", "LONGIL"])]
    # LMP NYC only
    df_lmp_nyc = df_lmp[df_lmp["zone"] == "N.Y.C."].copy()

    logger.info(f"  load_actual (hourly): {len(df_load_hr):,} rows")
    logger.info(f"  lmp_dayahead (NYC):   {len(df_lmp_nyc):,} rows")

    # ── Fetch weather ──────────────────────────────────────────────────────────
    logger.info("Fetching weather data ...")
    weather_train = fetch_weather_archive("2024-01-01", "2024-12-31")
    logger.info("  2024 archive fetched")
    weather_test  = fetch_weather_forecast_hist("2025-01-01", "2025-12-31")
    logger.info("  2025 historical forecast fetched")
    weather = pd.concat([weather_train, weather_test], ignore_index=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info("Engineering load features ...")
    df_load_feat = engineer_load_features(df_load_hr, weather)

    logger.info("Training load models ...")
    load_models = train_load_models(df_load_feat)

    # Save as native XGBoost binary (.ubj) — version-stable across XGBoost major versions.
    # joblib-serialized sklearn XGBRegressor breaks when local XGBoost version (3.2.x)
    # differs from the Lambda layer version (3.0.x); .ubj avoids that entirely.
    load_models["N.Y.C."].get_booster().save_model(str(MODEL_DIR / "load_model_nyc.ubj"))
    logger.info(f"  Saved: {MODEL_DIR / 'load_model_nyc.ubj'}")
    load_models["LONGIL"].get_booster().save_model(str(MODEL_DIR / "load_model_longil.ubj"))
    logger.info(f"  Saved: {MODEL_DIR / 'load_model_longil.ubj'}")

    # ── LMP model ──────────────────────────────────────────────────────────────
    logger.info("Engineering LMP features ...")
    df_lmp_feat = engineer_lmp_features(df_lmp_nyc, weather)

    logger.info("Training LMP model ...")
    lmp_model = train_lmp_model(df_lmp_feat)

    # Overwrite the production .ubj used by Lambda inference (same filename as before).
    lmp_model.get_booster().save_model(str(MODEL_DIR / "lmp_r1_production.ubj"))
    logger.info(f"  Saved: {MODEL_DIR / 'lmp_r1_production.ubj'}")

    logger.info("=== export_models.py complete ===")
    logger.info(f"  Artifacts in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
