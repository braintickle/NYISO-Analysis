"""
inference.py — XGBoost model loading and prediction for Lambda.

Models are bundled with the Lambda deployment package under models/.
Run scripts/export_models.py locally to generate the .joblib files,
then include the models/ directory when deploying.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from features import LOAD_FEATURE_COLS, LMP_FEATURE_COLS

logger = logging.getLogger(__name__)

# Resolve model directory: /var/task/models/ in Lambda, ./models/ locally
_REPO_ROOT = Path(__file__).parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(_REPO_ROOT / "models")))


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(name: str):
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. "
            "Run scripts/export_models.py to generate model artifacts."
        )
    model = joblib.load(path)
    logger.info(f"  Loaded model: {name}")
    return model


_load_models_cache: dict = {}


def get_load_models() -> dict:
    """Return {zone: model} dict for load forecasting (cached after first load)."""
    global _load_models_cache
    if "load" not in _load_models_cache:
        _load_models_cache["load"] = {
            "N.Y.C.": _load_model("load_model_nyc.joblib"),
            "LONGIL": _load_model("load_model_longil.joblib"),
        }
    return _load_models_cache["load"]


def get_lmp_model():
    """Return LMP XGBoost model for N.Y.C. (cached after first load)."""
    global _load_models_cache
    if "lmp" not in _load_models_cache:
        _load_models_cache["lmp"] = _load_model("lmp_model_nyc.joblib")
    return _load_models_cache["lmp"]


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_load(zone: str, feature_df: pd.DataFrame) -> pd.Series:
    """Run load forecast for a single zone. Returns Series indexed by timestamp."""
    models = get_load_models()
    if zone not in models:
        raise ValueError(f"No load model for zone '{zone}'. Available: {list(models)}")
    model = models[zone]
    X = feature_df[LOAD_FEATURE_COLS].ffill().bfill()
    preds = model.predict(X)
    return pd.Series(preds, index=feature_df["timestamp"], name="load_forecast_mw")


def predict_lmp(feature_df: pd.DataFrame) -> pd.Series:
    """Run LMP forecast for N.Y.C. Returns Series indexed by timestamp."""
    model = get_lmp_model()
    X = feature_df[LMP_FEATURE_COLS].ffill().bfill()
    preds = model.predict(X)
    return pd.Series(preds, index=feature_df["timestamp"], name="lmp_forecast")


# ── Postgres writers ──────────────────────────────────────────────────────────

def write_load_forecast(
    conn,
    zone: str,
    timestamps: pd.DatetimeIndex,
    forecasts: pd.Series,
    model_version: Optional[str] = None,
) -> int:
    """Upsert load forecasts into load_forecast table. Returns rows inserted."""
    rows = [
        (ts, zone, float(fcst), None, None, model_version)
        for ts, fcst in zip(timestamps, forecasts)
        if not np.isnan(fcst)
    ]
    if not rows:
        return 0

    sql = """
        INSERT INTO load_forecast
            (timestamp, zone, load_forecast_mw, load_actual_mw, forecast_error_mw, model_version)
        VALUES %s
        ON CONFLICT (timestamp, zone) DO UPDATE SET
            load_forecast_mw = EXCLUDED.load_forecast_mw,
            model_version    = EXCLUDED.model_version
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows)
    conn.commit()
    logger.info(f"  load_forecast: {len(rows)} rows written for zone {zone}")
    return len(rows)


def write_lmp_forecast(
    conn,
    zone: str,
    timestamps: pd.DatetimeIndex,
    forecasts: pd.Series,
    model_version: Optional[str] = None,
) -> int:
    """Upsert LMP forecasts into lmp_forecast table. Returns rows inserted."""
    rows = [
        (ts, zone, float(fcst), None, None, model_version)
        for ts, fcst in zip(timestamps, forecasts)
        if not np.isnan(fcst)
    ]
    if not rows:
        return 0

    sql = """
        INSERT INTO lmp_forecast
            (timestamp, zone, lmp_forecast, lmp_actual, forecast_error, model_version)
        VALUES %s
        ON CONFLICT (timestamp, zone) DO UPDATE SET
            lmp_forecast  = EXCLUDED.lmp_forecast,
            model_version = EXCLUDED.model_version
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows)
    conn.commit()
    logger.info(f"  lmp_forecast: {len(rows)} rows written for zone {zone}")
    return len(rows)


def backfill_forecast_actuals(conn) -> None:
    """
    Fill in lmp_actual / forecast_error and load_actual_mw / forecast_error_mw
    for past forecast rows where actuals are now available.

    Called at the end of each LMP ingest run.
    """
    with conn.cursor() as cur:
        # LMP actuals
        cur.execute("""
            UPDATE lmp_forecast f
            SET
                lmp_actual     = a.lmp_total,
                forecast_error = f.lmp_forecast - a.lmp_total
            FROM lmp_dayahead a
            WHERE f.zone = a.zone
              AND f.timestamp = a.timestamp
              AND f.lmp_actual IS NULL
        """)
        lmp_updated = cur.rowcount

        # Load actuals (hourly average from 5-min data)
        cur.execute("""
            UPDATE load_forecast f
            SET
                load_actual_mw    = a.avg_load,
                forecast_error_mw = f.load_forecast_mw - a.avg_load
            FROM (
                SELECT
                    date_trunc('hour', timestamp) AS ts_hour,
                    zone,
                    AVG(load_mw) AS avg_load
                FROM load_actual
                GROUP BY 1, 2
            ) a
            WHERE f.zone = a.zone
              AND f.timestamp = a.ts_hour
              AND f.load_actual_mw IS NULL
        """)
        load_updated = cur.rowcount

    conn.commit()
    logger.info(f"  Backfilled actuals: {lmp_updated} LMP rows, {load_updated} load rows")
