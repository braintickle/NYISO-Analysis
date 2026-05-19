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

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from features import LOAD_FEATURE_COLS, LMP_FEATURE_COLS

logger = logging.getLogger(__name__)

# Resolve model directory: /var/task/models/ in Lambda, ./models/ locally
# In Lambda, __file__ is /var/task/inference.py so models live at /var/task/models/
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(Path(__file__).parent / "models")))


# ── Model loading ──────────────────────────────────────────────────────────────

_load_models_cache: dict = {}


def get_load_models() -> dict:
    """Return {zone: xgb.Booster} dict for load forecasting (cached after first load).

    Uses native XGBoost binary format (.ubj) — version-stable across XGBoost major
    versions, unlike joblib-serialized sklearn XGBRegressors which break when the
    local training version (3.2.x) differs from the Lambda layer version (3.0.x).
    """
    import xgboost as xgb
    global _load_models_cache
    if "load" not in _load_models_cache:
        boosters = {}
        for zone, fname in [("N.Y.C.", "load_model_nyc.ubj"), ("LONGIL", "load_model_longil.ubj")]:
            path = MODEL_DIR / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Model not found: {path}. "
                    "Run scripts/export_models.py to generate model artifacts."
                )
            booster = xgb.Booster()
            booster.load_model(str(path))
            logger.info(f"  Loaded model: {fname}")
            boosters[zone] = booster
        _load_models_cache["load"] = boosters
    return _load_models_cache["load"]


def get_lmp_model():
    """Return LMP XGBoost Booster for N.Y.C. (cached after first load).

    Uses the native XGBoost binary format (.ubj) so sklearn is not required
    at Lambda inference time.  lmp_r1_production.ubj is the rolling R1
    production model (MAPE 11.60%), trained without gas features.
    """
    import xgboost as xgb
    global _load_models_cache
    if "lmp" not in _load_models_cache:
        path = MODEL_DIR / "lmp_r1_production.ubj"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. "
                "Run scripts/export_models.py to generate model artifacts."
            )
        booster = xgb.Booster()
        booster.load_model(str(path))
        logger.info("  Loaded model: lmp_r1_production.ubj")
        _load_models_cache["lmp"] = booster
    return _load_models_cache["lmp"]


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_load(zone: str, feature_df: pd.DataFrame) -> pd.Series:
    """Run load forecast for a single zone. Returns Series indexed by timestamp."""
    import xgboost as xgb
    models = get_load_models()
    if zone not in models:
        raise ValueError(f"No load model for zone '{zone}'. Available: {list(models)}")
    booster = models[zone]
    X = feature_df[LOAD_FEATURE_COLS].ffill().bfill()
    dmat = xgb.DMatrix(X)
    preds = booster.predict(dmat)
    return pd.Series(preds, index=feature_df["timestamp"], name="load_forecast_mw")


def predict_lmp(feature_df: pd.DataFrame) -> pd.Series:
    """Run LMP forecast for N.Y.C. Returns Series indexed by timestamp."""
    import xgboost as xgb
    booster = get_lmp_model()
    X = feature_df[LMP_FEATURE_COLS].ffill().bfill()
    dmat = xgb.DMatrix(X)
    preds = booster.predict(dmat)
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
