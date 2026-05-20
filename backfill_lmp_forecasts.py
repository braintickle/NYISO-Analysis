"""
backfill_lmp_forecasts.py — Re-run LMP forecast backfill with retrained 2024+2025 model.

Overwrites lmp_forecast rows for Jul 2025 – yesterday using the new
lmp_r1_production.ubj (trained on 2024+2025, MAPE 15.13% on 2026 holdout).

Usage:
    conda activate nyiso
    cd /c/Users/rahi/nyiso/NYISO-analysis
    python backfill_lmp_forecasts.py
"""

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Environment setup ─────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).parent
# Point inference.py at the repo-root models/ directory
os.environ["MODEL_DIR"] = str(REPO_ROOT / "models")

# Make lambda/ importable
sys.path.insert(0, str(REPO_ROOT / "lambda"))

# ── Imports (after sys.path setup) ───────────────────────────────────────────

import pandas as pd
import psycopg2

from features import (
    fetch_weather_archive,
    fetch_weather_forecast,
    build_lmp_features,
)
from inference import (
    predict_lmp,
    write_lmp_forecast,
    backfill_forecast_actuals,
    _load_models_cache,
)

MODEL_VERSION = "lmp-2024-2025-v1"
START_DATE    = date(2025, 7, 1)
END_DATE      = date.today() - timedelta(days=1)   # up to yesterday


def main():
    logger.info(f"LMP backfill: {START_DATE} → {END_DATE}  model_version={MODEL_VERSION}")

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set — check .env")

    conn = psycopg2.connect(db_url)

    # Warm the model cache once so all days use the same loaded model
    from inference import get_lmp_model
    get_lmp_model()
    logger.info(f"  Model loaded from: {os.environ['MODEL_DIR']}")

    # Fetch weather for the full range in one shot (archive for older dates,
    # forecast API for recent 7 days where ERA5 reanalysis may lag)
    today = date.today()
    archive_cutoff = today - timedelta(days=7)
    archive_end    = min(END_DATE, archive_cutoff)
    recent_start   = archive_cutoff + timedelta(days=1)

    weather_parts = []
    if archive_end >= START_DATE:
        logger.info(f"  Fetching archive weather: {START_DATE} → {archive_end}")
        weather_parts.append(fetch_weather_archive(START_DATE.isoformat(), archive_end.isoformat()))
    if recent_start <= END_DATE:
        logger.info(f"  Fetching forecast weather: {recent_start} → {END_DATE}")
        weather_parts.append(fetch_weather_forecast(recent_start.isoformat(), END_DATE.isoformat()))

    weather = pd.concat(weather_parts, ignore_index=True)
    logger.info(f"  Weather rows: {len(weather):,}")

    lmp_total = 0
    errors    = []
    current   = START_DATE

    while current <= END_DATE:
        try:
            midnight_et = pd.Timestamp(current).tz_localize("US/Eastern")
            target_ts   = pd.date_range(start=midnight_et, periods=24, freq="h")

            feat  = build_lmp_features(conn, target_ts, "N.Y.C.", weather)
            preds = predict_lmp(feat)
            n     = write_lmp_forecast(conn, "N.Y.C.", target_ts, preds, MODEL_VERSION)
            lmp_total += n

        except Exception as exc:
            errors.append((current, str(exc)))
            logger.warning(f"  {current}: {exc}")

        if current.day == 1:
            logger.info(f"  Progress: {current}  lmp_total={lmp_total:,}")

        current += timedelta(days=1)

    logger.info(f"  Backfill complete: {lmp_total:,} lmp_forecast rows written")

    # Fill in lmp_actual / forecast_error where DA actuals are now available
    backfill_forecast_actuals(conn)

    conn.close()

    if errors:
        logger.warning(f"  Errors on {len(errors)} days:")
        for d, e in errors:
            logger.warning(f"    {d}: {e}")

    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
