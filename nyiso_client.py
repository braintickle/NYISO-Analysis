"""
nyiso_client.py
---------------
Handles all data fetching from the NYISO public CSV API.

NYISO publishes daily CSV files at predictable URLs — no API key required.
Data is updated ~hourly for real-time feeds and daily for historical.

Key datasets used in this project:
  - pal   : Actual Load (hourly, by zone)
  - isolf : Day-Ahead Load Forecast
  - damlbmp: Day-Ahead Market LMP (prices by zone)
  - rtlbmp: Real-Time LMP
  - fuel_mix: Hourly generation fuel mix
"""

import requests
import pandas as pd
import zipfile
import io
import os
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── NYISO Base URL ────────────────────────────────────────────────────────────
BASE_URL = "https://mis.nyiso.com/public/csv"

# Map dataset short names to their NYISO URL path segments
DATASET_PATHS = {
    "load_actual":      "pal",
    "load_forecast":    "isolf",
    "lmp_dayahead":     "damlbmp",
    "lmp_realtime":     "rtlbmp",
    "fuel_mix":         "rtfuelmix",
}

# The 11 NYISO zones
NYISO_ZONES = [
    "CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL",
    "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"
]


def build_url(dataset: str, date: datetime) -> str:
    """
    Build the NYISO CSV download URL for a given dataset and date.

    Most datasets use monthly ZIP files:
      https://mis.nyiso.com/public/csv/{path}/{YYYYMMDD}{path}_csv.zip

    Exception — damlbmp uses direct monthly CSV with _zone suffix:
      http://mis.nyiso.com/public/csv/damlbmp/{YYYYMMDD}damlbmp_zone_csv.zip
    """
    path = DATASET_PATHS[dataset]
    date_str = date.strftime("%Y%m%d")

    if dataset == "lmp_dayahead":
        # LMP day-ahead uses a different naming convention
        return f"http://mis.nyiso.com/public/csv/{path}/{date_str}{path}_zone_csv.zip"

    return f"{BASE_URL}/{path}/{date_str}{path}_csv.zip"


def fetch_day(dataset: str, date: datetime, retries: int = 3) -> pd.DataFrame:
    """
    Download and parse one month of NYISO data for a given dataset.

    Returns a DataFrame or empty DataFrame if the file is unavailable.
    """
    url = build_url(dataset, date)
    logger.info(f"Fetching {dataset} for {date.strftime('%Y-%m-%d')} ...")

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 404:
                logger.warning(f"  No data available: {url}")
                return pd.DataFrame()
            response.raise_for_status()

            # All NYISO datasets are zipped — extract all CSVs inside
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                frames = [pd.read_csv(z.open(f)) for f in csv_files]

            df = pd.concat(frames, ignore_index=True)
            logger.info(f"  ✓ {len(df)} rows fetched")
            return df

        except requests.exceptions.RequestException as e:
            logger.warning(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    logger.error(f"  All retries failed for {dataset} on {date.strftime('%Y-%m-%d')}")
    return pd.DataFrame()


def fetch_date_range(
    dataset: str,
    start: datetime,
    end: datetime,
    save_dir: str = None,
    pause: float = 0.5
) -> pd.DataFrame:
    """
    Fetch multiple months of data and concatenate into one DataFrame.

    IMPORTANT: NYISO publishes monthly ZIP files, not daily ones.
    Each ZIP contains all days in that month.
    We always request the 1st of each month to get the full monthly file.

    Args:
        dataset   : One of the keys in DATASET_PATHS
        start     : Start date (we snap to the 1st of this month)
        end       : End date (inclusive)
        save_dir  : If provided, caches monthly parquets locally
        pause     : Seconds to wait between requests

    Returns:
        Combined DataFrame for the full date range.
    """
    all_frames = []

    # Snap to the 1st of the start month
    current = start.replace(day=1)

    while current <= end:
        # Check local cache first to avoid re-downloading
        if save_dir:
            cache_path = os.path.join(
                save_dir,
                f"{dataset}_{current.strftime('%Y%m')}.parquet"
            )
            if os.path.exists(cache_path):
                logger.info(f"  Loading from cache: {cache_path}")
                all_frames.append(pd.read_parquet(cache_path))
                # Advance to next month
                month = current.month + 1 if current.month < 12 else 1
                year  = current.year if current.month < 12 else current.year + 1
                current = current.replace(year=year, month=month, day=1)
                continue

        df = fetch_day(dataset, current)

        if not df.empty:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                df.to_parquet(cache_path, index=False)
            all_frames.append(df)

        # Advance to next month
        month = current.month + 1 if current.month < 12 else 1
        year  = current.year if current.month < 12 else current.year + 1
        current = current.replace(year=year, month=month, day=1)
        time.sleep(pause)

    if not all_frames:
        logger.warning(f"No data returned for {dataset} between {start} and {end}")
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)
