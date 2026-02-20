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

    NYISO URL pattern:
      https://mis.nyiso.com/public/csv/{path}/{YYYYMMDD}{path}_csv.zip

    Example:
      https://mis.nyiso.com/public/csv/pal/20240101pal_csv.zip
    """
    path = DATASET_PATHS[dataset]
    date_str = date.strftime("%Y%m%d")
    return f"{BASE_URL}/{path}/{date_str}{path}_csv.zip"


def fetch_day(dataset: str, date: datetime, retries: int = 3) -> pd.DataFrame:
    """
    Download and parse one day of NYISO data for a given dataset.

    Returns a DataFrame or empty DataFrame if the file is unavailable
    (e.g., future dates, weekends for some datasets).
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

            # NYISO zips contain one or more CSVs
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                frames = [pd.read_csv(z.open(f)) for f in csv_files]

            df = pd.concat(frames, ignore_index=True)
            logger.info(f"  ✓ {len(df)} rows fetched")
            return df

        except requests.exceptions.RequestException as e:
            logger.warning(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff

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
    Fetch multiple days of data and concatenate into one DataFrame.

    Args:
        dataset   : One of the keys in DATASET_PATHS
        start     : Start date (inclusive)
        end       : End date (inclusive)
        save_dir  : If provided, saves daily raw CSVs here for caching
        pause     : Seconds to wait between requests (be polite to NYISO servers)

    Returns:
        Combined DataFrame for the full date range.
    """
    all_frames = []
    current = start

    while current <= end:
        # Check local cache first to avoid re-downloading
        if save_dir:
            cache_path = os.path.join(
                save_dir,
                f"{dataset}_{current.strftime('%Y%m%d')}.parquet"
            )
            if os.path.exists(cache_path):
                logger.info(f"  Loading from cache: {cache_path}")
                all_frames.append(pd.read_parquet(cache_path))
                current += timedelta(days=1)
                continue

        df = fetch_day(dataset, current)

        if not df.empty:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                df.to_parquet(cache_path, index=False)
            all_frames.append(df)

        current += timedelta(days=1)
        time.sleep(pause)

    if not all_frames:
        logger.warning(f"No data returned for {dataset} between {start} and {end}")
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)
