"""
features.py — Feature engineering for Lambda inference.

Builds load and LMP feature matrices by:
  1. Fetching NWP weather forecast from Open-Meteo
  2. Pulling historical lags from Postgres
  3. Adding calendar and degree-day features
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import holidays
import numpy as np
import pandas as pd
import psycopg2
import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BALANCE_LOAD = 62.0   # degree-day balance point for load model (°F)
BALANCE_LMP  = 65.0   # degree-day balance point for LMP model (°F)

WEATHER_PARAMS = {
    "latitude":         40.7829,    # Central Park, NYC
    "longitude":        -73.9654,
    "hourly":           "temperature_2m,relative_humidity_2m,wind_speed_10m,apparent_temperature",
    "timezone":         "America/New_York",
    "temperature_unit": "fahrenheit",
}

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


# ── Weather ───────────────────────────────────────────────────────────────────

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


def fetch_weather_forecast(start_date: str, end_date: str) -> pd.DataFrame:
    """NWP forecast from Open-Meteo (up to 16 days ahead)."""
    url = "https://api.open-meteo.com/v1/forecast"
    r = requests.get(url, params={**WEATHER_PARAMS, "start_date": start_date, "end_date": end_date}, timeout=30)
    r.raise_for_status()
    return _parse_weather(r.json())


def fetch_weather_archive(start_date: str, end_date: str) -> pd.DataFrame:
    """ERA5 reanalysis from Open-Meteo archive (historical actuals)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    r = requests.get(url, params={**WEATHER_PARAMS, "start_date": start_date, "end_date": end_date}, timeout=30)
    r.raise_for_status()
    return _parse_weather(r.json())


# ── Calendar helpers ──────────────────────────────────────────────────────────

def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    us_holidays = holidays.US(years=df["timestamp"].dt.year.unique().tolist())
    df = df.copy()
    df["hour"]         = df["timestamp"].dt.hour
    df["dow"]          = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"]        = df["timestamp"].dt.month
    df["quarter"]      = df["timestamp"].dt.quarter
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["dow"] >= 5).astype(int)
    df["is_holiday"]   = df["timestamp"].dt.date.apply(
        lambda d: 1 if d in us_holidays else 0
    )
    return df


def _merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Left-join hourly weather onto df on floored timestamp."""
    w = weather.copy()
    w["ts_hour"] = w["timestamp"].dt.floor("h")
    w = w.drop_duplicates(subset=["ts_hour"])
    df = df.copy()
    df["ts_hour"] = df["timestamp"].dt.floor("h")
    df = df.merge(
        w[["ts_hour", "temp_f", "feels_like_f", "humidity_pct", "wind_speed_kmh"]],
        on="ts_hour", how="left"
    ).drop(columns=["ts_hour"])
    return df


# ── Postgres lag helpers ──────────────────────────────────────────────────────

def _fetch_load_history(conn, zone: str, oldest_needed: datetime) -> pd.Series:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT timestamp, load_mw FROM load_actual
            WHERE zone = %s AND timestamp >= %s
            ORDER BY timestamp
        """, (zone, oldest_needed))
        rows = cur.fetchall()
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    df = pd.DataFrame(rows, columns=["timestamp", "load_mw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")["load_mw"].sort_index()


def _fetch_lmp_history(conn, zone: str, oldest_needed: datetime) -> pd.Series:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT timestamp, lmp_total FROM lmp_dayahead
            WHERE zone = %s AND timestamp >= %s
            ORDER BY timestamp
        """, (zone, oldest_needed))
        rows = cur.fetchall()
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    df = pd.DataFrame(rows, columns=["timestamp", "lmp_total"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")["lmp_total"].sort_index()


def _lag(hist: pd.Series, ts: pd.Timestamp, hours: int) -> float:
    """Exact-hour lag lookup with 30-min fallback tolerance."""
    target = ts - pd.Timedelta(hours=hours)
    if target in hist.index:
        return hist[target]
    # Find nearest within 30 min
    candidates = hist.index[abs(hist.index - target) <= pd.Timedelta("30min")]
    if len(candidates):
        return hist[candidates[0]]
    return np.nan


# ── Feature builders ──────────────────────────────────────────────────────────

def build_load_features(
    conn,
    target_timestamps: pd.DatetimeIndex,
    zone: str,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build load feature matrix for inference.

    Fetches enough load history from Postgres to compute 168h lags,
    then joins weather and adds calendar/degree-day features.

    Returns DataFrame with LOAD_FEATURE_COLS (plus timestamp).
    """
    oldest_needed = target_timestamps.min() - pd.Timedelta(hours=192)
    hist = _fetch_load_history(conn, zone, oldest_needed)
    logger.info(f"  load history: {len(hist)} rows for zone {zone}")

    rows = []
    for ts in target_timestamps:
        past = hist[hist.index < ts]
        row = {
            "timestamp":          ts,
            "load_lag_24h":       _lag(hist, ts, 24),
            "load_lag_48h":       _lag(hist, ts, 48),
            "load_lag_168h":      _lag(hist, ts, 168),
            "load_roll_mean_24h": past.iloc[-24:].mean() if len(past) >= 24 else np.nan,
            "load_roll_std_24h":  past.iloc[-24:].std()  if len(past) >= 24 else np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _merge_weather(df, weather)
    df = _add_calendar(df)
    df["HDD"]       = (BALANCE_LOAD - df["temp_f"]).clip(lower=0)
    df["CDD"]       = (df["temp_f"] - BALANCE_LOAD).clip(lower=0)
    df["HDD_feels"] = (BALANCE_LOAD - df["feels_like_f"]).clip(lower=0)
    df["CDD_feels"] = (df["feels_like_f"] - BALANCE_LOAD).clip(lower=0)

    missing = [c for c in LOAD_FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"  Missing load features: {missing}")

    return df


def build_lmp_features(
    conn,
    target_timestamps: pd.DatetimeIndex,
    zone: str,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build LMP feature matrix for inference.

    Fetches DA LMP history from Postgres to compute 168h lags,
    then joins weather and adds calendar/degree-day features.

    Returns DataFrame with LMP_FEATURE_COLS (plus timestamp).
    """
    oldest_needed = target_timestamps.min() - pd.Timedelta(hours=192)
    hist = _fetch_lmp_history(conn, zone, oldest_needed)
    logger.info(f"  LMP history: {len(hist)} rows for zone {zone}")

    spike_threshold = hist.quantile(0.90) if len(hist) >= 100 else 100.0

    rows = []
    for ts in target_timestamps:
        past = hist[hist.index < ts]
        lag_24 = _lag(hist, ts, 24)
        row = {
            "timestamp":          ts,
            "lmp_lag_24h":        lag_24,
            "lmp_lag_48h":        _lag(hist, ts, 48),
            "lmp_lag_168h":       _lag(hist, ts, 168),
            "lmp_roll_mean_24h":  past.iloc[-24:].mean() if len(past) >= 24 else np.nan,
            "lmp_roll_std_24h":   past.iloc[-24:].std()  if len(past) >= 24 else np.nan,
            "lmp_spike_24h":      int(lag_24 > spike_threshold) if not np.isnan(lag_24) else 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _merge_weather(df, weather)
    df = _add_calendar(df)
    df["HDD"] = (BALANCE_LMP - df["temp_f"]).clip(lower=0)
    df["CDD"] = (df["temp_f"] - BALANCE_LMP).clip(lower=0)

    missing = [c for c in LMP_FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"  Missing LMP features: {missing}")

    return df
