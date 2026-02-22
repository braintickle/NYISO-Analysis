"""
clean.py
--------
Standardizes raw NYISO DataFrames into clean, analysis-ready form.

NYISO column names vary slightly across datasets but follow patterns.
This module handles:
  - Datetime parsing and timezone localization (NYISO is US/Eastern)
  - Column renaming to snake_case
  - Zone name standardization
  - Duplicate removal
  - Basic outlier flagging
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Column name mappings per dataset ─────────────────────────────────────────
# NYISO uses slightly different column names across datasets.
# We normalize everything to a consistent schema.

COLUMN_MAPS = {
    "load_actual": {
        "Time Stamp":   "timestamp",
        "Name":         "zone",
        "Load":         "load_mw",
    },
    "load_forecast": {
        "Time Stamp":   "timestamp",
        "Zone Name":    "zone",
        "Load":         "load_forecast_mw",
    },
    "lmp_dayahead": {
        "Time Stamp":                        "timestamp",
        "Name":                              "zone",
        "LBMP ($/MWHr)":                     "lmp_total",
        "Marginal Cost Losses ($/MWHr)":     "lmp_losses",
        "Marginal Cost Congestion ($/MWHr)": "lmp_congestion",
    },
    "lmp_realtime": {
        "Time Stamp":                        "timestamp",
        "Name":                              "zone",
        "LBMP ($/MWHr)":                     "lmp_total",
        "Marginal Cost Losses ($/MWHr)":     "lmp_losses",
        "Marginal Cost Congestion ($/MWHr)": "lmp_congestion",
    },
    "fuel_mix": {
        "Time Stamp":       "timestamp",
        "Fuel Category":    "fuel_type",
        "Gen MW":           "gen_mw",
    },
}


def clean(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Master cleaning function — dispatches to dataset-specific logic.

    Args:
        df      : Raw DataFrame from nyiso_client
        dataset : Dataset name matching DATASET_PATHS keys

    Returns:
        Cleaned, typed DataFrame
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. Rename columns
    col_map = COLUMN_MAPS.get(dataset, {})
    df.rename(columns=col_map, inplace=True)

    # 2. Parse timestamp — NYISO timestamps are "MM/DD/YYYY HH:MM:SS" or ISO
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Localize to Eastern time (NYISO operates on US/Eastern)
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(
                "US/Eastern", ambiguous="infer", nonexistent="shift_forward"
            )

    # 3. Standardize zone names (strip whitespace, uppercase)
    if "zone" in df.columns:
        df["zone"] = df["zone"].str.strip().str.upper()

    # 4. Cast numeric columns
    numeric_cols = [c for c in df.columns if c not in ("timestamp", "zone", "fuel_type")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Drop full duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    dupes = before - len(df)
    if dupes > 0:
        logger.info(f"  Dropped {dupes} duplicate rows")

    # 6. Sort by timestamp
    if "timestamp" in df.columns:
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # 7. Flag outliers (values beyond 4 standard deviations) — don't remove, just flag
    # Only flag outliers on columns we actually care about
    cols_to_check = ["lmp_total", "lmp_losses", "lmp_congestion", "load_mw", "gen_mw"]
    numeric_cols = [c for c in cols_to_check if c in df.columns]
    
    for col in numeric_cols:
        if df[col].notna().sum() > 10:
            mean, std = df[col].mean(), df[col].std()
            df[f"{col}_outlier"] = (df[col] - mean).abs() > 4 * std

    logger.info(f"  Clean complete: {len(df)} rows, {df.shape[1]} columns")
    return df


def make_hourly(df: pd.DataFrame, value_col: str, group_cols: list = None) -> pd.DataFrame:
    """
    Resample sub-hourly data to clean hourly intervals.
    NYISO real-time data is 5-minute; day-ahead is already hourly.

    Args:
        df         : Cleaned DataFrame with 'timestamp' column
        value_col  : Column to aggregate (mean for prices, sum for generation)
        group_cols : Additional grouping columns (e.g., ['zone'])

    Returns:
        Hourly resampled DataFrame
    """
    df = df.copy()
    df = df.set_index("timestamp")

    if group_cols:
        df = (
            df.groupby(group_cols)
            .resample("h")[value_col]
            .mean()
            .reset_index()
        )
    else:
        df = df.resample("h")[value_col].mean().reset_index()

    return df


def pivot_zones(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Pivot a zone-level DataFrame into wide format for correlation / heatmap analysis.
    Each column becomes one NYISO zone.

    Example:
        timestamp | CAPITL | CENTRL | N.Y.C. | ...
    """
    return df.pivot_table(
        index="timestamp",
        columns="zone",
        values=value_col,
        aggfunc="mean"
    ).reset_index()
