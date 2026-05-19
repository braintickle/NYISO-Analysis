"""
ingest.py — NYISO data fetching and Postgres ingestion for Lambda.

Fetches the current and previous month's data from NYISO public CSV API
and inserts new rows into Postgres (idempotent via ON CONFLICT DO NOTHING).

NYISO publishes monthly ZIP files keyed by the first of the month:
  http://mis.nyiso.com/public/csv/{path}/{YYYYMM01}{path}_csv.zip
"""

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg2
import psycopg2.extras
import requests

logger = logging.getLogger(__name__)

NYISO_BASE = "http://mis.nyiso.com/public/csv"
BATCH_SIZE = 5_000


# ── URL builders ──────────────────────────────────────────────────────────────

def _url_load(month_start: datetime) -> str:
    d = month_start.strftime("%Y%m%d")
    return f"{NYISO_BASE}/pal/{d}pal_csv.zip"


def _url_fuel(month_start: datetime) -> str:
    d = month_start.strftime("%Y%m%d")
    return f"{NYISO_BASE}/rtfuelmix/{d}rtfuelmix_csv.zip"


def _url_lmp_da(month_start: datetime) -> str:
    d = month_start.strftime("%Y%m%d")
    return f"{NYISO_BASE}/damlbmp/{d}damlbmp_zone_csv.zip"


def _url_lmp_rt(month_start: datetime) -> str:
    d = month_start.strftime("%Y%m%d")
    return f"{NYISO_BASE}/rtlbmp/{d}rtlbmp_zone_csv.zip"


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _fetch_zip(url: str, retries: int = 3) -> list[pd.DataFrame]:
    """Download a NYISO ZIP and return a list of DataFrames (one per CSV inside)."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=45)
            if resp.status_code == 404:
                logger.warning(f"  404 — no data at {url}")
                return []
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                return [pd.read_csv(z.open(f)) for f in z.namelist() if f.endswith(".csv")]
        except requests.exceptions.RequestException as exc:
            logger.warning(f"  Attempt {attempt + 1} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    logger.error(f"  All retries failed for {url}")
    return []


def _month_starts(n_months_back: int = 1) -> list[datetime]:
    """Return first-of-month datetimes for the last n_months_back+1 months."""
    now = datetime.now(tz=timezone.utc)
    starts = []
    for i in range(n_months_back + 1):
        month = now.month - i
        year = now.year
        while month <= 0:
            month += 12
            year -= 1
        starts.append(datetime(year, month, 1))
    return starts


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _clean_load(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    ts_col = next((c for c in df.columns if "time stamp" in c.lower() or "timestamp" in c.lower()), None)
    if ts_col is None:
        logger.error(f"  No timestamp column in load data. Columns: {list(df.columns)}")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df[ts_col]).dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="shift_forward"
    )
    df = df.dropna(subset=["timestamp"])

    # "Name" column holds zone names (N.Y.C., LONGIL, etc.); "Time Zone" holds EDT/EST.
    # Must match "Name" exactly before "zone" in c.lower() would match "Time Zone".
    zone_col  = next((c for c in df.columns if c.strip().lower() == "name"), None)
    if zone_col is None:
        zone_col = next((c for c in df.columns if "name" in c.lower() or "zone" in c.lower()), None)
    ptid_col  = next((c for c in df.columns if "ptid" in c.lower()), None)
    load_col  = next((c for c in df.columns if c.strip().lower() == "load" or ("load" in c.lower() and "mw" in c.lower())), None)

    out = pd.DataFrame({"timestamp": df["timestamp"]})
    out["zone"]       = df[zone_col].astype(str).str.strip() if zone_col else None
    out["ptid"]       = df[ptid_col].astype("Int64")          if ptid_col else None
    out["load_mw"]    = pd.to_numeric(df[load_col], errors="coerce") if load_col else None
    out["is_outlier"] = False

    return out.dropna(subset=["zone", "load_mw"])


def _clean_fuel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    ts_col   = next((c for c in df.columns if "time stamp" in c.lower() or "timestamp" in c.lower()), None)
    fuel_col = next((c for c in df.columns if "fuel" in c.lower()), None)
    gen_col  = next((c for c in df.columns if "gen" in c.lower() and "mw" in c.lower()), None)

    if not all([ts_col, fuel_col, gen_col]):
        logger.error(f"  Missing columns in fuel_mix. Have: {list(df.columns)}")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df[ts_col]).dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="shift_forward"
    )
    df = df.dropna(subset=["timestamp"])

    out = pd.DataFrame({
        "timestamp":  df["timestamp"],
        "fuel_type":  df[fuel_col].astype(str).str.strip(),
        "gen_mw":     pd.to_numeric(df[gen_col], errors="coerce"),
        "is_outlier": False,
    })
    return out.dropna(subset=["fuel_type", "gen_mw"])


def _clean_lmp(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    ts_col    = next((c for c in df.columns if "time stamp" in c.lower() or "timestamp" in c.lower()), None)
    zone_col  = next((c for c in df.columns if "name" in c.lower() or "zone" in c.lower()), None)
    ptid_col  = next((c for c in df.columns if "ptid" in c.lower()), None)
    lmp_col   = next((c for c in df.columns if "lbmp" in c.strip().lower()), None)
    loss_col  = next((c for c in df.columns if "loss" in c.lower()), None)
    cong_col  = next((c for c in df.columns if "congestion" in c.lower()), None)

    if not ts_col:
        logger.error(f"  No timestamp column in LMP data. Have: {list(df.columns)}")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df[ts_col]).dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="shift_forward"
    )
    df = df.dropna(subset=["timestamp"])

    # Resample to hourly (NYISO DA is already hourly; RT may need floor)
    df["timestamp"] = df["timestamp"].dt.floor("h")

    out = pd.DataFrame({"timestamp": df["timestamp"]})
    out["zone"]           = df[zone_col].astype(str).str.strip() if zone_col else None
    out["ptid"]           = df[ptid_col].astype("Int64")          if ptid_col else None
    out["lmp_total"]      = pd.to_numeric(df[lmp_col],  errors="coerce") if lmp_col  else None
    out["lmp_losses"]     = pd.to_numeric(df[loss_col], errors="coerce") if loss_col else None
    out["lmp_congestion"] = pd.to_numeric(df[cong_col], errors="coerce") if cong_col else None
    out["is_outlier"]     = False

    return out.dropna(subset=["zone", "lmp_total"])


# ── Postgres bulk insert ──────────────────────────────────────────────────────

def _bulk_insert(conn, table: str, df: pd.DataFrame, conflict_cols: list[str]) -> int:
    if df.empty:
        return 0
    cols = list(df.columns)
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s "
        f"ON CONFLICT ({', '.join(conflict_cols)}) DO NOTHING"
    )
    import numpy as np
    def _to_native(v):
        if v is None or (isinstance(v, float) and v != v):
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    rows = [tuple(_to_native(v) for v in r) for r in df.itertuples(index=False, name=None)]
    inserted = 0
    with conn.cursor() as cur:
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start: start + BATCH_SIZE]
            psycopg2.extras.execute_values(cur, sql, batch, page_size=BATCH_SIZE)
            inserted += cur.rowcount
    conn.commit()
    return inserted


# ── Public ingest functions ───────────────────────────────────────────────────

def ingest_load(conn, n_months_back: int = 1) -> dict:
    """Fetch load_actual from NYISO and insert new rows. Returns counts."""
    counts = {}
    for month_start in _month_starts(n_months_back):
        url = _url_load(month_start)
        logger.info(f"  Fetching load_actual: {url}")
        frames = _fetch_zip(url)
        df = _clean_load(frames)
        if df.empty:
            continue
        keep = ["timestamp", "zone", "ptid", "load_mw", "is_outlier"]
        df = df[[c for c in keep if c in df.columns]]
        n = _bulk_insert(conn, "load_actual", df, ["timestamp", "zone"])
        counts[month_start.strftime("%Y-%m")] = n
        logger.info(f"    load_actual {month_start:%Y-%m}: {n} new rows")
    return counts


def ingest_fuel(conn, n_months_back: int = 1) -> dict:
    """Fetch fuel_mix from NYISO and insert new rows. Returns counts."""
    counts = {}
    for month_start in _month_starts(n_months_back):
        url = _url_fuel(month_start)
        logger.info(f"  Fetching fuel_mix: {url}")
        frames = _fetch_zip(url)
        df = _clean_fuel(frames)
        if df.empty:
            continue
        keep = ["timestamp", "fuel_type", "gen_mw", "is_outlier"]
        df = df[[c for c in keep if c in df.columns]]
        n = _bulk_insert(conn, "fuel_mix", df, ["timestamp", "fuel_type"])
        counts[month_start.strftime("%Y-%m")] = n
        logger.info(f"    fuel_mix {month_start:%Y-%m}: {n} new rows")
    return counts


def ingest_lmp_dayahead(conn, n_months_back: int = 1) -> dict:
    """Fetch DA LMP from NYISO and insert new rows. Returns counts."""
    counts = {}
    for month_start in _month_starts(n_months_back):
        url = _url_lmp_da(month_start)
        logger.info(f"  Fetching lmp_dayahead: {url}")
        frames = _fetch_zip(url)
        df = _clean_lmp(frames)
        if df.empty:
            continue
        keep = ["timestamp", "zone", "ptid", "lmp_total", "lmp_losses", "lmp_congestion", "is_outlier"]
        df = df[[c for c in keep if c in df.columns]]
        n = _bulk_insert(conn, "lmp_dayahead", df, ["timestamp", "zone"])
        counts[month_start.strftime("%Y-%m")] = n
        logger.info(f"    lmp_dayahead {month_start:%Y-%m}: {n} new rows")
    return counts


def ingest_lmp_realtime(conn, n_months_back: int = 1) -> dict:
    """Fetch RT LMP from NYISO and insert new rows. Returns counts."""
    counts = {}
    for month_start in _month_starts(n_months_back):
        url = _url_lmp_rt(month_start)
        logger.info(f"  Fetching lmp_realtime: {url}")
        frames = _fetch_zip(url)
        df = _clean_lmp(frames)
        if df.empty:
            continue
        keep = ["timestamp", "zone", "ptid", "lmp_total", "lmp_losses", "lmp_congestion", "is_outlier"]
        df = df[[c for c in keep if c in df.columns]]
        n = _bulk_insert(conn, "lmp_realtime", df, ["timestamp", "zone"])
        counts[month_start.strftime("%Y-%m")] = n
        logger.info(f"    lmp_realtime {month_start:%Y-%m}: {n} new rows")
    return counts
