"""
backfill_2026.py — One-time backfill of NYISO data from Jan 1 – May 17, 2026.

Reuses the fetch/clean/insert logic from lambda/ingest.py.
Idempotent: ON CONFLICT DO NOTHING, safe to re-run.

Usage:
    conda activate nyiso
    python scripts/backfill_2026.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import psycopg2
from dotenv import dotenv_values

# ── Path setup ────────────────────────────────────────────────────────────────
# Add lambda/ to sys.path so we can import ingest directly
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "lambda"))

from ingest import (
    _fetch_zip,
    _clean_load,
    _clean_fuel,
    _clean_lmp,
    _bulk_insert,
    _url_load,
    _url_fuel,
    _url_lmp_da,
    _url_lmp_rt,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
START = datetime(2026, 1, 1)
END   = datetime(2026, 5, 17)

DATASETS = [
    ("load_actual",  _url_load,   _clean_load, ["timestamp", "zone"],
     ["timestamp", "zone", "ptid", "load_mw", "is_outlier"]),
    ("fuel_mix",     _url_fuel,   _clean_fuel, ["timestamp", "fuel_type"],
     ["timestamp", "fuel_type", "gen_mw", "is_outlier"]),
    ("lmp_dayahead", _url_lmp_da, _clean_lmp,  ["timestamp", "zone"],
     ["timestamp", "zone", "ptid", "lmp_total", "lmp_losses", "lmp_congestion", "is_outlier"]),
    ("lmp_realtime", _url_lmp_rt, _clean_lmp,  ["timestamp", "zone"],
     ["timestamp", "zone", "ptid", "lmp_total", "lmp_losses", "lmp_congestion", "is_outlier"]),
]


def month_starts(start: datetime, end: datetime) -> list[datetime]:
    """Return first-of-month datetimes from start month through end month."""
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append(current)
        month = current.month + 1 if current.month < 12 else 1
        year  = current.year if current.month < 12 else current.year + 1
        current = current.replace(year=year, month=month)
    return months


def main() -> None:
    cfg = dotenv_values(REPO_ROOT / ".env")
    db_url = cfg.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not found in .env")
        sys.exit(1)

    conn = psycopg2.connect(db_url)
    logger.info(f"Connected to Postgres. Backfilling {START:%Y-%m-%d} → {END:%Y-%m-%d}")

    months = month_starts(START, END)
    logger.info(f"Months to fetch: {[m.strftime('%Y-%m') for m in months]}")

    totals: dict[str, int] = {name: 0 for name, *_ in DATASETS}

    for month in months:
        logger.info(f"\n{'='*60}")
        logger.info(f"Month: {month:%Y-%m}")
        logger.info(f"{'='*60}")

        for table, url_fn, clean_fn, conflict_cols, keep_cols in DATASETS:
            url = url_fn(month)
            logger.info(f"  [{table}] Fetching {url}")
            frames = _fetch_zip(url)
            df = clean_fn(frames)

            if df.empty:
                logger.warning(f"  [{table}] No data returned — skipping.")
                continue

            df = df[[c for c in keep_cols if c in df.columns]]
            n = _bulk_insert(conn, table, df, conflict_cols)
            totals[table] += n
            logger.info(f"  [{table}] {month:%Y-%m}: {n:,} new rows inserted")

    conn.close()

    logger.info(f"\n{'='*60}")
    logger.info("Backfill complete. Totals inserted:")
    for table, count in totals.items():
        logger.info(f"  {table:<16} {count:>10,} rows")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
