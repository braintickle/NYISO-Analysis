"""
migrate_to_postgres.py
----------------------
Idempotent migration script: loads all processed parquets into Supabase Postgres.

Usage:
    python src/migrate_to_postgres.py                  # migrate all tables
    python src/migrate_to_postgres.py --table fuel_mix  # single table
    python src/migrate_to_postgres.py --dry-run         # print counts only

Connection: reads DATABASE_URL from environment (set in .env or shell).
Idempotency: INSERT ... ON CONFLICT DO NOTHING — safe to re-run at any time.
"""

import os
import sys
import argparse
import logging
import pathlib
from typing import Optional

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SCHEMA_FILE = REPO_ROOT / "sql" / "schema.sql"

BATCH_SIZE = 10_000


# ── Connection ─────────────────────────────────────────────────────────────────

def get_connection() -> psycopg2.extensions.connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL environment variable not set.")
        sys.exit(1)
    return psycopg2.connect(url)


# ── Schema ─────────────────────────────────────────────────────────────────────

def apply_schema(conn: psycopg2.extensions.connection) -> None:
    logger.info("Applying schema from sql/schema.sql ...")
    ddl = SCHEMA_FILE.read_text()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    logger.info("Schema applied.")


# ── Bulk insert helper ─────────────────────────────────────────────────────────

def bulk_insert(
    conn: psycopg2.extensions.connection,
    table: str,
    df: pd.DataFrame,
    conflict_cols: list[str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Insert DataFrame rows into table using execute_values in batches.
    Rows that violate the unique constraint are silently skipped.

    Returns (rows_attempted, rows_inserted).
    """
    if df.empty:
        logger.warning(f"  [{table}] DataFrame is empty — nothing to insert.")
        return 0, 0

    cols = list(df.columns)
    col_list = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))
    conflict_target = ", ".join(conflict_cols)

    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES %s "
        f"ON CONFLICT ({conflict_target}) DO NOTHING"
    )

    rows = [tuple(row) for row in df.itertuples(index=False, name=None)]
    attempted = len(rows)

    if dry_run:
        logger.info(f"  [DRY RUN] {table}: {attempted:,} rows would be inserted")
        return attempted, 0

    inserted = 0
    with conn.cursor() as cur:
        for start in range(0, attempted, BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            psycopg2.extras.execute_values(cur, sql, batch, page_size=BATCH_SIZE)
            inserted += cur.rowcount
            logger.info(
                f"  [{table}] Batch {start // BATCH_SIZE + 1}: "
                f"{cur.rowcount:,} inserted / {len(batch):,} attempted"
            )
    conn.commit()

    skipped = attempted - inserted
    logger.info(
        f"  [{table}] Done — {inserted:,} inserted, {skipped:,} skipped (duplicates)"
    )
    return attempted, inserted


# ── Per-table migration functions ──────────────────────────────────────────────

def migrate_load_actual(conn, dry_run: bool = False) -> None:
    logger.info("Migrating load_actual ...")
    df = pd.read_parquet(PROCESSED_DIR / "load_actual.parquet")

    # Drop NaN-only column present in raw data
    if "Time Zone" in df.columns:
        df = df.drop(columns=["Time Zone"])

    # Normalize column names
    df = df.rename(columns={
        "PTID": "ptid",
        "load_mw_outlier": "is_outlier",
    })

    # Keep only schema columns (drop any extras)
    keep = ["timestamp", "zone", "ptid", "load_mw", "is_outlier"]
    df = df[[c for c in keep if c in df.columns]]

    logger.info(f"  Rows: {len(df):,} | Columns: {list(df.columns)}")
    bulk_insert(conn, "load_actual", df, ["timestamp", "zone"], dry_run)


def migrate_lmp_dayahead(conn, dry_run: bool = False) -> None:
    logger.info("Migrating lmp_dayahead ...")
    df = pd.read_parquet(PROCESSED_DIR / "lmp_dayahead.parquet")

    df = df.rename(columns={
        "PTID": "ptid",
        "lmp_total_outlier": "is_outlier",
    })

    keep = ["timestamp", "zone", "ptid", "lmp_total", "lmp_losses", "lmp_congestion", "is_outlier"]
    df = df[[c for c in keep if c in df.columns]]

    logger.info(f"  Rows: {len(df):,} | Columns: {list(df.columns)}")
    bulk_insert(conn, "lmp_dayahead", df, ["timestamp", "zone"], dry_run)


def migrate_lmp_realtime(conn, dry_run: bool = False) -> None:
    logger.info("Migrating lmp_realtime ...")
    df = pd.read_parquet(PROCESSED_DIR / "lmp_realtime.parquet")

    # ptid already lowercase in this parquet
    df = df.rename(columns={
        "lmp_total_outlier": "is_outlier",
        "PTID": "ptid",  # normalize in case it varies
    })

    keep = ["timestamp", "zone", "ptid", "lmp_total", "lmp_losses", "lmp_congestion", "is_outlier"]
    df = df[[c for c in keep if c in df.columns]]

    logger.info(f"  Rows: {len(df):,} | Columns: {list(df.columns)}")
    bulk_insert(conn, "lmp_realtime", df, ["timestamp", "zone"], dry_run)


def migrate_fuel_mix(conn, dry_run: bool = False) -> None:
    logger.info("Migrating fuel_mix ...")
    df = pd.read_parquet(PROCESSED_DIR / "fuel_mix.parquet")

    if "Time Zone" in df.columns:
        df = df.drop(columns=["Time Zone"])

    df = df.rename(columns={"gen_mw_outlier": "is_outlier"})

    keep = ["timestamp", "fuel_type", "gen_mw", "is_outlier"]
    df = df[[c for c in keep if c in df.columns]]

    logger.info(f"  Rows: {len(df):,} | Columns: {list(df.columns)}")
    bulk_insert(conn, "fuel_mix", df, ["timestamp", "fuel_type"], dry_run)


def migrate_system_load(conn, dry_run: bool = False) -> None:
    logger.info("Migrating system_load ...")
    df = pd.read_parquet(PROCESSED_DIR / "system_load.parquet")

    keep = ["timestamp", "total_load_mw"]
    df = df[[c for c in keep if c in df.columns]]

    logger.info(f"  Rows: {len(df):,} | Columns: {list(df.columns)}")
    bulk_insert(conn, "system_load", df, ["timestamp"], dry_run)


def migrate_lmp_forecast(conn, dry_run: bool = False) -> None:
    logger.info("Migrating lmp_forecast (from lmp_forecast_2025.parquet) ...")
    df = pd.read_parquet(PROCESSED_DIR / "lmp_forecast_2025.parquet")

    # Historical forecasts are NYC-only; add zone and model_version columns
    if "zone" not in df.columns:
        df["zone"] = "N.Y.C."
    if "model_version" not in df.columns:
        df["model_version"] = None

    keep = ["timestamp", "zone", "lmp_forecast", "lmp_actual", "forecast_error", "model_version"]
    df = df[[c for c in keep if c in df.columns]]

    logger.info(f"  Rows: {len(df):,} | Columns: {list(df.columns)}")
    bulk_insert(conn, "lmp_forecast", df, ["timestamp", "zone"], dry_run)


# ── Table registry ─────────────────────────────────────────────────────────────

TABLES = {
    "load_actual":   migrate_load_actual,
    "lmp_dayahead":  migrate_lmp_dayahead,
    "lmp_realtime":  migrate_lmp_realtime,
    "fuel_mix":      migrate_fuel_mix,
    "system_load":   migrate_system_load,
    "lmp_forecast":  migrate_lmp_forecast,
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate processed parquets to Supabase Postgres")
    parser.add_argument(
        "--table",
        choices=list(TABLES.keys()),
        help="Migrate only this table (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print row counts and column names without writing to Postgres",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== DRY RUN — no data will be written ===")

    conn = get_connection() if not args.dry_run else None

    # For dry-run we still need a connection to apply schema (idempotent DDL).
    # If DATABASE_URL is not set at all in dry-run mode, skip schema and proceed.
    if args.dry_run and os.environ.get("DATABASE_URL"):
        conn = get_connection()
    elif args.dry_run:
        conn = None

    try:
        if conn and not args.dry_run:
            apply_schema(conn)
        elif conn and args.dry_run:
            # Apply schema in dry-run too so we catch DDL errors early
            apply_schema(conn)

        tables_to_run = [args.table] if args.table else list(TABLES.keys())

        for table_name in tables_to_run:
            migrate_fn = TABLES[table_name]
            try:
                migrate_fn(conn, dry_run=args.dry_run)
            except FileNotFoundError as e:
                logger.warning(f"  Skipping {table_name}: parquet not found — {e}")
            except Exception as e:
                logger.error(f"  Failed to migrate {table_name}: {e}")
                if conn:
                    conn.rollback()
                raise

    finally:
        if conn:
            conn.close()

    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
