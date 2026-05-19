"""
handler.py — AWS Lambda entry points for NYISO Live Dashboard.

Three handler functions, each mapped to a separate Lambda + EventBridge rule:

  ingest_load_fuel(event, context)
      EventBridge cron: rate(5 minutes)
      Fetches load_actual + fuel_mix from NYISO → Postgres.

  ingest_lmp(event, context)
      EventBridge cron: rate(1 hour)
      Fetches lmp_dayahead + lmp_realtime → Postgres.
      Also runs XGBoost inference to write load + LMP forecasts.

  run_bess_dispatch(event, context)
      EventBridge cron: cron(0 13 * * ? *)  — 1pm UTC = ~9am ET (after DA clears)
      Reads today's DA LMP from Postgres, runs LP, writes bess_dispatch.

Deploy:
  All three can share one Lambda function with handler routing via event['task'],
  or be deployed as separate Lambdas pointing to different handlers.

Environment variables required:
  DATABASE_URL   — Postgres connection string (from AWS Secrets Manager)
  MODEL_DIR      — (optional) path to model artifacts; defaults to ./models/
  MODEL_VERSION  — (optional) git SHA or tag of this deployment
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

# XGBoost 1.x imports scipy.sparse unconditionally at the top of xgboost/core.py.
# scipy is not in the Lambda layer (stripped to stay within the 250 MB limit).
# We install a minimal stub so xgboost's import succeeds; the stub causes all
# scipy.sparse.issparse() checks to return False, which is correct because our
# feature matrices are always dense numpy arrays / pandas DataFrames.
try:
    import scipy.sparse  # noqa — works fine if scipy is present
except ImportError:
    # scipy is stripped from the Lambda layer (saves ~100 MB).
    # XGBoost 1.x imports scipy.sparse unconditionally; sklearn's predict path
    # also touches scipy.special.  Provide proper ModuleType stubs so that
    # Python treats scipy as a package (SimpleNamespace fails the package check).
    from types import ModuleType
    _scipy = ModuleType("scipy")
    _scipy.__path__ = []          # mark as package so sub-imports succeed
    _scipy.__spec__ = None
    _scipy_sparse = ModuleType("scipy.sparse")
    _scipy_sparse.issparse = lambda x: False
    _scipy_sparse.spmatrix = type("spmatrix", (), {})
    _scipy_sparse.csr_matrix = type("csr_matrix", (), {})
    _scipy_sparse.csc_matrix = type("csc_matrix", (), {})
    _scipy_sparse.coo_matrix = type("coo_matrix", (), {})
    _scipy_special = ModuleType("scipy.special")
    # xgboost/sklearn.py imports softmax + expit at module level (used for
    # predict_proba in classifiers; not called during regression predict).
    def _softmax(x, axis=None):
        import numpy as _np
        x = _np.asarray(x, dtype=float)
        e = _np.exp(x - _np.max(x, axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)
    def _expit(x):
        import numpy as _np
        return 1.0 / (1.0 + _np.exp(-_np.asarray(x, dtype=float)))
    _scipy_special.softmax = _softmax
    _scipy_special.expit = _expit
    _scipy_linalg = ModuleType("scipy.linalg")
    _scipy_stats = ModuleType("scipy.stats")
    _scipy.sparse = _scipy_sparse
    _scipy.special = _scipy_special
    _scipy.linalg = _scipy_linalg
    _scipy.stats = _scipy_stats
    sys.modules.setdefault("scipy", _scipy)
    sys.modules.setdefault("scipy.sparse", _scipy_sparse)
    sys.modules.setdefault("scipy.special", _scipy_special)
    sys.modules.setdefault("scipy.linalg", _scipy_linalg)
    sys.modules.setdefault("scipy.stats", _scipy_stats)

import psycopg2
from dotenv import load_dotenv

# Allow local imports when running outside Lambda (e.g., scripts/export_models.py)
sys.path.insert(0, os.path.dirname(__file__))

# Heavy ML/ingest imports are intentionally NOT imported here.
# They are lazy-loaded inside each handler so that update_dns cold-starts
# don't pay the cost of importing xgboost, joblib, pulp, pandas, etc.
# update_dns only needs boto3 + requests, both of which it imports inline.

load_dotenv()

# Lambda's runtime pre-configures the root logger before our code runs.
# logging.basicConfig() is a no-op when handlers already exist, so the
# level stays at WARNING and every logger.info() is silently dropped.
# Directly setting the root logger level is the correct fix.
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

MODEL_VERSION = os.environ.get("MODEL_VERSION")

# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_conn() -> psycopg2.extensions.connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError("DATABASE_URL environment variable not set.")
    return psycopg2.connect(url)


def _ok(body: dict) -> dict:
    return {"statusCode": 200, "body": json.dumps(body)}


def _err(msg: str, exc: Exception = None) -> dict:
    logger.error(f"{msg}: {exc}" if exc else msg)
    return {"statusCode": 500, "body": json.dumps({"error": msg, "detail": str(exc) if exc else None})}


# ── Handler: load + fuel ingest (every 5 min) ─────────────────────────────────

def ingest_load_fuel(event: dict, context=None) -> dict:
    """
    Fetch load_actual and fuel_mix for the current and previous month.
    Idempotent — new rows are inserted, existing rows are skipped.
    """
    from ingest import ingest_load, ingest_fuel
    logger.info("=== ingest_load_fuel START ===")
    conn = None
    try:
        conn = _get_conn()
        load_counts = ingest_load(conn, n_months_back=1)
        fuel_counts = ingest_fuel(conn, n_months_back=1)
        result = {"load_actual": load_counts, "fuel_mix": fuel_counts}
        logger.info(f"=== ingest_load_fuel DONE: {result} ===")
        return _ok(result)
    except Exception as exc:
        return _err("ingest_load_fuel failed", exc)
    finally:
        if conn:
            conn.close()


# ── Handler: LMP ingest + forecasting (every hour) ───────────────────────────

def ingest_lmp(event: dict, context=None) -> dict:
    """
    1. Fetch lmp_dayahead + lmp_realtime for current/previous month.
    2. Run XGBoost load + LMP forecasts for the next 24 hours.
    3. Backfill forecast actuals for past hours.
    """
    import pandas as pd
    from ingest import ingest_lmp_dayahead, ingest_lmp_realtime
    from features import fetch_weather_forecast, build_load_features, build_lmp_features
    from inference import (
        predict_load, predict_lmp,
        write_load_forecast, write_lmp_forecast, backfill_forecast_actuals,
    )
    logger.info("=== ingest_lmp START ===")
    conn = None
    try:
        conn = _get_conn()

        # 1. Ingest raw prices
        da_counts = ingest_lmp_dayahead(conn, n_months_back=1)
        rt_counts = ingest_lmp_realtime(conn, n_months_back=1)

        # 2. Build forecast horizon: next 24 hours (hourly)
        now_et = pd.Timestamp.now(tz="US/Eastern").floor("h")
        target_ts = pd.date_range(start=now_et + pd.Timedelta(hours=1), periods=24, freq="h")
        start_str = target_ts.min().date().isoformat()
        end_str   = target_ts.max().date().isoformat()

        logger.info(f"  Fetching weather forecast {start_str} → {end_str}")
        weather = fetch_weather_forecast(start_str, end_str)

        # 3. Load forecasts (NYC + LONGIL) — wrapped so LMP still runs if load fails
        load_written = {}
        for zone in ["N.Y.C.", "LONGIL"]:
            try:
                feat_df = build_load_features(conn, target_ts, zone, weather)
                preds   = predict_load(zone, feat_df)
                n = write_load_forecast(conn, zone, target_ts, preds, MODEL_VERSION)
                load_written[zone] = n
            except Exception as load_exc:
                logger.warning(f"  load forecast {zone} failed (skipping): {load_exc}")
                load_written[zone] = 0

        # 4. LMP forecast (NYC) — always runs regardless of load forecast outcome
        lmp_feat_df = build_lmp_features(conn, target_ts, "N.Y.C.", weather)
        lmp_preds   = predict_lmp(lmp_feat_df)
        lmp_written = write_lmp_forecast(conn, "N.Y.C.", target_ts, lmp_preds, MODEL_VERSION)

        # 5. Backfill actuals for past forecasts
        backfill_forecast_actuals(conn)

        result = {
            "lmp_dayahead":  da_counts,
            "lmp_realtime":  rt_counts,
            "load_forecast": load_written,
            "lmp_forecast":  {"N.Y.C.": lmp_written},
        }
        logger.info(f"=== ingest_lmp DONE: {result} ===")
        return _ok(result)

    except Exception as exc:
        return _err("ingest_lmp failed", exc)
    finally:
        if conn:
            conn.close()


# ── Handler: BESS dispatch (daily at ~9am ET, after DA market clears) ─────────

def run_bess_dispatch(event: dict, context=None) -> dict:
    """
    Fetch today's DA LMP for N.Y.C., run LP optimizer, write hourly dispatch plan.
    """
    import pandas as pd
    from bess_dispatch import fetch_da_lmp_today, optimize_dispatch, write_dispatch
    logger.info("=== run_bess_dispatch START ===")
    conn = None
    try:
        conn = _get_conn()
        zone = "N.Y.C."

        lmp_today = fetch_da_lmp_today(conn, zone)
        if lmp_today.empty:
            return _err(f"No DA LMP available for {zone} today — dispatch skipped")

        dispatch_df = optimize_dispatch(lmp_today)

        # Delivery day: today (DA prices apply to today's delivery)
        today_et = pd.Timestamp.now(tz="US/Eastern").date()
        midnight  = pd.Timestamp(today_et).tz_localize("US/Eastern")

        n = write_dispatch(conn, dispatch_df, today_et, zone, midnight)

        total_rev = dispatch_df["revenue_usd"].sum()
        result = {
            "run_date":    today_et.isoformat(),
            "zone":        zone,
            "rows_written": n,
            "total_revenue_usd": round(total_rev, 2),
        }
        logger.info(f"=== run_bess_dispatch DONE: {result} ===")
        return _ok(result)

    except Exception as exc:
        return _err("run_bess_dispatch failed", exc)
    finally:
        if conn:
            conn.close()


# ── Handler: daily BESS revenue summary (10am ET, after DA clears) ───────────

def compute_revenue_summary(event: dict, context=None) -> dict:
    """
    For each day not yet in bess_revenue_summary (or within the last 7 days to
    pick up retroactive DA LMP corrections), compute:

      perfect_foresight_revenue: LP on actual DA LMP prices (upper bound)
      naive_revenue:             Fixed schedule (charge 0-5am, discharge 2-6pm)
      lp_revenue:                LP on XGBoost forecast prices, dispatch evaluated
                                 at actual DA prices (NULL when no forecast data)

    Self-creates the table if missing. Idempotent — safe to run multiple times.
    """
    import pandas as pd
    from datetime import date, timedelta
    from bess_dispatch import optimize_dispatch, naive_daily_revenue

    logger.info("=== compute_revenue_summary START ===")
    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()

        # Self-create table (harmless if already exists)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bess_revenue_summary (
                date                      DATE PRIMARY KEY,
                perfect_foresight_revenue DOUBLE PRECISION NOT NULL,
                naive_revenue             DOUBLE PRECISION NOT NULL,
                lp_revenue                DOUBLE PRECISION,
                created_at                TIMESTAMPTZ DEFAULT NOW(),
                updated_at                TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.commit()

        # Only (re)compute dates not yet stored, plus last 7 days for corrections
        cur.execute("SELECT COALESCE(MAX(date), '1970-01-01'::date) FROM bess_revenue_summary")
        last_stored = cur.fetchone()[0]
        one_year_ago = date.today() - timedelta(days=365)
        compute_from = max(one_year_ago, last_stored - timedelta(days=7))
        logger.info(f"  Computing from {compute_from} (last stored: {last_stored})")

        # Actual DA LMP for all target days
        df_da = pd.read_sql("""
            SELECT (timestamp AT TIME ZONE 'US/Eastern')::date AS day, lmp_total
            FROM lmp_dayahead
            WHERE zone = 'N.Y.C.'
              AND (timestamp AT TIME ZONE 'US/Eastern')::date >= %(from_d)s
              AND timestamp < NOW()
            ORDER BY day, timestamp
        """, conn, params={"from_d": compute_from})

        # Forecast data — only rows where actuals have been backfilled
        df_fcst = pd.read_sql("""
            SELECT (timestamp AT TIME ZONE 'US/Eastern')::date AS day,
                   EXTRACT(HOUR FROM timestamp AT TIME ZONE 'US/Eastern')::int AS hour,
                   lmp_forecast, lmp_actual
            FROM lmp_forecast
            WHERE zone = 'N.Y.C.'
              AND (timestamp AT TIME ZONE 'US/Eastern')::date >= %(from_d)s
              AND lmp_actual IS NOT NULL
              AND timestamp < NOW()
            ORDER BY day, hour
        """, conn, params={"from_d": compute_from})

        fcst_by_day = {d: g for d, g in df_fcst.groupby("day")} if not df_fcst.empty else {}

        rows_written = 0
        for day, grp in df_da.groupby("day"):
            prices = grp["lmp_total"].tolist()
            if len(prices) < 20:
                continue

            # Perfect foresight: LP on actual DA prices
            pf_res = optimize_dispatch(pd.Series(prices, dtype=float))
            pf_rev = round(float(pf_res["revenue_usd"].sum()), 2)

            # Naive: fixed schedule
            naive_rev = round(naive_daily_revenue(pd.Series(prices, dtype=float)), 2)

            # Our optimizer: LP on forecast, evaluated at actual DA prices
            lp_rev = None
            if day in fcst_by_day:
                fg = fcst_by_day[day].sort_values("hour")
                if len(fg) >= 20:
                    lp_res = optimize_dispatch(pd.Series(fg["lmp_forecast"].tolist(), dtype=float))
                    actual_px = fg["lmp_actual"].tolist()
                    day_lp = 0.0
                    for _, row in lp_res.iterrows():
                        h = int(row["hour"])
                        if h < len(actual_px) and actual_px[h] is not None:
                            p = float(actual_px[h])
                            day_lp += float(row["discharge_mw"]) * p - float(row["charge_mw"]) * p
                    lp_rev = round(day_lp, 2)

            cur.execute("""
                INSERT INTO bess_revenue_summary
                    (date, perfect_foresight_revenue, naive_revenue, lp_revenue, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (date) DO UPDATE SET
                    perfect_foresight_revenue = EXCLUDED.perfect_foresight_revenue,
                    naive_revenue             = EXCLUDED.naive_revenue,
                    lp_revenue                = EXCLUDED.lp_revenue,
                    updated_at                = NOW()
            """, (day, pf_rev, naive_rev, lp_rev))
            rows_written += 1

        conn.commit()
        logger.info(f"=== compute_revenue_summary DONE: {rows_written} rows written ===")
        return _ok({"days_written": rows_written, "computed_from": str(compute_from)})

    except Exception as exc:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return _err("compute_revenue_summary failed", exc)
    finally:
        if conn:
            conn.close()


# ── Handler: historical forecast backfill ─────────────────────────────────────

def backfill_forecasts(event: dict, context=None) -> dict:
    """
    Backfill lmp_forecast and load_forecast for historical dates using ERA5 archive weather.

    Call with: {"task": "backfill_forecasts", "start_date": "2026-01-01", "end_date": "2026-01-31"}
    Defaults to 2026-01-01 through yesterday.

    Uses Open-Meteo archive API for older dates and the forecast API for the
    most recent 7 days (where ERA5 reanalysis may not yet be available).
    """
    import pandas as pd
    from datetime import date, timedelta
    from features import (
        fetch_weather_archive,
        fetch_weather_forecast,
        build_load_features,
        build_lmp_features,
    )
    from inference import (
        predict_load, predict_lmp,
        write_load_forecast, write_lmp_forecast, backfill_forecast_actuals,
    )

    logger.info("=== backfill_forecasts START ===")
    conn = None
    try:
        conn = _get_conn()

        today = date.today()
        start_date = date.fromisoformat(event.get("start_date", "2026-01-01"))
        end_date   = date.fromisoformat(event.get("end_date", (today - timedelta(days=1)).isoformat()))
        logger.info(f"  Range: {start_date} → {end_date}")

        # Backfill missing load_actual data (needed for lag features).
        # Uses n_months_back=6 to cover Jan–Jun 2026 from a single Lambda invocation.
        from ingest import ingest_load
        logger.info("  Backfilling load_actual (n_months_back=6)...")
        load_counts = ingest_load(conn, n_months_back=6)
        logger.info(f"  load_actual backfill: {load_counts}")

        # ERA5 archive has a ~5-7 day lag; use forecast API for recent days
        archive_cutoff = today - timedelta(days=7)
        archive_end  = min(end_date, archive_cutoff)
        recent_start = archive_cutoff + timedelta(days=1)

        weather_parts = []
        if archive_end >= start_date:
            logger.info(f"  Fetching archive weather {start_date} → {archive_end}")
            weather_parts.append(fetch_weather_archive(start_date.isoformat(), archive_end.isoformat()))
        if recent_start <= end_date:
            logger.info(f"  Fetching forecast weather {recent_start} → {end_date}")
            weather_parts.append(fetch_weather_forecast(recent_start.isoformat(), end_date.isoformat()))

        weather = pd.concat(weather_parts, ignore_index=True) if weather_parts else pd.DataFrame()
        logger.info(f"  Weather rows fetched: {len(weather)}")

        lmp_total  = 0
        load_total = {"N.Y.C.": 0, "LONGIL": 0}

        current = start_date
        while current <= end_date:
            midnight_et = pd.Timestamp(current).tz_localize("US/Eastern")
            target_ts   = pd.date_range(start=midnight_et, periods=24, freq="h")

            for zone in ["N.Y.C.", "LONGIL"]:
                try:
                    feat  = build_load_features(conn, target_ts, zone, weather)
                    preds = predict_load(zone, feat)
                    n = write_load_forecast(conn, zone, target_ts, preds, MODEL_VERSION)
                    load_total[zone] += n
                except Exception as exc:
                    logger.warning(f"  load {zone} {current}: {exc}")

            try:
                lmp_feat  = build_lmp_features(conn, target_ts, "N.Y.C.", weather)
                lmp_preds = predict_lmp(lmp_feat)
                n = write_lmp_forecast(conn, "N.Y.C.", target_ts, lmp_preds, MODEL_VERSION)
                lmp_total += n
            except Exception as exc:
                logger.warning(f"  lmp N.Y.C. {current}: {exc}")

            if current.day == 1:
                logger.info(f"  Progress: {current}")
            current += timedelta(days=1)

        backfill_forecast_actuals(conn)

        result = {
            "start_date": start_date.isoformat(),
            "end_date":   end_date.isoformat(),
            "lmp_forecast_rows":  lmp_total,
            "load_forecast_rows": load_total,
        }
        logger.info(f"=== backfill_forecasts DONE: {result} ===")
        return _ok(result)

    except Exception as exc:
        return _err("backfill_forecasts failed", exc)
    finally:
        if conn:
            conn.close()


# ── Handler: Cloudflare DNS update (ECS Task State Change) ───────────────────

def update_dns(event: dict, context=None) -> dict:
    """
    Called by EventBridge on ECS Task State Change (lastStatus=RUNNING).
    Extracts the new Fargate task's public IP from the event and updates
    the Cloudflare A record so the domain always points at the live task.

    Env vars required: CF_API_TOKEN, CF_ZONE_ID, CF_RECORD_NAME
    """
    import boto3
    import requests as _requests

    logger.info("=== update_dns START ===")
    detail = event.get("detail", {})

    # Extract ENI ID from ECS task event attachments
    eni_id = None
    for att in detail.get("attachments", []):
        for d in att.get("details", []):
            if d.get("name") == "networkInterfaceId":
                eni_id = d["value"]
                break
        if eni_id:
            break

    if not eni_id:
        return _err("No networkInterfaceId in ECS task event — cannot update DNS")

    # Resolve ENI → public IP
    ec2 = boto3.client("ec2", region_name="us-east-1")
    eni = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])["NetworkInterfaces"][0]
    public_ip = eni.get("Association", {}).get("PublicIp")

    if not public_ip:
        return _err(f"ENI {eni_id} has no public IP yet — task may still be provisioning")

    # Cloudflare DNS update
    token   = os.environ["CF_API_TOKEN"]
    zone_id = os.environ["CF_ZONE_ID"]
    name    = os.environ.get("CF_RECORD_NAME", "nyiso.rahishah.com")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Look up record ID dynamically so no hardcoded IDs in code
    r = _requests.get(
        f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records",
        headers=headers,
        params={"name": name, "type": "A"},
        timeout=10,
    )
    records = r.json().get("result", [])
    if not records:
        return _err(f"No A record found for {name} in Cloudflare zone {zone_id}")

    record_id = records[0]["id"]
    old_ip    = records[0]["content"]

    r = _requests.patch(
        f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records/{record_id}",
        headers=headers,
        json={"content": public_ip},
        timeout=10,
    )
    resp = r.json()
    if not resp.get("success"):
        return _err(f"Cloudflare update failed: {resp.get('errors')}")

    logger.info(f"DNS updated: {name}  {old_ip} → {public_ip}")
    return _ok({"record": name, "old_ip": old_ip, "new_ip": public_ip})


# ── Unified dispatcher (single Lambda with task routing) ──────────────────────

def lambda_handler(event: dict, context=None) -> dict:
    """
    Single entry point for all three tasks — routes on event['task'].

    EventBridge input transformer for each rule should inject:
      {"task": "ingest_load_fuel"}
      {"task": "ingest_lmp"}
      {"task": "bess_dispatch"}
    """
    # ECS Task State Change events come directly from EventBridge (no 'task' key)
    if event.get("detail-type") == "ECS Task State Change":
        return update_dns(event, context)

    task = event.get("task", "ingest_lmp")
    logger.info(f"lambda_handler dispatching task='{task}'")

    dispatch = {
        "ingest_load_fuel":        ingest_load_fuel,
        "ingest_lmp":              ingest_lmp,
        "bess_dispatch":           run_bess_dispatch,
        "compute_revenue_summary": compute_revenue_summary,
        "backfill_forecasts":      backfill_forecasts,
        "update_dns":              update_dns,
    }

    if task not in dispatch:
        return _err(f"Unknown task '{task}'. Valid: {list(dispatch)}")

    return dispatch[task](event, context)
