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

import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Allow local imports when running outside Lambda (e.g., scripts/export_models.py)
sys.path.insert(0, os.path.dirname(__file__))

from ingest import ingest_load, ingest_fuel, ingest_lmp_dayahead, ingest_lmp_realtime
from features import (
    fetch_weather_forecast,
    build_load_features,
    build_lmp_features,
)
from inference import (
    predict_load,
    predict_lmp,
    write_load_forecast,
    write_lmp_forecast,
    backfill_forecast_actuals,
)
from bess_dispatch import fetch_da_lmp_today, optimize_dispatch, write_dispatch

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s — %(message)s",
    datefmt="%H:%M:%S",
)
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

        # 3. Load forecasts (NYC + LONGIL)
        load_written = {}
        for zone in ["N.Y.C.", "LONGIL"]:
            feat_df = build_load_features(conn, target_ts, zone, weather)
            preds   = predict_load(zone, feat_df)
            n = write_load_forecast(conn, zone, target_ts, preds, MODEL_VERSION)
            load_written[zone] = n

        # 4. LMP forecast (NYC)
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
        "ingest_load_fuel": ingest_load_fuel,
        "ingest_lmp":       ingest_lmp,
        "bess_dispatch":    run_bess_dispatch,
        "update_dns":       update_dns,
    }

    if task not in dispatch:
        return _err(f"Unknown task '{task}'. Valid: {list(dispatch)}")

    return dispatch[task](event, context)
