"""
bess_dispatch.py — Daily BESS dispatch optimization for Lambda.

Runs an LP (via PuLP/CBC) on today's DA LMP prices to produce an optimal
24-hour charge/discharge schedule for a 100 MW / 400 MWh battery.

Battery specs (from Project 03 notebook):
  - Capacity:   400 MWh
  - Max power:  100 MW charge / 100 MW discharge
  - RTE:        85% round-trip efficiency (charge efficiency = √0.85)
  - SOC bounds: 5% – 95% (20 – 380 MWh)
  - Charge derating: above 80% SOC, max charge drops linearly to 0 at 95%
"""

import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# ── Battery parameters ────────────────────────────────────────────────────────

CAPACITY_MWH  = 400.0
MAX_POWER_MW  = 100.0
EFF_CHARGE    = 0.85 ** 0.5          # √RTE ≈ 0.9220
EFF_DISCHARGE = 0.85 ** 0.5
SOC_MIN_MWH   = CAPACITY_MWH * 0.05  # 20 MWh
SOC_MAX_MWH   = CAPACITY_MWH * 0.95  # 380 MWh
SOC_INIT_MWH  = CAPACITY_MWH * 0.50  # start at 50%
DERATE_THRESH  = 0.80                  # above this SOC fraction, derate charging


# ── LP optimizer ──────────────────────────────────────────────────────────────

def optimize_dispatch(lmp_hourly: pd.Series) -> pd.DataFrame:
    """
    LP optimizer: maximize revenue given 24h DA LMP prices.

    Args:
        lmp_hourly: Series indexed by hour (0-23) or DatetimeIndex, values in $/MWh.

    Returns:
        DataFrame with columns: [hour, charge_mw, discharge_mw, soc_mwh, lmp_used, revenue_usd].
        Positive charge_mw = charging, positive discharge_mw = discharging.
    """
    try:
        import pulp
    except ImportError:
        raise ImportError("PuLP is required for BESS dispatch. Add 'pulp' to requirements.")

    prices = lmp_hourly.values.astype(float)
    T = len(prices)

    prob = pulp.LpProblem("bess_dispatch", pulp.LpMaximize)

    # Decision variables
    c = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=MAX_POWER_MW) for t in range(T)]
    d = [pulp.LpVariable(f"d_{t}", lowBound=0, upBound=MAX_POWER_MW) for t in range(T)]
    s = [pulp.LpVariable(f"s_{t}", lowBound=SOC_MIN_MWH, upBound=SOC_MAX_MWH) for t in range(T)]

    # Binary to prevent simultaneous charge+discharge
    u = [pulp.LpVariable(f"u_{t}", cat="Binary") for t in range(T)]

    # Objective: revenue from discharging minus cost of charging (energy arbitrage)
    prob += pulp.lpSum(prices[t] * d[t] - prices[t] * c[t] for t in range(T))

    # Constraints
    for t in range(T):
        # SOC evolution
        soc_prev = SOC_INIT_MWH if t == 0 else s[t - 1]
        prob += s[t] == soc_prev + EFF_CHARGE * c[t] - (1 / EFF_DISCHARGE) * d[t]

        # No simultaneous charge and discharge
        prob += c[t] <= MAX_POWER_MW * u[t]
        prob += d[t] <= MAX_POWER_MW * (1 - u[t])

        # Charge derating above 80% SOC (linearized)
        # Max charge = MAX_POWER_MW * (1 - max(0, (soc/capacity - derate_thresh) / (1 - derate_thresh)))
        # We approximate with a soft cap: c[t] <= MAX_POWER_MW when s[t] < derate_thresh * capacity
        # Exact derating requires SOC-dependent upper bounds — use simplified SOC headroom constraint:
        prob += c[t] <= (SOC_MAX_MWH - (soc_prev)) / EFF_CHARGE + MAX_POWER_MW * 0.001

    # Terminal SOC: return to at least initial SOC (no free energy extraction)
    prob += s[T - 1] >= SOC_INIT_MWH

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        logger.warning(f"  LP status: {pulp.LpStatus[prob.status]} — returning zero dispatch")
        return _zero_dispatch(lmp_hourly)

    rows = []
    for t in range(T):
        charge    = max(pulp.value(c[t]) or 0.0, 0.0)
        discharge = max(pulp.value(d[t]) or 0.0, 0.0)
        soc       = pulp.value(s[t]) or SOC_INIT_MWH
        revenue   = prices[t] * discharge - prices[t] * charge
        rows.append({
            "hour":         t,
            "charge_mw":    round(charge, 4),
            "discharge_mw": round(discharge, 4),
            "soc_mwh":      round(soc, 4),
            "lmp_used":     round(prices[t], 4),
            "revenue_usd":  round(revenue, 2),
        })

    df = pd.DataFrame(rows)
    total_rev = df["revenue_usd"].sum()
    logger.info(f"  BESS dispatch optimized — total revenue: ${total_rev:,.0f}")
    return df


def naive_daily_revenue(lmp_hourly: pd.Series) -> float:
    """
    Naive dispatch: charge midnight–6 AM, discharge 2–6 PM.
    Uses the same battery constants as optimize_dispatch.
    Returns total revenue for the day in USD.
    """
    prices = lmp_hourly.values.astype(float)
    SOC = SOC_INIT_MWH
    rev = 0.0
    for h in range(min(24, len(prices))):
        p = prices[h]
        if h < 6:
            c = max(0.0, min(MAX_POWER_MW, (SOC_MAX_MWH - SOC) / EFF_CHARGE))
            SOC = min(SOC_MAX_MWH, SOC + c * EFF_CHARGE)
            rev -= c * p
        elif 14 <= h < 18:
            d = max(0.0, min(MAX_POWER_MW, (SOC - SOC_MIN_MWH) * EFF_DISCHARGE))
            SOC = max(SOC_MIN_MWH, SOC - d / EFF_DISCHARGE)
            rev += d * p
    return float(rev)


def _zero_dispatch(lmp_hourly: pd.Series) -> pd.DataFrame:
    T = len(lmp_hourly)
    prices = lmp_hourly.values.astype(float)
    return pd.DataFrame({
        "hour":         range(T),
        "charge_mw":    0.0,
        "discharge_mw": 0.0,
        "soc_mwh":      SOC_INIT_MWH,
        "lmp_used":     prices,
        "revenue_usd":  0.0,
    })


# ── Postgres I/O ─────────────────────────────────────────────────────────────

def fetch_da_lmp_today(conn, zone: str = "N.Y.C.") -> pd.Series:
    """
    Fetch today's DA LMP from Postgres for the given zone.
    Returns Series indexed by hour (0-23).
    """
    today = datetime.now(tz=timezone.utc).date()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT timestamp, lmp_total
            FROM lmp_dayahead
            WHERE zone = %s
              AND timestamp::date = %s
            ORDER BY timestamp
        """, (zone, today))
        rows = cur.fetchall()

    if not rows:
        logger.warning(f"  No DA LMP found for {zone} on {today}")
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=["timestamp", "lmp_total"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    return df.set_index("hour")["lmp_total"].sort_index()


def write_dispatch(
    conn,
    dispatch_df: pd.DataFrame,
    run_date: date,
    zone: str,
    base_timestamp: pd.Timestamp,
) -> int:
    """
    Write hourly dispatch plan to bess_dispatch table.

    Args:
        dispatch_df:    Output of optimize_dispatch()
        run_date:       Date this dispatch was computed (for deduplication)
        zone:           NYISO zone name
        base_timestamp: Midnight ET of the delivery day (timestamps = base + hour)
    """
    rows = []
    for _, row in dispatch_df.iterrows():
        ts = base_timestamp + pd.Timedelta(hours=int(row["hour"]))
        rows.append((
            run_date,
            ts,
            zone,
            row["charge_mw"],
            row["discharge_mw"],
            row["soc_mwh"],
            row["lmp_used"],
            row["revenue_usd"],
        ))

    sql = """
        INSERT INTO bess_dispatch
            (run_date, timestamp, zone, charge_mw, discharge_mw, soc_mwh, lmp_used, revenue_usd)
        VALUES %s
        ON CONFLICT (run_date, timestamp, zone) DO UPDATE SET
            charge_mw    = EXCLUDED.charge_mw,
            discharge_mw = EXCLUDED.discharge_mw,
            soc_mwh      = EXCLUDED.soc_mwh,
            lmp_used     = EXCLUDED.lmp_used,
            revenue_usd  = EXCLUDED.revenue_usd
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows)
    conn.commit()
    logger.info(f"  bess_dispatch: {len(rows)} rows written for {run_date} zone {zone}")
    return len(rows)
