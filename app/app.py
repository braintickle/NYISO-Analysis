"""
app.py — NYISO Energy Dashboard

Data source priority:
  1. Postgres (DATABASE_URL env var)  — live cloud deployment
  2. Local parquets (data/processed/) — local development
  3. Synthetic data                   — demo fallback

All queries hardcoded to last 365 days. No date range picker.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Lambda path: import BESS optimizer from lambda/bess_dispatch.py ──────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "lambda"))
try:
    from bess_dispatch import optimize_dispatch
    _BESS_OK = True
except ImportError:
    _BESS_OK = False

# ── Page config — must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="NYISO Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .metric-card {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 12px; color: #64748b; text-transform: uppercase;
                    letter-spacing: 1px; margin-top: 4px; }
    .stSelectbox label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CUTOFF_DAYS  = 365


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pg():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def _to_et(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("US/Eastern")
    return df


def _cutoff() -> str:
    return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=CUTOFF_DAYS)).isoformat()


# ── Historical data (cached 5 min) ────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    """
    Returns (df_load, df_lmp, df_fuel, df_system, df_lmp_rt,
             df_lmp_fcst, df_load_fcst, source_label, using_synthetic).
    """
    if DATABASE_URL:
        try:
            data = _from_postgres()
            return (*data, "Postgres (live)", False)
        except Exception as e:
            st.warning(f"Postgres unavailable ({e}), falling back to parquets.")
    try:
        data = _from_parquets()
        return (*data, "Local parquets", False)
    except FileNotFoundError:
        data = _from_synthetic()
        return (*data, "Synthetic data", True)


def _from_postgres():
    conn = _pg()
    p = {"c": _cutoff()}

    df_load = _to_et(pd.read_sql(
        "SELECT timestamp, zone, load_mw FROM load_actual "
        "WHERE timestamp >= %(c)s ORDER BY timestamp",
        conn, params=p))

    df_lmp = _to_et(pd.read_sql(
        "SELECT timestamp, zone, lmp_total FROM lmp_dayahead "
        "WHERE timestamp >= %(c)s ORDER BY timestamp",
        conn, params=p))

    df_fuel = _to_et(pd.read_sql(
        "SELECT timestamp, fuel_type, gen_mw FROM fuel_mix "
        "WHERE timestamp >= %(c)s ORDER BY timestamp",
        conn, params=p))

    df_lmp_rt = _to_et(pd.read_sql(
        "SELECT timestamp, zone, lmp_total AS lmp_rt FROM lmp_realtime "
        "WHERE timestamp >= %(c)s ORDER BY timestamp",
        conn, params=p))

    # lmp_forecast has lmp_forecast (predicted) and lmp_actual (filled by backfill)
    df_lmp_fcst = _to_et(pd.read_sql("""
        SELECT timestamp, lmp_forecast, lmp_actual FROM lmp_forecast
        WHERE zone = 'N.Y.C.' AND timestamp >= %(c)s ORDER BY timestamp
    """, conn, params=p))

    # load_forecast has load_forecast_mw and load_actual_mw (filled by backfill)
    df_load_fcst = _to_et(pd.read_sql("""
        SELECT timestamp, zone, load_forecast_mw, load_actual_mw FROM load_forecast
        WHERE zone = 'N.Y.C.' AND timestamp >= %(c)s ORDER BY timestamp
    """, conn, params=p))

    conn.close()

    df_system = (
        df_load.groupby("timestamp")["load_mw"]
        .sum().reset_index()
        .rename(columns={"load_mw": "total_load_mw"})
    )
    return df_load, df_lmp, df_fuel, df_system, df_lmp_rt, df_lmp_fcst, df_load_fcst


def _from_parquets():
    df_load   = pd.read_parquet(os.path.join(DATA_DIR, "load_actual.parquet"))
    df_lmp    = pd.read_parquet(os.path.join(DATA_DIR, "lmp_dayahead.parquet"))
    df_fuel   = pd.read_parquet(os.path.join(DATA_DIR, "fuel_mix.parquet"))
    df_system = pd.read_parquet(os.path.join(DATA_DIR, "system_load.parquet"))
    df_lmp_rt = pd.read_parquet(os.path.join(DATA_DIR, "lmp_realtime.parquet")).rename(
        columns={"lmp_total": "lmp_rt"})
    if "PTID" in df_lmp.columns:
        df_lmp = df_lmp.rename(columns={"PTID": "ptid"})
    empty = pd.DataFrame()
    return df_load, df_lmp, df_fuel, df_system, df_lmp_rt, empty, empty


def _from_synthetic():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from generate_synthetic import make_synthetic_data
    from datetime import datetime as dt
    df_load, df_lmp, df_fuel = make_synthetic_data(dt(2024, 1, 1), dt(2024, 3, 31))
    df_system = (df_load.groupby("timestamp")["load_mw"].sum()
                 .reset_index().rename(columns={"load_mw": "total_load_mw"}))
    empty = pd.DataFrame()
    return df_load, df_lmp, df_fuel, df_system, empty, empty, empty


# ── Today's LMP for live BESS dispatch (cached 5 min) ────────────────────────

@st.cache_data(ttl=300)
def load_bess_prices():
    """
    Returns (prices: pd.Series[lmp], source_label: str).

    Tries lmp_forecast for today first (XGBoost predictions written by Lambda).
    Falls back to lmp_dayahead for today (actual DA prices, same source Lambda LP uses).
    """
    if not DATABASE_URL:
        return pd.Series(dtype=float), "no database"
    try:
        conn = _pg()
        today = str(pd.Timestamp.now(tz="US/Eastern").date())

        # Cast timestamp to ET date for correct midnight boundary
        df = pd.read_sql("""
            SELECT timestamp, lmp_forecast FROM lmp_forecast
            WHERE zone = 'N.Y.C.'
              AND (timestamp AT TIME ZONE 'US/Eastern')::date = %(t)s
            ORDER BY timestamp
        """, conn, params={"t": today})

        if not df.empty and len(df) >= 20:
            df = _to_et(df)
            conn.close()
            return df.set_index("timestamp")["lmp_forecast"], "XGBoost forecast"

        # Fallback: actual DA prices (Lambda's own BESS dispatch also uses these)
        df2 = pd.read_sql("""
            SELECT timestamp, lmp_total FROM lmp_dayahead
            WHERE zone = 'N.Y.C.'
              AND (timestamp AT TIME ZONE 'US/Eastern')::date = %(t)s
            ORDER BY timestamp
        """, conn, params={"t": today})
        conn.close()

        if not df2.empty:
            df2 = _to_et(df2)
            return df2.set_index("timestamp")["lmp_total"], "DA LMP (actual)"

        return pd.Series(dtype=float), "no data for today"
    except Exception as exc:
        return pd.Series(dtype=float), f"error: {exc}"


# ── Load everything ───────────────────────────────────────────────────────────

(df_load, df_lmp, df_fuel, df_system,
 df_lmp_rt, df_lmp_fcst, df_load_fcst,
 source_label, using_synthetic) = load_data()

bess_prices, bess_source = load_bess_prices()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NYISO Dashboard")
    st.markdown("---")
    st.caption(f"Data source: **{source_label}**")
    if using_synthetic:
        st.warning("Using synthetic data. Run the EDA notebook to generate real data.")

    all_zones = sorted(df_load["zone"].unique().tolist()) if not df_load.empty else []
    preferred = ["N.Y.C.", "LONGIL", "CAPITL"]
    selected_zones = st.multiselect(
        "Zones to display",
        options=all_zones,
        default=[z for z in preferred if z in all_zones] or all_zones[:3],
    )

    st.markdown("---")
    st.caption("Last 365 days · Refreshes every 5 min")
    st.markdown(
        "Data: [NYISO public CSV API](https://www.nyiso.com/custom-reports). "
        "No API key required."
    )


# ── Zone filter ───────────────────────────────────────────────────────────────

def _by_zone(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "zone" not in df.columns or not selected_zones:
        return df
    return df[df["zone"].isin(selected_zones)]


load_filt = _by_zone(df_load)
lmp_filt  = _by_zone(df_lmp)
fuel_filt = df_fuel   # fuel_mix has no zone column


# ── Header ────────────────────────────────────────────────────────────────────
st.title("NYISO Energy Market Dashboard")
zones_str = ", ".join(selected_zones) if selected_zones else "all zones"
st.markdown(f"Showing last 365 days · Zones: {zones_str}")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    peak = df_system["total_load_mw"].max() if not df_system.empty else 0
    st.metric("Peak Load", f"{peak / 1000:.1f} GW")

with col2:
    avg_load = df_system["total_load_mw"].mean() if not df_system.empty else 0
    st.metric("Avg System Load", f"{avg_load / 1000:.1f} GW")

with col3:
    avg_lmp = lmp_filt["lmp_total"].mean() if not lmp_filt.empty else 0
    st.metric("Avg DA LMP (selected)", f"${avg_lmp:.2f}/MWh")

with col4:
    if not lmp_filt.empty:
        spikes = (lmp_filt["lmp_total"] > lmp_filt["lmp_total"].quantile(0.99)).sum()
        st.metric("Price Spikes (>99th pct)", f"{spikes}")

st.markdown("---")

# ── System Load: Actual vs Forecast ──────────────────────────────────────────
st.subheader("System Load — Actual vs Forecast")

fig_load = go.Figure()
fig_load.add_trace(go.Scatter(
    x=df_system["timestamp"], y=df_system["total_load_mw"],
    name="Actual (all zones)", line=dict(color="#00d4ff", width=1.2),
))
if not df_load_fcst.empty and "load_forecast_mw" in df_load_fcst.columns:
    fig_load.add_trace(go.Scatter(
        x=df_load_fcst["timestamp"], y=df_load_fcst["load_forecast_mw"],
        name="XGBoost Forecast (N.Y.C.)", line=dict(color="#ff6b35", width=1.5, dash="dot"),
    ))
fig_load.update_layout(
    template="plotly_dark", height=280, margin=dict(t=10, b=10),
    yaxis_title="MW", xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_load, use_container_width=True)

# ── Forecast vs Actual — LMP ──────────────────────────────────────────────────
st.subheader("Forecast vs Actual — LMP (N.Y.C. DA)")

if not df_lmp_fcst.empty:
    nyc_da = (df_lmp[df_lmp["zone"] == "N.Y.C."][["timestamp", "lmp_total"]]
              if not df_lmp.empty else pd.DataFrame())

    fig_lv = go.Figure()
    if not nyc_da.empty:
        fig_lv.add_trace(go.Scatter(
            x=nyc_da["timestamp"], y=nyc_da["lmp_total"],
            name="Actual DA LMP", line=dict(color="#00d4ff", width=1.2),
        ))
    fcst_rows = df_lmp_fcst.dropna(subset=["lmp_forecast"])
    fig_lv.add_trace(go.Scatter(
        x=fcst_rows["timestamp"], y=fcst_rows["lmp_forecast"],
        name="XGBoost Forecast", line=dict(color="#ff6b35", width=1.5, dash="dot"),
    ))
    fig_lv.update_layout(
        template="plotly_dark", height=280, margin=dict(t=10, b=10),
        yaxis_title="$/MWh", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_lv, use_container_width=True)

    matched = df_lmp_fcst.dropna(subset=["lmp_forecast", "lmp_actual"])
    if not matched.empty:
        mape = (
            (matched["lmp_forecast"] - matched["lmp_actual"]).abs()
            / matched["lmp_actual"].abs().clip(lower=1)
        ).mean() * 100
        mae = (matched["lmp_forecast"] - matched["lmp_actual"]).abs().mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("LMP MAPE", f"{mape:.1f}%")
        m2.metric("LMP MAE", f"${mae:.2f}/MWh")
        m3.metric("Forecast–Actual pairs", f"{len(matched):,}")
else:
    st.info("No LMP forecast data yet — Lambda writes forecasts every hour.")

# ── Forecast vs Actual — Load ─────────────────────────────────────────────────
st.subheader("Forecast vs Actual — Load (N.Y.C.)")

if not df_load_fcst.empty:
    fig_lf = go.Figure()
    actual_rows = df_load_fcst.dropna(subset=["load_actual_mw"])
    if not actual_rows.empty:
        fig_lf.add_trace(go.Scatter(
            x=actual_rows["timestamp"], y=actual_rows["load_actual_mw"],
            name="Actual (hourly avg)", line=dict(color="#00d4ff", width=1.2),
        ))
    fcst_load = df_load_fcst.dropna(subset=["load_forecast_mw"])
    if not fcst_load.empty:
        fig_lf.add_trace(go.Scatter(
            x=fcst_load["timestamp"], y=fcst_load["load_forecast_mw"],
            name="XGBoost Forecast", line=dict(color="#ff6b35", width=1.5, dash="dot"),
        ))
    fig_lf.update_layout(
        template="plotly_dark", height=280, margin=dict(t=10, b=10),
        yaxis_title="MW", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_lf, use_container_width=True)

    matched_l = df_load_fcst.dropna(subset=["load_forecast_mw", "load_actual_mw"])
    if not matched_l.empty:
        mape_l = (
            (matched_l["load_forecast_mw"] - matched_l["load_actual_mw"]).abs()
            / matched_l["load_actual_mw"].abs().clip(lower=1)
        ).mean() * 100
        mae_l = (matched_l["load_forecast_mw"] - matched_l["load_actual_mw"]).abs().mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Load MAPE", f"{mape_l:.1f}%")
        m2.metric("Load MAE", f"{mae_l:.1f} MW")
        m3.metric("Forecast–Actual pairs", f"{len(matched_l):,}")
else:
    st.info("Load forecast data not available — load_forecast table is empty or Lambda has not run yet.")

# ── DA LMP by Zone + Fuel Mix ─────────────────────────────────────────────────
col_lmp, col_fuel = st.columns(2)

with col_lmp:
    st.subheader("DA LMP by Zone")
    if not lmp_filt.empty:
        lmp_wide = (
            lmp_filt
            .pivot_table(index="timestamp", columns="zone", values="lmp_total", aggfunc="mean")
            .reset_index()
        )
        zone_cols = [c for c in lmp_wide.columns if c != "timestamp"]
        fig_lmp = px.line(
            lmp_wide, x="timestamp", y=zone_cols,
            labels={"value": "$/MWh", "timestamp": "", "variable": "Zone"},
            template="plotly_dark",
        )
        fig_lmp.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_lmp, use_container_width=True)
    else:
        st.info("No LMP data for selected zones.")

with col_fuel:
    st.subheader("Generation Fuel Mix")
    if not fuel_filt.empty:
        fuel_avg = (
            fuel_filt.groupby("fuel_type")["gen_mw"]
            .mean().sort_values(ascending=False).reset_index()
        )
        fig_fuel = px.pie(
            fuel_avg, values="gen_mw", names="fuel_type",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_fuel.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_fuel, use_container_width=True)

# ── DA / RT Spread ────────────────────────────────────────────────────────────
st.subheader("DA / RT LMP Spread — N.Y.C.")

if not df_lmp_rt.empty and not df_lmp.empty:
    nyc_da_s = (df_lmp[df_lmp["zone"] == "N.Y.C."][["timestamp", "lmp_total"]]
                .rename(columns={"lmp_total": "da"}))
    nyc_rt_col = df_lmp_rt[df_lmp_rt["zone"] == "N.Y.C."] if "zone" in df_lmp_rt.columns else df_lmp_rt
    nyc_rt_s = nyc_rt_col[["timestamp", "lmp_rt"]]
    spread = nyc_da_s.merge(nyc_rt_s, on="timestamp", how="inner")
    spread["spread"] = spread["da"] - spread["lmp_rt"]

    if not spread.empty:
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(
            x=spread["timestamp"], y=spread["spread"],
            fill="tozeroy", fillcolor="rgba(0,212,255,0.15)",
            line=dict(color="#00d4ff", width=1), name="DA − RT spread",
        ))
        fig_sp.add_hline(y=0, line_color="#64748b", line_dash="dot")
        fig_sp.update_layout(
            template="plotly_dark", height=220, margin=dict(t=10, b=10),
            yaxis_title="$/MWh", xaxis_title="",
        )
        st.plotly_chart(fig_sp, use_container_width=True)
        st.caption(f"Mean DA−RT spread: ${spread['spread'].mean():.2f}/MWh  (positive = DA > RT)")
    else:
        st.info("No overlapping DA and RT timestamps.")
else:
    st.info("RT LMP data not available — spread tracker requires a Postgres connection.")

# ── BESS Dispatch — Today ─────────────────────────────────────────────────────
st.subheader("BESS Dispatch — Today (N.Y.C.)")

# Specs panel
s1, s2, s3, s4 = st.columns(4)
s1.metric("Capacity", "100 MW / 400 MWh")
s2.metric("Round-trip Efficiency", "85%")
s3.metric("SOC Bounds", "5–95%  (20–380 MWh)")
s4.metric("Charge Derating", "Linear above 80% SOC")

if not _BESS_OK:
    st.warning("BESS optimizer (PuLP) not importable in this environment.")
elif bess_prices.empty or len(bess_prices) < 4:
    st.info(f"No LMP data for today ({bess_source}) — dispatch unavailable until prices are published.")
else:
    lmp_series = pd.Series(bess_prices.values, dtype=float)
    dispatch_df = optimize_dispatch(lmp_series)

    # Align timestamps to ET hours
    today_et = pd.Timestamp.now(tz="US/Eastern").normalize()
    dispatch_df["timestamp"] = [
        today_et + pd.Timedelta(hours=int(h)) for h in dispatch_df["hour"]
    ]

    col_chart, col_rev = st.columns([3, 1])

    with col_chart:
        fig_bess = go.Figure()
        fig_bess.add_trace(go.Bar(
            x=dispatch_df["timestamp"], y=dispatch_df["discharge_mw"],
            name="Discharge (sell)", marker_color="#00d4ff",
        ))
        fig_bess.add_trace(go.Bar(
            x=dispatch_df["timestamp"], y=-dispatch_df["charge_mw"],
            name="Charge (buy)", marker_color="#ff6b35",
        ))
        fig_bess.add_trace(go.Scatter(
            x=dispatch_df["timestamp"], y=dispatch_df["lmp_used"],
            name=f"LMP ({bess_source})", yaxis="y2",
            line=dict(color="#ffd700", width=1.5),
        ))
        fig_bess.update_layout(
            template="plotly_dark", height=300, barmode="relative",
            margin=dict(t=10, b=10),
            yaxis=dict(title="MW  (+ discharge / − charge)"),
            yaxis2=dict(title="$/MWh", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bess, use_container_width=True)

    with col_rev:
        total_rev   = dispatch_df["revenue_usd"].sum()
        total_disch = dispatch_df["discharge_mw"].sum()
        total_chg   = dispatch_df["charge_mw"].sum()
        st.metric("Expected Revenue", f"${total_rev:,.0f}")
        st.metric("Total Discharge",  f"{total_disch:.0f} MWh")
        st.metric("Total Charge",     f"{total_chg:.0f} MWh")
        st.caption(f"Price source: {bess_source}")
        st.caption("Optimizer: PuLP LP / CBC")

# ── Load Heatmap ──────────────────────────────────────────────────────────────
st.subheader("Load Heatmap — Hour of Day × Day of Week")

if not df_system.empty:
    sys_h = df_system.copy()
    sys_h["hour"] = sys_h["timestamp"].dt.hour
    sys_h["dow"]  = sys_h["timestamp"].dt.day_name()

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = (
        sys_h.groupby(["dow", "hour"])["total_load_mw"].mean().reset_index()
        .pivot(index="dow", columns="hour", values="total_load_mw")
        .reindex([d for d in dow_order if d in sys_h["dow"].unique()])
    )

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=heatmap_data.index.tolist(),
        colorscale="Blues",
        colorbar=dict(title="MW"),
    ))
    fig_heat.update_layout(
        template="plotly_dark", height=260, margin=dict(t=10, b=10),
        xaxis_title="Hour of Day", yaxis_title="",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: NYISO Public CSV API · Built with Streamlit + Plotly · Project 04 of NYISO DS Portfolio"
)
