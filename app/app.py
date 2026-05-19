"""
app.py — NYC Zone J BESS Optimizer
All queries hardcoded to N.Y.C. (Zone J). No zone selector.
Data source priority: Postgres → local parquets → synthetic.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "lambda"))
try:
    from bess_dispatch import optimize_dispatch
    _BESS_OK = True
except ImportError:
    _BESS_OK = False

st.set_page_config(
    page_title="NYC Zone J BESS Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    [data-testid="stMetricLabel"] { font-size: 12px !important; }
    [data-testid="stMetricValue"] { font-size: 22px !important; }
</style>
""", unsafe_allow_html=True)

DATABASE_URL = os.environ.get("DATABASE_URL")
CUTOFF_DAYS  = 365


def _pg():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def _to_et(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("US/Eastern")
    return df


def _cutoff() -> str:
    return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=CUTOFF_DAYS)).isoformat()


# ── Main data load (5-min cache) ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    if DATABASE_URL:
        try:
            return (*_from_postgres(), "Postgres (live)", False)
        except Exception as e:
            st.warning(f"Postgres unavailable ({e}), falling back to local data.")
    try:
        return (*_from_parquets(), "Local parquets", False)
    except FileNotFoundError:
        return (*_from_synthetic(), "Synthetic data", True)


def _from_postgres():
    conn = _pg()
    p = {"c": _cutoff()}

    df_load = _to_et(pd.read_sql(
        "SELECT timestamp, load_mw FROM load_actual "
        "WHERE zone = 'N.Y.C.' AND timestamp >= %(c)s ORDER BY timestamp",
        conn, params=p))

    df_lmp = _to_et(pd.read_sql(
        "SELECT timestamp, lmp_total FROM lmp_dayahead "
        "WHERE zone = 'N.Y.C.' AND timestamp >= %(c)s ORDER BY timestamp",
        conn, params=p))

    df_lmp_rt = _to_et(pd.read_sql(
        "SELECT timestamp, lmp_total AS lmp_rt FROM lmp_realtime "
        "WHERE zone = 'N.Y.C.' AND timestamp >= %(c)s ORDER BY timestamp",
        conn, params=p))

    df_fuel = _to_et(pd.read_sql(
        "SELECT timestamp, fuel_type, gen_mw FROM fuel_mix "
        "WHERE timestamp >= %(c)s ORDER BY timestamp",
        conn, params=p))

    df_lmp_fcst = _to_et(pd.read_sql(
        "SELECT timestamp, lmp_forecast, lmp_actual FROM lmp_forecast "
        "WHERE zone = 'N.Y.C.' AND timestamp >= %(c)s ORDER BY timestamp",
        conn, params=p))

    df_load_fcst = _to_et(pd.read_sql(
        "SELECT timestamp, load_forecast_mw, load_actual_mw FROM load_forecast "
        "WHERE zone = 'N.Y.C.' AND timestamp >= %(c)s ORDER BY timestamp",
        conn, params=p))

    conn.close()
    return df_load, df_lmp, df_lmp_rt, df_fuel, df_lmp_fcst, df_load_fcst


def _from_parquets():
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    df_load = pd.read_parquet(os.path.join(DATA_DIR, "load_actual.parquet"))
    df_lmp  = pd.read_parquet(os.path.join(DATA_DIR, "lmp_dayahead.parquet"))
    df_fuel = pd.read_parquet(os.path.join(DATA_DIR, "fuel_mix.parquet"))
    df_lmp_rt = pd.read_parquet(os.path.join(DATA_DIR, "lmp_realtime.parquet"))
    for df, col in [(df_lmp_rt, "lmp_total")]:
        if col in df.columns:
            df.rename(columns={col: "lmp_rt"}, inplace=True)
    for df in [df_load, df_lmp, df_lmp_rt]:
        if "zone" in df.columns:
            df = df[df["zone"] == "N.Y.C."]
    empty = pd.DataFrame()
    return df_load, df_lmp, df_lmp_rt, df_fuel, empty, empty


def _from_synthetic():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from generate_synthetic import make_synthetic_data
    from datetime import datetime as dt
    df_load, df_lmp, df_fuel = make_synthetic_data(dt(2024, 1, 1), dt(2024, 3, 31))
    for df in [df_load, df_lmp]:
        if "zone" in df.columns:
            df.drop(df[df["zone"] != "N.Y.C."].index, inplace=True)
    empty = pd.DataFrame()
    return df_load, df_lmp, empty, df_fuel, empty, empty


# ── Today's BESS prices (5-min cache) ────────────────────────────────────────

@st.cache_data(ttl=300)
def load_bess_prices():
    if not DATABASE_URL:
        return pd.Series(dtype=float), "no database"
    try:
        conn = _pg()
        today = str(pd.Timestamp.now(tz="US/Eastern").date())
        df = pd.read_sql("""
            SELECT timestamp, lmp_forecast FROM lmp_forecast
            WHERE zone = 'N.Y.C.'
              AND (timestamp AT TIME ZONE 'US/Eastern')::date = %(t)s
            ORDER BY timestamp
        """, conn, params={"t": today})
        if not df.empty and len(df) >= 20:
            conn.close()
            return _to_et(df).set_index("timestamp")["lmp_forecast"], "XGBoost forecast"
        df2 = pd.read_sql("""
            SELECT timestamp, lmp_total FROM lmp_dayahead
            WHERE zone = 'N.Y.C.'
              AND (timestamp AT TIME ZONE 'US/Eastern')::date = %(t)s
            ORDER BY timestamp
        """, conn, params={"t": today})
        conn.close()
        if not df2.empty:
            return _to_et(df2).set_index("timestamp")["lmp_total"], "DA LMP (actual)"
        return pd.Series(dtype=float), "no data for today"
    except Exception as exc:
        return pd.Series(dtype=float), f"error: {exc}"


# ── BESS revenue summary (read from Postgres, cached 1 hour) ─────────────────

@st.cache_data(ttl=3600)
def load_revenue_summary() -> pd.DataFrame:
    """Query pre-computed bess_revenue_summary. Lambda populates this daily."""
    if not DATABASE_URL:
        return pd.DataFrame()
    try:
        conn = _pg()
        df = pd.read_sql(
            "SELECT date, perfect_foresight_revenue, naive_revenue, lp_revenue "
            "FROM bess_revenue_summary ORDER BY date",
            conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ── Load everything ───────────────────────────────────────────────────────────

(df_load, df_lmp, df_lmp_rt, df_fuel,
 df_lmp_fcst, df_load_fcst,
 source_label, using_synthetic) = load_data()

bess_prices, bess_source = load_bess_prices()

now_et       = pd.Timestamp.now(tz="US/Eastern")
today_et     = now_et.date()
yesterday_et = (now_et - pd.Timedelta(days=1)).date()
yesterday_str = (now_et - pd.Timedelta(days=1)).strftime("%d/%m/%y")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ NYC Zone J BESS Optimizer")
    st.markdown("---")
    st.caption(f"Data source: **{source_label}**")
    if using_synthetic:
        st.warning("Using synthetic data.")
    st.markdown("---")
    st.caption("Zone: **N.Y.C. (Zone J)** · Last 365 days")
    st.markdown(
        "Data: [NYISO public CSV API](https://www.nyiso.com/custom-reports)"
    )


# ── Header ────────────────────────────────────────────────────────────────────

st.title("NYC Zone J BESS Optimizer")
st.caption(f"N.Y.C. (Zone J) · Last 365 days · {source_label}")

# ── KPI Cards — Today's Projections ──────────────────────────────────────────

today_lmp_f  = df_lmp_fcst[df_lmp_fcst["timestamp"].dt.date == today_et] \
               if not df_lmp_fcst.empty else pd.DataFrame()
today_load_f = df_load_fcst[df_load_fcst["timestamp"].dt.date == today_et] \
               if not df_load_fcst.empty else pd.DataFrame()

k1, k2, k3, k4 = st.columns(4)

with k1:
    val = f"{today_load_f['load_forecast_mw'].max():,.0f} MW" \
          if not today_load_f.empty else "Updating..."
    st.metric("Projected Peak Load", val)

with k2:
    val = f"{today_load_f['load_forecast_mw'].sum():,.0f} MWh" \
          if not today_load_f.empty else "Updating..."
    st.metric("Projected Total Load", val)

with k3:
    val = f"${today_lmp_f['lmp_forecast'].mean():.2f}/MWh" \
          if not today_lmp_f.empty else "Updating..."
    st.metric("Projected Avg LMP", val)

with k4:
    val = f"${today_lmp_f['lmp_forecast'].max():.2f}/MWh" \
          if not today_lmp_f.empty else "Updating..."
    st.metric("Projected Peak LMP", val)

if today_lmp_f.empty and today_load_f.empty:
    st.caption("Today's projections not yet available — Lambda writes forecasts hourly.")

st.markdown("---")


# ── N.Y.C. Load — Actual vs Forecast ─────────────────────────────────────────

st.subheader("Actual vs Forecast — N.Y.C. Load")

fig_load = go.Figure()
if not df_load.empty:
    load_hourly = (
        df_load.set_index("timestamp")["load_mw"]
        .resample("h").mean()
        .reset_index()
    )
    fig_load.add_trace(go.Scatter(
        x=load_hourly["timestamp"], y=load_hourly["load_mw"],
        name="Actual", line=dict(color="#00d4ff", width=1.2),
    ))
if not df_load_fcst.empty:
    fcst_all = df_load_fcst.dropna(subset=["load_forecast_mw"])
    fig_load.add_trace(go.Scatter(
        x=fcst_all["timestamp"], y=fcst_all["load_forecast_mw"],
        name="XGBoost Forecast (incl. next 24 h)",
        line=dict(color="#ff6b35", width=1.5, dash="dot"),
    ))
fig_load.update_layout(
    template="plotly_dark", height=280, margin=dict(t=10, b=10),
    yaxis_title="MW", xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_load, use_container_width=True)

matched_load = (df_load_fcst.dropna(subset=["load_forecast_mw", "load_actual_mw"])
                if not df_load_fcst.empty else pd.DataFrame())
if not matched_load.empty:
    mape_l = ((matched_load["load_forecast_mw"] - matched_load["load_actual_mw"]).abs()
              / matched_load["load_actual_mw"].abs().clip(lower=1)).mean() * 100
    mae_l  = (matched_load["load_forecast_mw"] - matched_load["load_actual_mw"]).abs().mean()
    m1, m2, _ = st.columns(3)
    m1.metric("Annual MAPE", f"{mape_l:.2f}%")
    m2.metric("Annual MAE",  f"{mae_l:.1f} MW")

st.markdown("---")


# ── LMP Forecast vs Actual ────────────────────────────────────────────────────

st.subheader("Actual vs Forecast — N.Y.C. DA LMP")

if not df_lmp_fcst.empty:
    fig_lf = go.Figure()

    actual_lmp_rows = df_lmp_fcst.dropna(subset=["lmp_actual"])
    if not actual_lmp_rows.empty:
        fig_lf.add_trace(go.Scatter(
            x=actual_lmp_rows["timestamp"], y=actual_lmp_rows["lmp_actual"],
            name="Actual DA LMP", line=dict(color="#00d4ff", width=1.2),
        ))

    fcst_lmp_rows = df_lmp_fcst.dropna(subset=["lmp_forecast"])
    if not fcst_lmp_rows.empty:
        fig_lf.add_trace(go.Scatter(
            x=fcst_lmp_rows["timestamp"], y=fcst_lmp_rows["lmp_forecast"],
            name="XGBoost Forecast (incl. next 24 h)",
            line=dict(color="#ff6b35", width=1.5, dash="dot"),
        ))

    fig_lf.update_layout(
        template="plotly_dark", height=280, margin=dict(t=10, b=10),
        yaxis_title="$/MWh", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_lf, use_container_width=True)

    matched_lmp = df_lmp_fcst.dropna(subset=["lmp_forecast", "lmp_actual"])
    if not matched_lmp.empty:
        mape_lmp = ((matched_lmp["lmp_forecast"] - matched_lmp["lmp_actual"]).abs()
                    / matched_lmp["lmp_actual"].abs().clip(lower=1)).mean() * 100
        mae_lmp  = (matched_lmp["lmp_forecast"] - matched_lmp["lmp_actual"]).abs().mean()
        m1, m2, _ = st.columns(3)
        m1.metric("Annual MAPE", f"{mape_lmp:.2f}%")
        m2.metric("Annual MAE",  f"${mae_lmp:.2f}/MWh")
else:
    st.info("LMP forecast data not yet available — Lambda writes forecasts every hour.")

st.markdown("---")


# ── Yesterday: DA LMP Actual vs Forecast  +  Fuel Mix ────────────────────────

col_yda, col_fuel_y = st.columns(2)

with col_yda:
    st.subheader(f"DA LMP — {yesterday_str}")
    st.caption("Actual (DA) vs XGBoost forecast for N.Y.C.")

    yest_da   = df_lmp[df_lmp["timestamp"].dt.date == yesterday_et] \
                if not df_lmp.empty else pd.DataFrame()
    yest_fcst = df_lmp_fcst[df_lmp_fcst["timestamp"].dt.date == yesterday_et] \
                if not df_lmp_fcst.empty else pd.DataFrame()

    if not yest_da.empty or not yest_fcst.empty:
        fig_yda = go.Figure()
        if not yest_da.empty:
            fig_yda.add_trace(go.Bar(
                x=yest_da["timestamp"].dt.hour,
                y=yest_da["lmp_total"],
                name="Actual DA LMP",
                marker_color="#00d4ff", opacity=0.85,
            ))
        if not yest_fcst.empty:
            yf = yest_fcst.dropna(subset=["lmp_forecast"])
            fig_yda.add_trace(go.Bar(
                x=yf["timestamp"].dt.hour,
                y=yf["lmp_forecast"],
                name="Forecast",
                marker_color="#ff6b35", opacity=0.85,
            ))
        fig_yda.update_layout(
            template="plotly_dark", height=300, barmode="group",
            margin=dict(t=10, b=10),
            xaxis_title="Hour of Day", yaxis_title="$/MWh",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_yda, use_container_width=True)
    else:
        st.info(f"No DA LMP data for {yesterday_str}.")

with col_fuel_y:
    st.subheader(f"Generation Mix — {yesterday_str}")

    yest_fuel = df_fuel[df_fuel["timestamp"].dt.date == yesterday_et] \
                if not df_fuel.empty else pd.DataFrame()

    if not yest_fuel.empty:
        fuel_avg = (yest_fuel.groupby("fuel_type")["gen_mw"]
                   .mean().sort_values(ascending=False).reset_index())
        fig_fuel = px.pie(
            fuel_avg, values="gen_mw", names="fuel_type",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_fuel.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_fuel, use_container_width=True)
    else:
        st.info(f"No fuel mix data for {yesterday_str}.")

st.markdown("---")


# ── DA / RT Spread ────────────────────────────────────────────────────────────

st.subheader("DA / RT LMP Spread — N.Y.C.")

if not df_lmp_rt.empty and not df_lmp.empty:
    spread = (df_lmp.rename(columns={"lmp_total": "da"})
              .merge(df_lmp_rt.rename(columns={"lmp_rt": "rt"}),
                     on="timestamp", how="inner"))
    spread["spread"] = spread["da"] - spread["rt"]

    if not spread.empty:
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(
            x=spread["timestamp"], y=spread["spread"],
            fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
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

st.markdown("---")


# ── BESS Section ──────────────────────────────────────────────────────────────

st.header("BESS — NYC Zone J Dispatch")

# Model Constraints
st.subheader("Model Constraints")
st.markdown("""
- **Max Power:** 100 MW charge / 100 MW discharge
- **Energy Capacity:** 400 MWh (4-hour duration battery)
- **Round-trip Efficiency:** 85% RTE (√0.85 ≈ 92.2% on each leg)
- **SOC Bounds:** 5–95% state of charge (20–380 MWh); initialized at 50% (200 MWh) each day
- **Charge Derating:** Linear power derating above 80% SOC to protect battery longevity
- **ICAP Revenue:** Real 2025 NYC Zone J strip auction prices; capacity commitment modeled in annual backtest
""")

# Today's Dispatch
st.subheader(f"Today's Optimal Dispatch — {now_et.strftime('%d/%m/%y')}")

if not _BESS_OK:
    st.warning("BESS optimizer (PuLP) not importable in this environment.")
elif bess_prices.empty or len(bess_prices) < 4:
    st.info(f"No LMP data for today ({bess_source}) — dispatch unavailable until prices are published.")
else:
    lmp_series  = pd.Series(bess_prices.values, dtype=float)
    dispatch_df = optimize_dispatch(lmp_series)

    today_midnight = pd.Timestamp(today_et).tz_localize("US/Eastern")
    dispatch_df["timestamp"] = [
        today_midnight + pd.Timedelta(hours=int(h)) for h in dispatch_df["hour"]
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
        st.metric("Expected Revenue",  f"${total_rev:,.0f}")
        st.metric("Total Discharge",   f"{total_disch:.0f} MWh")
        st.metric("Total Charge",      f"{total_chg:.0f} MWh")
        st.caption(f"Price source: {bess_source}")
        st.caption("Optimizer: PuLP LP / CBC")

# Annual Revenue Comparison Table
st.subheader("Annual Revenue Comparison")

_NB_LP    = 9_062_076
_NB_NAIVE = 5_572_918
_NB_DAYS  = 365

rev_df = load_revenue_summary()

if rev_df.empty:
    st.info(
        "Revenue data updating — Lambda computes this daily at 10am ET after DA prices clear. "
        "First run populates the full year."
    )
else:
    n_days   = len(rev_df)
    pf_total = float(rev_df["perfect_foresight_revenue"].sum())
    nv_total = float(rev_df["naive_revenue"].sum())
    pf_daily = pf_total / n_days
    nv_daily = nv_total / n_days
    date_min = pd.to_datetime(rev_df["date"]).min().strftime("%Y-%m-%d")
    date_max = pd.to_datetime(rev_df["date"]).max().strftime("%Y-%m-%d")

    table = pd.DataFrame(
        {
            "Perfect Foresight (LP)": [
                f"${pf_total:,.0f}",
                f"${pf_daily:,.0f}",
                f"+{(pf_total/nv_total - 1)*100:.1f}%" if nv_total > 0 else "—",
                "—",
            ],
            "Naive Strategy": [
                f"${nv_total:,.0f}",
                f"${nv_daily:,.0f}",
                "—",
                f"{(nv_total/pf_total - 1)*100:.1f}%" if pf_total > 0 else "—",
            ],
            "BESS Optimizer": [
                f"${_NB_LP:,.0f}",
                f"${_NB_LP // _NB_DAYS:,.0f}",
                f"+{(_NB_LP/_NB_NAIVE - 1)*100:.1f}%",
                "reference",
            ],
        },
        index=["Total Revenue", "Avg Daily Revenue", "vs Naive", "vs Perfect Foresight"],
    )
    st.dataframe(table, use_container_width=True)
    st.caption(
        f"Live data: {date_min} — {date_max} ({n_days} days) · "
        "Perfect Foresight: LP on actual DA prices · "
        "Naive: charge 0–6 AM, discharge 2–6 PM · "
        "BESS Optimizer: 2025 full-year backtest with ICAP revenue"
    )

st.markdown("---")


# ── Load Heatmap ──────────────────────────────────────────────────────────────

st.subheader("Load Pattern — Hour of Day × Day of Week")

if not df_load.empty:
    lh = df_load.copy()
    lh["hour"] = lh["timestamp"].dt.hour
    lh["dow"]  = lh["timestamp"].dt.day_name()

    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    hmap = (
        lh.groupby(["dow","hour"])["load_mw"].mean().reset_index()
        .pivot(index="dow", columns="hour", values="load_mw")
        .reindex([d for d in dow_order if d in lh["dow"].unique()])
    )
    fig_heat = go.Figure(data=go.Heatmap(
        z=hmap.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=hmap.index.tolist(),
        colorscale="Blues",
        colorbar=dict(title="MW"),
    ))
    fig_heat.update_layout(
        template="plotly_dark", height=260, margin=dict(t=10, b=10),
        xaxis_title="Hour of Day", yaxis_title="",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")


# ── About This Project ────────────────────────────────────────────────────────

with st.expander("About This Project"):
    st.markdown("""
### NYC Zone J BESS Optimizer

This dashboard is **Project 04** of a NYISO energy-market data-science portfolio, built to
demonstrate end-to-end skills across data engineering, machine learning, optimization, and
real-time cloud deployment.

**What it does**
- Ingests NYISO public market data (load, DA/RT LMP, fuel mix) every 5 minutes via AWS Lambda
- Runs XGBoost forecasts for load and LMP 24 hours ahead, written to Postgres each hour
- Solves a PuLP linear-program each morning to schedule a 100 MW / 400 MWh BESS for maximum
  energy-arbitrage revenue using that day's cleared DA LMP
- Displays all results live in this Streamlit dashboard, hosted on AWS ECS Fargate

**Technology stack**

| Layer | Tools |
|---|---|
| Ingestion | AWS Lambda (Python 3.11), EventBridge cron |
| Storage | Supabase Postgres (free tier) |
| ML | XGBoost (load + LMP forecasting), trained on 2024–2026 NYISO data |
| Optimization | PuLP LP / CBC solver (BESS dispatch) |
| Dashboard | Streamlit on AWS ECS Fargate |
| CI/CD | GitHub Actions → ECR → ECS |
| DNS / HTTPS | Cloudflare proxy + Lambda auto-update on task restart |

**Forecasting accuracy**
- Load MAPE: 2.31% (XGBoost vs. industry benchmark 2–4%)
- LMP MAPE: 11.60% (rolling R1, retrained every 30 days)

**BESS optimization results (2025 full-year backtest)**
- LP perfect foresight (energy only): **$9.06M** (+62.6% vs. naive)
- LP + ICAP revenue: **$14.61M**
- LP forecasted prices + ICAP: **$14.06M** ($550K forecast-error penalty, 3.8%)

**Data source:** [NYISO Public CSV API](https://www.nyiso.com/custom-reports) — no API key required.

**GitHub:** [braintickle/NYISO-Analysis](https://github.com/braintickle/NYISO-Analysis)
    """)


# ── Footer ────────────────────────────────────────────────────────────────────

st.caption(
    "Data: NYISO Public CSV API · Built with Streamlit + Plotly · "
    "Project 04 of NYISO DS Portfolio"
)
