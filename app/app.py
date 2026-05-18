"""
app.py — NYISO Energy Dashboard
--------------------------------
Run with:  streamlit run app/app.py

Data source priority:
  1. Postgres (DATABASE_URL env var set)  — live cloud deployment
  2. Local parquets (data/processed/)     — local development
  3. Synthetic data                       — demo / Streamlit Cloud fallback

Features:
  - Zone selector
  - Date range filter
  - Live load vs forecast
  - LMP heatmap by zone
  - DA/RT spread tracker
  - Fuel mix breakdown
  - BESS dispatch recommendation
  - Key metrics cards
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config — must be first Streamlit call ────────────────────────────────
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
    .metric-label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .stSelectbox label, .stDateInput label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
DATABASE_URL = os.environ.get("DATABASE_URL")


def _load_from_postgres() -> tuple:
    """Query Postgres for the last 90 days of data."""
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=548)
    cutoff_str = cutoff.isoformat()

    df_load = pd.read_sql("""
        SELECT timestamp, zone, load_mw FROM load_actual
        WHERE timestamp >= %(cutoff)s ORDER BY timestamp
    """, conn, params={"cutoff": cutoff_str})

    df_lmp = pd.read_sql("""
        SELECT timestamp, zone, lmp_total FROM lmp_dayahead
        WHERE timestamp >= %(cutoff)s ORDER BY timestamp
    """, conn, params={"cutoff": cutoff_str})

    df_fuel = pd.read_sql("""
        SELECT timestamp, fuel_type, gen_mw FROM fuel_mix
        WHERE timestamp >= %(cutoff)s ORDER BY timestamp
    """, conn, params={"cutoff": cutoff_str})

    df_lmp_rt = pd.read_sql("""
        SELECT timestamp, zone, lmp_total AS lmp_rt FROM lmp_realtime
        WHERE timestamp >= %(cutoff)s ORDER BY timestamp
    """, conn, params={"cutoff": cutoff_str})

    df_load_fcst = pd.read_sql("""
        SELECT timestamp, zone, load_forecast_mw FROM load_forecast
        WHERE timestamp >= %(cutoff)s ORDER BY timestamp
    """, conn, params={"cutoff": cutoff_str})

    df_lmp_fcst = pd.read_sql("""
        SELECT timestamp, zone, lmp_forecast FROM lmp_forecast
        WHERE timestamp >= %(cutoff)s ORDER BY timestamp
    """, conn, params={"cutoff": cutoff_str})

    df_bess = pd.read_sql("""
        SELECT timestamp, charge_mw, discharge_mw, soc_mwh, lmp_used, revenue_usd
        FROM bess_dispatch
        WHERE run_date = CURRENT_DATE
        ORDER BY timestamp
    """, conn)

    conn.close()

    for df in [df_load, df_lmp, df_fuel, df_lmp_rt, df_load_fcst, df_lmp_fcst, df_bess]:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("US/Eastern")

    df_system = (
        df_load.groupby("timestamp")["load_mw"]
        .sum().reset_index()
        .rename(columns={"load_mw": "total_load_mw"})
    )

    return df_load, df_lmp, df_fuel, df_system, df_lmp_rt, df_load_fcst, df_lmp_fcst, df_bess


def _load_from_parquets() -> tuple:
    """Load from local processed parquets."""
    df_load   = pd.read_parquet(os.path.join(DATA_DIR, 'load_actual.parquet'))
    df_lmp    = pd.read_parquet(os.path.join(DATA_DIR, 'lmp_dayahead.parquet'))
    df_fuel   = pd.read_parquet(os.path.join(DATA_DIR, 'fuel_mix.parquet'))
    df_system = pd.read_parquet(os.path.join(DATA_DIR, 'system_load.parquet'))

    df_lmp_rt     = pd.read_parquet(os.path.join(DATA_DIR, 'lmp_realtime.parquet')).rename(columns={"lmp_total": "lmp_rt"})
    df_load_fcst  = pd.DataFrame()
    df_lmp_fcst   = pd.DataFrame()
    df_bess       = pd.DataFrame()

    # Normalize DA LMP column if PTID uppercase
    if "PTID" in df_lmp.columns:
        df_lmp = df_lmp.rename(columns={"PTID": "ptid"})

    return df_load, df_lmp, df_fuel, df_system, df_lmp_rt, df_load_fcst, df_lmp_fcst, df_bess


def _load_synthetic() -> tuple:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from generate_synthetic import make_synthetic_data
    from datetime import datetime
    df_load, df_lmp, df_fuel = make_synthetic_data(datetime(2024, 1, 1), datetime(2024, 3, 31))
    df_system = (
        df_load.groupby('timestamp')['load_mw']
        .sum().reset_index()
        .rename(columns={'load_mw': 'total_load_mw'})
    )
    return df_load, df_lmp, df_fuel, df_system, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=300)
def load_data():
    """
    Load data from the best available source.
    Returns (df_load, df_lmp, df_fuel, df_system,
             df_lmp_rt, df_load_fcst, df_lmp_fcst, df_bess,
             source_label, using_synthetic).
    """
    if DATABASE_URL:
        try:
            result = _load_from_postgres()
            return (*result, "Postgres (live)", False)
        except Exception as e:
            st.warning(f"Postgres unavailable ({e}), falling back to parquets.")

    try:
        result = _load_from_parquets()
        return (*result, "Local parquets", False)
    except FileNotFoundError:
        result = _load_synthetic()
        return (*result, "Synthetic data", True)

df_load, df_lmp, df_fuel, df_system, df_lmp_rt, df_load_fcst, df_lmp_fcst, df_bess, source_label, using_synthetic = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NYISO Dashboard")
    st.markdown("---")

    st.caption(f"Data source: **{source_label}**")
    if using_synthetic:
        st.warning("Using synthetic data. Run the EDA notebook to generate real data.")

    all_zones = sorted(df_load['zone'].unique().tolist())
    preferred = ["N.Y.C.", "LONGIL", "CAPITL"]
    selected_zones = st.multiselect(
        "Zones to display",
        options=all_zones,
        default=[z for z in preferred if z in all_zones] or all_zones[:3]
    )

    min_date = df_system['timestamp'].min().date()
    max_date = df_system['timestamp'].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "Data sourced from [NYISO public CSV API](https://www.nyiso.com/custom-reports). "
        "No API key required."
    )

# ── Date filtering ─────────────────────────────────────────────────────────────
if len(date_range) == 2:
    start_ts = pd.Timestamp(date_range[0]).tz_localize('US/Eastern')
    end_ts   = pd.Timestamp(date_range[1]).tz_localize('US/Eastern') + pd.Timedelta(days=1)
else:
    start_ts = df_system['timestamp'].min()
    end_ts   = df_system['timestamp'].max()

def _filt(df, zone_filter=False):
    mask = (df['timestamp'] >= start_ts) & (df['timestamp'] < end_ts)
    if zone_filter and 'zone' in df.columns:
        mask &= df['zone'].isin(selected_zones)
    return df[mask]

sys_filt   = _filt(df_system)
load_filt  = _filt(df_load, zone_filter=True)
lmp_filt   = _filt(df_lmp,  zone_filter=True)
fuel_filt  = _filt(df_fuel)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("NYISO Energy Market Dashboard")
st.markdown(f"Showing **{len(sys_filt):,}** system-level observations | Zones: {', '.join(selected_zones)}")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    peak = sys_filt['total_load_mw'].max() if not sys_filt.empty else 0
    st.metric("Peak Load", f"{peak/1000:.1f} GW")

with col2:
    avg_load = sys_filt['total_load_mw'].mean() if not sys_filt.empty else 0
    st.metric("Avg System Load", f"{avg_load/1000:.1f} GW")

with col3:
    if not lmp_filt.empty:
        avg_lmp = lmp_filt['lmp_total'].mean()
        st.metric("Avg DA LMP (selected)", f"${avg_lmp:.2f}/MWh")

with col4:
    if not lmp_filt.empty:
        spike_count = (lmp_filt['lmp_total'] > lmp_filt['lmp_total'].quantile(0.99)).sum()
        st.metric("Price Spikes (>99th pct)", f"{spike_count}")

st.markdown("---")

# ── Row 1: Load actual vs forecast ────────────────────────────────────────────
st.subheader("System Load — Actual vs Forecast")

fig_load = go.Figure()
fig_load.add_trace(go.Scatter(
    x=sys_filt['timestamp'], y=sys_filt['total_load_mw'],
    name='Actual', line=dict(color='#00d4ff', width=1.2)
))

if not df_load_fcst.empty:
    fcst_sys = (
        _filt(df_load_fcst, zone_filter=True)
        .groupby('timestamp')['load_forecast_mw'].sum().reset_index()
    )
    fig_load.add_trace(go.Scatter(
        x=fcst_sys['timestamp'], y=fcst_sys['load_forecast_mw'],
        name='XGBoost Forecast', line=dict(color='#ff6b35', width=1.5, dash='dot')
    ))

fig_load.update_layout(
    template='plotly_dark', height=280, margin=dict(t=10, b=10),
    yaxis_title='MW', xaxis_title='',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
st.plotly_chart(fig_load, use_container_width=True)

# ── Row 2: DA LMP heatmap + Fuel Mix ──────────────────────────────────────────
col_lmp, col_fuel = st.columns(2)

with col_lmp:
    st.subheader("DA LMP by Zone")
    if not lmp_filt.empty:
        lmp_wide = (
            lmp_filt
            .pivot_table(index='timestamp', columns='zone', values='lmp_total', aggfunc='mean')
            .reset_index()
        )
        zone_cols = [c for c in lmp_wide.columns if c != 'timestamp']
        fig_lmp = px.line(
            lmp_wide, x='timestamp', y=zone_cols,
            labels={'value': '$/MWh', 'timestamp': '', 'variable': 'Zone'},
            template='plotly_dark',
        )
        fig_lmp.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_lmp, use_container_width=True)
    else:
        st.info("No LMP data for selected zones/dates.")

with col_fuel:
    st.subheader("Generation Fuel Mix")
    if not fuel_filt.empty:
        fuel_avg = (
            fuel_filt.groupby('fuel_type')['gen_mw']
            .mean().sort_values(ascending=False).reset_index()
        )
        fig_fuel = px.pie(
            fuel_avg, values='gen_mw', names='fuel_type',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_fuel.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_fuel, use_container_width=True)

# ── Row 3: DA/RT Spread tracker ───────────────────────────────────────────────
st.subheader("DA / RT LMP Spread — N.Y.C.")

if not df_lmp_rt.empty and not df_lmp.empty:
    nyc_da = df_lmp[df_lmp['zone'] == 'N.Y.C.'][['timestamp', 'lmp_total']].rename(columns={'lmp_total': 'da'})
    nyc_rt = df_lmp_rt[df_lmp_rt['zone'] == 'N.Y.C.'][['timestamp', 'lmp_rt']] if 'zone' in df_lmp_rt.columns else df_lmp_rt[['timestamp', 'lmp_rt']]
    spread = nyc_da.merge(nyc_rt, on='timestamp', how='inner')
    spread['spread'] = spread['da'] - spread['lmp_rt']
    spread = spread[(spread['timestamp'] >= start_ts) & (spread['timestamp'] < end_ts)]

    if not spread.empty:
        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(
            x=spread['timestamp'], y=spread['spread'],
            fill='tozeroy', fillcolor='rgba(0,212,255,0.15)',
            line=dict(color='#00d4ff', width=1),
            name='DA − RT spread'
        ))
        fig_spread.add_hline(y=0, line_color='#64748b', line_dash='dot')
        fig_spread.update_layout(
            template='plotly_dark', height=220, margin=dict(t=10, b=10),
            yaxis_title='$/MWh', xaxis_title=''
        )
        st.plotly_chart(fig_spread, use_container_width=True)
        avg_spread = spread['spread'].mean()
        st.caption(f"Mean DA−RT spread: ${avg_spread:.2f}/MWh (positive = DA > RT)")
    else:
        st.info("No overlapping DA and RT data for selected period.")
else:
    st.info("RT LMP data not available — spread tracker requires live Postgres connection.")

# ── Row 4: BESS Dispatch Today ────────────────────────────────────────────────
st.subheader("BESS Dispatch Recommendation — Today (N.Y.C.)")

if not df_bess.empty:
    col_bess1, col_bess2 = st.columns([2, 1])
    with col_bess1:
        fig_bess = go.Figure()
        fig_bess.add_trace(go.Bar(
            x=df_bess['timestamp'], y=df_bess['discharge_mw'],
            name='Discharge', marker_color='#00d4ff'
        ))
        fig_bess.add_trace(go.Bar(
            x=df_bess['timestamp'], y=-df_bess['charge_mw'],
            name='Charge', marker_color='#ff6b35'
        ))
        fig_bess.add_trace(go.Scatter(
            x=df_bess['timestamp'], y=df_bess['lmp_used'],
            name='DA LMP', yaxis='y2', line=dict(color='#ffd700', width=1.5)
        ))
        fig_bess.update_layout(
            template='plotly_dark', height=300, barmode='relative',
            margin=dict(t=10, b=10),
            yaxis=dict(title='MW'),
            yaxis2=dict(title='$/MWh', overlaying='y', side='right'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_bess, use_container_width=True)

    with col_bess2:
        total_rev  = df_bess['revenue_usd'].sum()
        charge_mwh = df_bess['charge_mw'].sum()
        disch_mwh  = df_bess['discharge_mw'].sum()
        st.metric("Today's Revenue", f"${total_rev:,.0f}")
        st.metric("Total Discharge", f"{disch_mwh:.0f} MWh")
        st.metric("Total Charge",    f"{charge_mwh:.0f} MWh")
else:
    st.info("No BESS dispatch plan for today. Lambda runs daily at 9am ET after DA market clears.")

# ── Row 5: Load heatmap ───────────────────────────────────────────────────────
st.subheader("Load Heatmap — Hour of Day × Day of Week")

if not sys_filt.empty:
    sys_h = sys_filt.copy()
    sys_h['hour'] = sys_h['timestamp'].dt.hour
    sys_h['dow']  = sys_h['timestamp'].dt.day_name()

    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heatmap_data = (
        sys_h
        .groupby(['dow','hour'])['total_load_mw']
        .mean()
        .reset_index()
        .pivot(index='dow', columns='hour', values='total_load_mw')
        .reindex([d for d in dow_order if d in sys_h['dow'].unique()])
    )

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f'{h:02d}:00' for h in range(24)],
        y=heatmap_data.index.tolist(),
        colorscale='Blues',
        colorbar=dict(title='MW')
    ))
    fig_heat.update_layout(
        template='plotly_dark', height=260, margin=dict(t=10, b=10),
        xaxis_title='Hour of Day', yaxis_title=''
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: NYISO Public CSV API · Built with Streamlit + Plotly · Project 04 of NYISO DS Portfolio")
