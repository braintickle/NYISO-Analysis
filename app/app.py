"""
app.py — NYISO Energy Dashboard
--------------------------------
Run with:  streamlit run app/app.py

A simple but polished Streamlit dashboard for exploring NYISO data.
Uses processed parquet files from the EDA notebook.

Features:
  - Zone selector
  - Date range filter
  - Load time series
  - LMP price heatmap
  - Fuel mix breakdown
  - Key metrics cards
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

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

@st.cache_data
def load_data():
    """Load processed parquet files. Falls back to synthetic data if not found."""
    try:
        df_load   = pd.read_parquet(os.path.join(DATA_DIR, 'load_actual.parquet'))
        df_lmp    = pd.read_parquet(os.path.join(DATA_DIR, 'lmp_dayahead.parquet'))
        df_fuel   = pd.read_parquet(os.path.join(DATA_DIR, 'fuel_mix.parquet'))
        df_system = pd.read_parquet(os.path.join(DATA_DIR, 'system_load.parquet'))
        return df_load, df_lmp, df_fuel, df_system, False
    except FileNotFoundError:
        # Generate synthetic data for demo purposes
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from generate_synthetic import make_synthetic_data
        from datetime import datetime
        df_load, df_lmp, df_fuel = make_synthetic_data(datetime(2024, 1, 1), datetime(2024, 3, 31))
        df_system = (
            df_load.groupby('timestamp')['load_mw']
            .sum().reset_index()
            .rename(columns={'load_mw': 'total_load_mw'})
        )
        df_system['hour']        = df_system['timestamp'].dt.hour
        df_system['day_of_week'] = df_system['timestamp'].dt.day_name()
        df_system['is_weekend']  = df_system['timestamp'].dt.dayofweek >= 5
        return df_load, df_lmp, df_fuel, df_system, True

df_load, df_lmp, df_fuel, df_system, using_synthetic = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NYISO Dashboard")
    st.markdown("---")

    if using_synthetic:
        st.warning("⚠️ Using synthetic data. Run the EDA notebook to generate real data.", icon="⚠️")

    # Zone selector
    all_zones = sorted(df_load['zone'].unique().tolist())
    selected_zones = st.multiselect(
        "Zones to display",
        options=all_zones,
        default=["N.Y.C.", "LONGIL", "CAPITL"]
    )

    # Date range
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

mask_sys  = (df_system['timestamp'] >= start_ts) & (df_system['timestamp'] < end_ts)
mask_load = (df_load['timestamp'] >= start_ts)   & (df_load['timestamp'] < end_ts)
mask_lmp  = (df_lmp['timestamp'] >= start_ts)    & (df_lmp['timestamp'] < end_ts)
mask_fuel = (df_fuel['timestamp'] >= start_ts)   & (df_fuel['timestamp'] < end_ts)

sys_filt  = df_system[mask_sys]
load_filt = df_load[mask_load & df_load['zone'].isin(selected_zones)]
lmp_filt  = df_lmp[mask_lmp & df_lmp['zone'].isin(selected_zones)]
fuel_filt = df_fuel[mask_fuel]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ NYISO Energy Market Dashboard")
st.markdown(f"Showing **{len(sys_filt):,}** hourly observations | Zones: {', '.join(selected_zones)}")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    peak = sys_filt['total_load_mw'].max()
    st.metric("Peak Load", f"{peak/1000:.1f} GW")

with col2:
    avg_load = sys_filt['total_load_mw'].mean()
    st.metric("Avg System Load", f"{avg_load/1000:.1f} GW")

with col3:
    if not lmp_filt.empty:
        avg_lmp = lmp_filt['lmp_total'].mean()
        st.metric("Avg LMP (selected zones)", f"${avg_lmp:.2f}/MWh")

with col4:
    if not lmp_filt.empty:
        spike_count = (lmp_filt['lmp_total'] > lmp_filt['lmp_total'].quantile(0.99)).sum()
        st.metric("Price Spikes (>99th pct)", f"{spike_count}")

st.markdown("---")

# ── Row 1: Load time series ───────────────────────────────────────────────────
st.subheader("System Load Over Time")

fig_load = px.line(
    sys_filt,
    x='timestamp', y='total_load_mw',
    labels={'total_load_mw': 'Load (MW)', 'timestamp': ''},
    template='plotly_dark',
    color_discrete_sequence=['#00d4ff']
)
fig_load.update_traces(line_width=1.2)
fig_load.update_layout(height=280, margin=dict(t=10, b=10))
st.plotly_chart(fig_load, use_container_width=True)

# ── Row 2: LMP + Fuel Mix ─────────────────────────────────────────────────────
col_lmp, col_fuel = st.columns(2)

with col_lmp:
    st.subheader("Day-Ahead LMP by Zone")
    if not lmp_filt.empty:
        lmp_wide = lmp_filt.pivot_table(index='timestamp', columns='zone', values='lmp_total', aggfunc='mean').reset_index()
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

# ── Row 3: Load heatmap ───────────────────────────────────────────────────────
st.subheader("Load Heatmap — Hour of Day × Day of Week")

sys_filt_copy = sys_filt.copy()
sys_filt_copy['hour'] = sys_filt_copy['timestamp'].dt.hour
sys_filt_copy['dow']  = sys_filt_copy['timestamp'].dt.day_name()

dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heatmap_data = (
    sys_filt_copy
    .groupby(['dow','hour'])['total_load_mw']
    .mean()
    .reset_index()
    .pivot(index='dow', columns='hour', values='total_load_mw')
    .reindex([d for d in dow_order if d in sys_filt_copy['dow'].unique()])
)

fig_heat = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=[f'{h:02d}:00' for h in range(24)],
    y=heatmap_data.index.tolist(),
    colorscale='Blues',
    colorbar=dict(title='MW')
))
fig_heat.update_layout(
    template='plotly_dark',
    height=260,
    margin=dict(t=10, b=10),
    xaxis_title='Hour of Day',
    yaxis_title=''
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: NYISO Public CSV API · Built with Streamlit + Plotly · Project 01 of NYISO DS Portfolio")
