"""
generate_synthetic.py
---------------------
Generates realistic NYISO-like data for offline development and testing.

The synthetic data mimics real NYISO patterns:
  - Load: seasonal trend + weekday/weekend + hourly profile + noise
  - LMP:  correlated with load, zone-differentiated, with occasional spikes
  - Fuel: realistic mix proportions that shift by hour

Use this when you don't want to hit the live API (demos, CI, development).
Set USE_SYNTHETIC = True in the notebook config.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


NYISO_ZONES = [
    "CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL",
    "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"
]

# Relative size of each zone (N.Y.C. and LONGIL are largest)
ZONE_WEIGHTS = {
    "N.Y.C.":  0.28,
    "LONGIL":  0.18,
    "CAPITL":  0.08,
    "CENTRL":  0.09,
    "DUNWOD":  0.07,
    "GENESE":  0.07,
    "HUD VL":  0.06,
    "MHK VL":  0.04,
    "MILLWD":  0.04,
    "NORTH":   0.05,
    "WEST":    0.04,
}

# Zone price multipliers (NYC and LI pay more due to congestion)
ZONE_PRICE_MULT = {
    "N.Y.C.":  1.35,
    "LONGIL":  1.25,
    "DUNWOD":  1.10,
    "MILLWD":  1.05,
    "HUD VL":  1.00,
    "CAPITL":  0.95,
    "MHK VL":  0.93,
    "CENTRL":  0.90,
    "GENESE":  0.88,
    "NORTH":   0.85,
    "WEST":    0.82,
}

FUEL_TYPES = ["Natural Gas", "Nuclear", "Hydro", "Wind", "Other Fossil Fuels", "Solar", "Other Renewables"]


def _hourly_load_profile() -> np.ndarray:
    """Typical shape of electricity demand over 24 hours (normalized 0-1)."""
    return np.array([
        0.72, 0.68, 0.65, 0.63, 0.62, 0.65,  # 12am-5am: overnight trough
        0.73, 0.84, 0.92, 0.95, 0.96, 0.97,  # 6am-11am: morning ramp
        0.97, 0.96, 0.95, 0.94, 0.95, 0.97,  # 12pm-5pm: mid-day plateau
        1.00, 0.99, 0.97, 0.92, 0.85, 0.78   # 6pm-11pm: evening peak then decline
    ])


def make_synthetic_data(
    start: datetime,
    end: datetime,
    total_system_peak_mw: float = 26000,
    base_lmp: float = 35.0,
    random_seed: int = 42
) -> tuple:
    """
    Generate synthetic load, LMP, and fuel mix DataFrames.

    Args:
        start               : Start datetime
        end                 : End datetime (inclusive)
        total_system_peak_mw: Approximate peak system load in MW
        base_lmp            : Base LMP in $/MWh before adjustments
        random_seed         : For reproducibility

    Returns:
        Tuple of (df_load, df_lmp, df_fuel) DataFrames matching
        the schema output by clean.clean()
    """
    rng = np.random.default_rng(random_seed)
    hourly_profile = _hourly_load_profile()

    # Build hourly timestamp index
    timestamps = pd.date_range(start=start, end=end + timedelta(days=1), freq="h", inclusive="left")
    n = len(timestamps)

    # ── Features ──────────────────────────────────────────────────────────────
    hours   = timestamps.hour.values
    dows    = timestamps.dayofweek.values       # 0=Monday ... 6=Sunday
    months  = timestamps.month.values
    is_wknd = (dows >= 5).astype(float)

    # Seasonal factor: higher summer (cooling) and winter (heating) load
    seasonal = 1.0 + 0.12 * np.cos((months - 7) * np.pi / 6)

    # Day-of-week factor
    dow_factor = np.where(is_wknd, 0.87, 1.0)

    # Hourly profile factor
    hour_factor = hourly_profile[hours]

    # Random noise
    noise = rng.normal(0, 0.025, n)

    # System-level load signal (normalized)
    load_signal = hour_factor * seasonal * dow_factor * (1 + noise)

    # Scale to MW
    system_load = load_signal * total_system_peak_mw

    # ── Build Load DataFrame ──────────────────────────────────────────────────
    load_rows = []
    for zone, weight in ZONE_WEIGHTS.items():
        zone_noise = rng.normal(0, 0.015, n)
        zone_load  = system_load * weight * (1 + zone_noise)
        load_rows.append(pd.DataFrame({
            "timestamp": timestamps,
            "zone":      zone,
            "load_mw":   np.maximum(zone_load, 50),   # floor at 50 MW
        }))
    df_load = pd.concat(load_rows, ignore_index=True)
    df_load["timestamp"] = df_load["timestamp"].dt.tz_localize("US/Eastern", ambiguous="infer", nonexistent="shift_forward")

    # ── Build LMP DataFrame ───────────────────────────────────────────────────
    # LMP correlates with load + adds congestion premium + occasional spikes
    lmp_base = base_lmp * load_signal  # scales with system demand

    # Inject ~1% price spikes (scarcity events)
    spike_mask = rng.random(n) < 0.01
    lmp_base[spike_mask] *= rng.uniform(5, 20, spike_mask.sum())

    lmp_rows = []
    for zone, mult in ZONE_PRICE_MULT.items():
        zone_price_noise = rng.normal(0, 1.5, n)
        congestion       = rng.exponential(0.5, n) if zone in ("N.Y.C.", "LONGIL") else rng.exponential(0.1, n)
        lmp_total        = lmp_base * mult + zone_price_noise
        lmp_rows.append(pd.DataFrame({
            "timestamp":       timestamps,
            "zone":            zone,
            "lmp_total":       lmp_total,
            "lmp_losses":      lmp_total * 0.03 + rng.normal(0, 0.3, n),
            "lmp_congestion":  congestion,
        }))
    df_lmp = pd.concat(lmp_rows, ignore_index=True)
    df_lmp["timestamp"] = df_lmp["timestamp"].dt.tz_localize("US/Eastern", ambiguous="infer", nonexistent="shift_forward")

    # ── Build Fuel Mix DataFrame ──────────────────────────────────────────────
    # Proportions shift by hour: gas peaks mid-day, nuclear is flat, wind peaks overnight
    fuel_rows = []
    for ts, load, hour in zip(timestamps, system_load, hours):
        gas_frac     = 0.35 + 0.15 * hour_factor[hour] + rng.normal(0, 0.03)
        nuclear_frac = 0.28 + rng.normal(0, 0.01)
        hydro_frac   = 0.18 + rng.normal(0, 0.02)
        wind_frac    = 0.08 + 0.05 * (1 - hour_factor[hour]) + rng.normal(0, 0.01)  # wind stronger overnight
        solar_frac   = max(0, 0.05 * np.sin(np.pi * max(0, hour - 6) / 12) + rng.normal(0, 0.01))
        other_re     = 0.03
        other_ff     = max(0, 1 - gas_frac - nuclear_frac - hydro_frac - wind_frac - solar_frac - other_re)

        fracs = [gas_frac, nuclear_frac, hydro_frac, wind_frac, other_ff, solar_frac, other_re]
        total = sum(fracs)
        for fuel, frac in zip(FUEL_TYPES, fracs):
            fuel_rows.append({
                "timestamp": ts,
                "fuel_type": fuel,
                "gen_mw":    max(0, (frac / total) * load),
            })

    df_fuel = pd.DataFrame(fuel_rows)
    df_fuel["timestamp"] = df_fuel["timestamp"].dt.tz_localize("US/Eastern", ambiguous="infer", nonexistent="shift_forward")

    print(f"Synthetic data generated: {len(timestamps)} hours ({start.date()} → {end.date()})")
    print(f"  Load:     {len(df_load):,} rows across {len(NYISO_ZONES)} zones")
    print(f"  LMP:      {len(df_lmp):,} rows")
    print(f"  Fuel mix: {len(df_fuel):,} rows")

    return df_load, df_lmp, df_fuel
