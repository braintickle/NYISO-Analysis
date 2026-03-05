-- schema.sql
-- Supabase Postgres schema for NYISO Live Dashboard (Project 04)
-- All timestamps stored as TIMESTAMPTZ (UTC internally, NYISO native is US/Eastern)
-- Unique constraints on natural keys enable idempotent INSERT ... ON CONFLICT DO NOTHING

-- ── 1. load_actual ────────────────────────────────────────────────────────────
-- Source: NYISO pal endpoint | 5-min resolution | 11 zones
CREATE TABLE IF NOT EXISTS load_actual (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    zone        TEXT        NOT NULL,
    ptid        INTEGER,
    load_mw     DOUBLE PRECISION,
    is_outlier  BOOLEAN     DEFAULT FALSE,
    CONSTRAINT uq_load_actual UNIQUE (timestamp, zone)
);

CREATE INDEX IF NOT EXISTS idx_load_actual_ts   ON load_actual (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_load_actual_zone ON load_actual (zone, timestamp DESC);


-- ── 2. lmp_dayahead ───────────────────────────────────────────────────────────
-- Source: NYISO damlbmp endpoint | hourly | 11 zones
CREATE TABLE IF NOT EXISTS lmp_dayahead (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    zone            TEXT        NOT NULL,
    ptid            INTEGER,
    lmp_total       DOUBLE PRECISION,
    lmp_losses      DOUBLE PRECISION,
    lmp_congestion  DOUBLE PRECISION,
    is_outlier      BOOLEAN     DEFAULT FALSE,
    CONSTRAINT uq_lmp_dayahead UNIQUE (timestamp, zone)
);

CREATE INDEX IF NOT EXISTS idx_lmp_dayahead_ts   ON lmp_dayahead (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_lmp_dayahead_zone ON lmp_dayahead (zone, timestamp DESC);


-- ── 3. lmp_realtime ───────────────────────────────────────────────────────────
-- Source: NYISO rtlbmp endpoint | hourly | 11 zones
CREATE TABLE IF NOT EXISTS lmp_realtime (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    zone            TEXT        NOT NULL,
    ptid            INTEGER,
    lmp_total       DOUBLE PRECISION,
    lmp_losses      DOUBLE PRECISION,
    lmp_congestion  DOUBLE PRECISION,
    is_outlier      BOOLEAN     DEFAULT FALSE,
    CONSTRAINT uq_lmp_realtime UNIQUE (timestamp, zone)
);

CREATE INDEX IF NOT EXISTS idx_lmp_realtime_ts   ON lmp_realtime (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_lmp_realtime_zone ON lmp_realtime (zone, timestamp DESC);


-- ── 4. fuel_mix ───────────────────────────────────────────────────────────────
-- Source: NYISO rtfuelmix endpoint | 5-min | ~10 fuel types
CREATE TABLE IF NOT EXISTS fuel_mix (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    fuel_type   TEXT        NOT NULL,
    gen_mw      DOUBLE PRECISION,
    is_outlier  BOOLEAN     DEFAULT FALSE,
    CONSTRAINT uq_fuel_mix UNIQUE (timestamp, fuel_type)
);

CREATE INDEX IF NOT EXISTS idx_fuel_mix_ts        ON fuel_mix (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_fuel_mix_fuel_type ON fuel_mix (fuel_type, timestamp DESC);


-- ── 5. system_load ────────────────────────────────────────────────────────────
-- Source: derived from load_actual (system-wide aggregate) | 5-min
CREATE TABLE IF NOT EXISTS system_load (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    total_load_mw   DOUBLE PRECISION,
    CONSTRAINT uq_system_load UNIQUE (timestamp)
);

CREATE INDEX IF NOT EXISTS idx_system_load_ts ON system_load (timestamp DESC);


-- ── 6. lmp_forecast ───────────────────────────────────────────────────────────
-- XGBoost DA LMP forecasts | hourly | zone-level
-- model_version tracks which trained artifact produced the forecast
CREATE TABLE IF NOT EXISTS lmp_forecast (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    zone            TEXT        NOT NULL,
    lmp_forecast    DOUBLE PRECISION,
    lmp_actual      DOUBLE PRECISION,   -- NULL until settlement
    forecast_error  DOUBLE PRECISION,   -- NULL until settlement
    model_version   TEXT,               -- e.g. git SHA of Lambda deployment
    CONSTRAINT uq_lmp_forecast UNIQUE (timestamp, zone)
);

CREATE INDEX IF NOT EXISTS idx_lmp_forecast_ts   ON lmp_forecast (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_lmp_forecast_zone ON lmp_forecast (zone, timestamp DESC);


-- ── 7. load_forecast ──────────────────────────────────────────────────────────
-- XGBoost load forecasts | hourly | zone-level
CREATE TABLE IF NOT EXISTS load_forecast (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL,
    zone                TEXT        NOT NULL,
    load_forecast_mw    DOUBLE PRECISION,
    load_actual_mw      DOUBLE PRECISION,   -- NULL until actual data arrives
    forecast_error_mw   DOUBLE PRECISION,   -- NULL until actual data arrives
    model_version       TEXT,
    CONSTRAINT uq_load_forecast UNIQUE (timestamp, zone)
);

CREATE INDEX IF NOT EXISTS idx_load_forecast_ts   ON load_forecast (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_load_forecast_zone ON load_forecast (zone, timestamp DESC);


-- ── 8. bess_dispatch ──────────────────────────────────────────────────────────
-- BESS daily dispatch recommendations from LP optimizer
-- One row per hour per dispatch run (identified by run_date)
CREATE TABLE IF NOT EXISTS bess_dispatch (
    id              BIGSERIAL PRIMARY KEY,
    run_date        DATE        NOT NULL,   -- date the dispatch was computed
    timestamp       TIMESTAMPTZ NOT NULL,   -- dispatch interval (hourly)
    zone            TEXT        NOT NULL,
    charge_mw       DOUBLE PRECISION,       -- positive = charging
    discharge_mw    DOUBLE PRECISION,       -- positive = discharging
    soc_mwh         DOUBLE PRECISION,       -- state of charge at end of interval
    lmp_used        DOUBLE PRECISION,       -- DA LMP used for optimization
    revenue_usd     DOUBLE PRECISION,       -- revenue for this interval
    CONSTRAINT uq_bess_dispatch UNIQUE (run_date, timestamp, zone)
);

CREATE INDEX IF NOT EXISTS idx_bess_dispatch_ts      ON bess_dispatch (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bess_dispatch_run     ON bess_dispatch (run_date DESC);
CREATE INDEX IF NOT EXISTS idx_bess_dispatch_zone    ON bess_dispatch (zone, timestamp DESC);
