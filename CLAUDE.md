# CLAUDE.md — NYISO Energy Market Analysis

# Working style: Always explain code after writing it. Walk through the logic, 
# design decisions, and tradeoffs. I need to understand everything in this 
# codebase well enough to explain it in a technical interview.

## Project Overview

NYISO power markets analysis portfolio demonstrating data engineering, ML forecasting, optimization, and visualization. Built to showcase energy data science skills for power markets analyst roles (e.g., Modo Energy).

**GitHub:** https://github.com/braintickle/NYISO-Analysis
**Live dashboard:** https://nyiso.rahishah.com
**Environment:** `conda activate nyiso` | Python 3.11

## Repository Structure

```
nyiso-analysis/
├── src/
│   ├── nyiso_client.py        # NYISO public CSV API fetcher, caching, retry
│   ├── clean.py               # Cleaning, typing, outlier flagging
│   ├── migrate_to_postgres.py # One-time migration: parquets → Supabase Postgres
│   └── generate_synthetic.py  # Synthetic data for Streamlit Cloud deploy
├── lambda/                    # AWS Lambda functions (Project 04)
│   ├── handler.py             # Entry points: ingest_load_fuel, ingest_lmp,
│   │                          #   run_bess_dispatch, update_dns
│   ├── ingest.py              # NYISO fetch + clean + Postgres insert
│   ├── features.py            # Feature engineering (lags, rolling, weather, calendar)
│   ├── inference.py           # XGBoost model loading + prediction + forecast writer
│   ├── bess_dispatch.py       # PuLP LP optimizer + Postgres writer
│   ├── Dockerfile             # Lambda container image (alternative to zip deploy)
│   └── requirements.txt       # Lambda-specific dependencies
├── scripts/
│   ├── export_models.py       # Train XGBoost models + save .joblib artifacts to models/
│   ├── backfill_2026.py       # One-time backfill: Jan–May 2026 for all 4 datasets
│   ├── deploy_lambda_zip.sh   # Lambda zip deploy (layer via S3, function via zip)
│   ├── deploy_lambda.sh       # Lambda container deploy (alternative, needs Docker)
│   └── setup_ecs.sh           # One-time ECS infrastructure setup
├── models/                    # Trained model artifacts (gitignored); run export_models.py
│   ├── load_model_nyc.joblib
│   ├── load_model_longil.joblib
│   ├── lmp_model_nyc.joblib
│   └── lmp_r1_production.ubj  # Rolling R1 production model (XGBoost binary format)
├── sql/
│   └── schema.sql             # Supabase Postgres DDL (8 tables)
├── notebooks/
│   ├── 01_eda.ipynb           # EDA: load, LMP, fuel mix, DA/RT spread
│   ├── 02_forecasting.ipynb   # Load + LMP forecasting (SARIMA, Prophet, XGBoost)
│   └── 03_bess_optimization.ipynb  # BESS dispatch LP optimizer
├── app/
│   └── app.py                 # Streamlit dashboard (Postgres-first, parquet fallback)
├── iam/
│   └── nyiso-deploy-policy.json  # IAM policy for nyiso-deploy user
├── data/
│   ├── raw/                   # Cached monthly parquets (gitignored)
│   └── processed/             # Clean analysis-ready parquets (gitignored)
├── .github/
│   └── workflows/
│       └── deploy.yml         # CI/CD: push to main → build linux/amd64 image → ECS deploy
├── Dockerfile                 # Streamlit app for ECS Fargate (root level)
├── requirements.txt
└── README.md
```

## Data Pipeline

Four datasets from NYISO public CSV API (no key required):

| Dataset      | NYISO endpoint | Resolution | Postgres table  |
|--------------|---------------|------------|-----------------|
| load_actual  | pal           | 5-min      | load_actual     |
| lmp_dayahead | damlbmp       | hourly     | lmp_dayahead    |
| lmp_realtime | rtlbmp        | hourly     | lmp_realtime    |
| fuel_mix     | rtfuelmix     | 5-min      | fuel_mix        |

**Date range:** 2024-01-01 through present (Lambda ingests continuously)
**Zones:** All 11 NYISO zones. Primary analysis on N.Y.C. (Zone J) and LONGIL (Zone K).
**Query engine:** DuckDB for local/notebook parquet access; psycopg2 for cloud Postgres.

### NYISO raw CSV column names (important for cleaning)
- `pal` (load): columns are `Time Stamp, Time Zone, Name, PTID, Load` — "Load" has no "MW" suffix
- `damlbmp`/`rtlbmp` (LMP): columns are `Time Stamp, Name, PTID, LBMP ($/MWHr), Marginal Cost Losses ($/MWHr), Marginal Cost Congestion ($/MWHr)`
- `rtfuelmix` (fuel): columns are `Time Stamp, Time Zone, Fuel Category, Gen MW`
- The `_clean_load` and `_clean_lmp` detectors in `lambda/ingest.py` were fixed to match these column names exactly.

## Key Packages

pandas, numpy, duckdb, xgboost, pulp, prophet, statsmodels, plotly, streamlit, mlflow, shap, scikit-learn, holidays, openmeteo-requests, psycopg2, boto3

## Project Results Summary

### 01 — EDA
- DA vs RT LMP spread: mean $0.81/MWh, std $50.93/MWh (NYC)
- Hour 9: most negative spread (RT < DA) — morning demand overestimated
- Hour 17: most positive spread (RT > DA) — evening peak underestimated
- Natural gas sets marginal price majority of hours
- NYC LMP highest due to transmission-constrained load pocket

### 02 — Forecasting
**Load (NYC + LONGIL):**
- XGBoost MAPE: 2.31% (industry benchmark 2–4%)
- SARIMA: 39.4%, Prophet: 79.9% (baselines)
- Train: 2024, Test: 2025 (temporal split, no leakage)
- Most important feature: `lag_168h` (same hour last week)

**LMP (NYC DA) — static model:**
- XGBoost MAPE: 12.89% (static, trained on 2024 only)
- Mean error: +5.19 $/MWh (systematic underprediction)
- Max error: +229 $/MWh (extreme spikes missed)

**LMP rolling day-ahead simulation (Section 15):**
- R1 (no gas, retrain every 30 days): MAPE=11.60% (-1.76pp vs static)
- R2 (+gas_price_lag1d): MAPE=12.30% — gas still hurts (+0.70pp vs R1)
- Gas degradation shrinks from +4.71pp (static) → +0.70pp (rolling): covariate shift
  partially mitigated, but Jan–Mar still affected (trailing window anchored in 2024 regime)
- Production model: R1 rolling, no gas features

**Gas ablation (Section 14):**
- S2 +price_level: 18.07% — covariate shift (2024 mean $2.25 → 2025 mean $3.54/MMBtu)
- S3 +pct_change: 13.34% — stationary but negligible signal (existing lags cover it)
- Conclusion: gas features do not improve hourly LMP MAPE given existing price history lags

**Feature set (19 LMP features):**
- Calendar: hour, dow, month, quarter, week_of_year, is_weekend, is_holiday
- Weather: temp_f, feels_like_f, humidity_pct, wind_speed_kmh
- Degree days: HDD, CDD (balance point 65°F, IPMVP standard)
- LMP lags: lmp_lag_24h, lmp_lag_48h, lmp_lag_168h
- LMP rolling: lmp_roll_mean_24h, lmp_roll_std_24h, lmp_spike_24h

**MLflow:** experiment tracking at `../mlruns`

### 03 — BESS Optimization
**Battery specs:** 100 MW / 400 MWh (4h duration), 85% RTE, 5–95% SOC, charge derating above 80%

| Strategy | Annual Revenue |
|----------|---------------|
| Naive dispatch | $5,572,918 |
| LP perfect foresight (energy only) | $9,062,076 (+62.6%) |
| LP perfect foresight + ICAP | $14,614,618 |
| LP forecasted prices + ICAP | $14,064,097 |

- Forecast error penalty: $550,522 (7.6% of perfect)
- Basis P&L (RT vs DA settlement): $681,199
- Optimizer: PuLP LP with CBC solver, Zone J, full year 2025
- ICAP: real 2025 NYC Zone J strip auction prices, dynamic commitment

## Project 04 — Live Dashboard (COMPLETE)

### Architecture

```
NYISO Public CSV API
      ↓ (EventBridge cron: 5min for load/fuel, 1hr for LMP)
AWS Lambda  [nyiso-ingestor]
  → fetch → clean → compute features → XGBoost inference
      ↓
Supabase Postgres (free tier)
  tables: load_actual, lmp_dayahead, lmp_realtime, fuel_mix,
          load_forecast, lmp_forecast, bess_dispatch
      ↓
Streamlit on AWS ECS Fargate  [nyiso-dashboard cluster]
  → queries Postgres → renders dashboard
      ↓
Cloudflare (proxy + HTTPS)
  → nyiso.rahishah.com
      ↑
EventBridge (ECS Task State Change → Lambda → Cloudflare API)
  → auto-updates DNS A record on every task restart
```

### AWS Infrastructure (all in us-east-1)

| Resource | Name / ID |
|----------|-----------|
| ECS Cluster | `nyiso-dashboard` |
| ECS Service | `nyiso-dashboard` (desired=1, Fargate) |
| ECR Repository | `nyiso-streamlit` |
| Lambda Function | `nyiso-ingestor` (Python 3.11, 1024MB, 300s timeout) |
| Lambda Layer | `nyiso-deps:1` (pandas, xgboost, psycopg2, pulp, holidays — 234MB unzipped) |
| Lambda Layer source | `s3://nyiso-artifacts-358592704128/nyiso-deps-layer.zip` |
| IAM roles | `nyiso-ecs-exec-role` (ECS task), `nyiso-lambda-role` (Lambda exec) |
| Security Group | `sg-0a59ca39eb5727332` — TCP 8501 inbound 0.0.0.0/0 |
| VPC | `vpc-06b70275d8de36415` |
| Subnets | `subnet-060f12554b8f1ea8c` (us-east-1a), `subnet-0397783105cb80f38` (us-east-1b) |
| CloudWatch Logs | `/ecs/nyiso-dashboard` |
| S3 Bucket | `nyiso-artifacts-358592704128` (Lambda layer storage) |

### EventBridge Rules

| Rule | Schedule / Pattern | Lambda input |
|------|-------------------|--------------|
| `nyiso-ingest-load-fuel` | rate(5 minutes) | `{"task": "ingest_load_fuel"}` |
| `nyiso-ingest-lmp` | rate(1 hour) | `{"task": "ingest_lmp"}` |
| `nyiso-bess-dispatch` | cron(0 13 * * ? *) — 1pm UTC / 9am ET | `{"task": "bess_dispatch"}` |
| `nyiso-ecs-task-running` | ECS Task State Change, lastStatus=RUNNING | full ECS event (no Input override) |

### DNS / HTTPS Setup

- Domain: `rahishah.com` on Namecheap; DNS managed by Cloudflare (free plan)
- `nyiso.rahishah.com` → Cloudflare A record, proxy enabled (orange cloud)
- Cloudflare proxy provides HTTPS with shared certificate — no ACM needed
- `nyiso-ecs-task-running` EventBridge rule fires on every Fargate task restart →
  Lambda reads ENI from event → EC2 resolves public IP → Cloudflare API updates A record
- DNS self-heals within ~30 seconds of any task restart at $0/month
- Cloudflare env vars in Lambda: `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_RECORD_NAME=nyiso.rahishah.com`

### Lambda Handler Tasks

`lambda/handler.py` routes on `event['task']` for cron events, and on `event['detail-type'] == 'ECS Task State Change'` for DNS updates:

| Task key | Function | Trigger |
|----------|----------|---------|
| `ingest_load_fuel` | Fetch load_actual + fuel_mix, insert to Postgres | Every 5 min |
| `ingest_lmp` | Fetch DA+RT LMP, run XGBoost forecasts, write forecasts | Every hour |
| `bess_dispatch` | Fetch today's DA LMP, LP optimize, write bess_dispatch | Daily 9am ET |
| `compute_revenue_summary` | Compute naive + PF revenue from bess_dispatch, write summary | Manual / daily |
| `backfill_forecasts` | Fetch historical actuals + weather, run forecasts for date range | Manual invoke |
| *(ECS event)* | `update_dns` — patch Cloudflare A record with new task IP | ECS task RUNNING |

### Lambda Layer Notes

- scipy and sklearn were stripped from the layer (not used at Lambda inference time)
- sklearn is only needed locally for `scripts/export_models.py` (training)
- Layer exceeds 50MB zip limit — uploaded to S3, then published via `--content S3Bucket=...`
- Layer permissions: `lambda:GetLayerVersion` must be on versioned ARN `layer:nyiso-*:*`, not unversioned `layer:nyiso-*`

### CI/CD Pipeline (`.github/workflows/deploy.yml`)

Triggers on push to `main` when `app/`, `src/`, `requirements.txt`, `Dockerfile`, or the workflow file changes:
1. Configure AWS credentials (`nyiso-deploy` IAM user)
2. Login to ECR
3. `docker build --platform linux/amd64` (Linux build on GitHub runner — no local Docker needed)
4. Push image tagged with git SHA + `latest`
5. Render new ECS task definition with updated image + `DATABASE_URL` secret
6. Deploy to ECS, wait for stability
7. Print public IP to job summary

GitHub Secrets required: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DATABASE_URL`

### Known Issues Fixed During Deployment

1. **Lambda column detection bugs** (`lambda/ingest.py`):
   - `_clean_load` zone_col: NYISO `pal` CSV columns are `Time Stamp, Time Zone, Name, PTID, Load`. "Time Zone" (EDT/EST) matches `"zone" in c.lower()` before "Name" (zone names). Fix: exact match `c.strip().lower() == "name"` first. Old bug stored zone='EDT'/'EST' and ON CONFLICT discarded all but first zone per timestamp.
   - `_clean_load` load_col: NYISO `pal` CSV has column `"Load"` (not `"Load MW"`) — fixed detector to match bare `"Load"`
   - `_clean_lmp`: NYISO LMP CSV has column `"LBMP ($/MWHr)"` — fixed detector from exact match `"lbmp"` to `"lbmp" in column.lower()`
   - `_bulk_insert`: numpy int64 scalars not serializable by psycopg2 — added `_to_native()` converter

2. **Lambda MODEL_DIR** (`lambda/inference.py`):
   - `Path(__file__).parent.parent / "models"` resolved to `/var/models/` in Lambda (not `/var/task/models/`)
   - Fixed to `Path(__file__).parent / "models"` = `/var/task/models/`

3. **LMP model** (`lambda/inference.py`):
   - `lmp_model_nyc.joblib` (sklearn XGBRegressor) requires scipy at inference time
   - Switched to `lmp_r1_production.ubj` (native XGBoost booster, MAPE 11.60% vs 12.89%)
   - Uses `xgb.Booster()` + `xgb.DMatrix()` — no sklearn/scipy needed

4. **scipy mock in Lambda** (`lambda/handler.py`):
   - XGBoost 1.x imports `scipy.sparse` at module level in `xgboost/core.py`
   - `xgboost/sklearn.py` imports `from scipy.special import softmax, expit` at module level
   - Mock must use `types.ModuleType` with `__path__ = []` (not SimpleNamespace — Python won't resolve sub-imports from non-package objects)
   - Mock must provide `softmax`, `expit`, and stubs for `scipy.linalg`, `scipy.stats`

5. **Empty Series RangeIndex** (`lambda/features.py`):
   - `_fetch_load_history` / `_fetch_lmp_history` returned `pd.Series(dtype=float)` (RangeIndex) for 0 rows
   - `hist.index < ts` (Timestamp comparison) fails on RangeIndex
   - Fixed: return `pd.Series(dtype=float, index=pd.DatetimeIndex([]))`

6. **Naive BESS revenue bug** (`lambda/bess_dispatch.py`):
   - Discharge limit used `SOC - SOC_MIN_MWH` (40 MWh floor), allowing 160 MWh of initial charge to be consumed
   - LP has terminal constraint `s[T-1] >= SOC_INIT_MWH = 200`, so naive was earning more than perfect foresight
   - Fixed to `SOC - SOC_INIT_MWH` (200 MWh floor)

7. **Postgres query window** (`app/app.py`):
   - Original 90-day cutoff excluded all 2024–2025 historical data (>18 months old)
   - Extended to 548 days to cover full dataset; tighten back to 90 days once Lambda has built up 3 months of live data

8. **Streamlit multiselect crash** (`app/app.py`):
   - Default zones `["N.Y.C.", "LONGIL", "CAPITL"]` crashed when `df_load` was empty
   - Fixed: filter defaults to zones that exist in loaded data, fall back to first 3 available

9. **Lambda layer size**: 430MB → 234MB by removing scipy + sklearn (not needed at inference time)

10. **ECS CloudWatch log group**: `setup_ecs.sh` silently failed to create `/ecs/nyiso-dashboard` due to Git Bash expanding `/ecs/` as a Windows path — created via boto3 instead

11. **IAM permissions discovered incrementally**:
    - `lambda:GetLayerVersion` must target versioned ARN `layer:nyiso-*:*`
    - `lambda:InvokeFunction` needed for smoke testing
    - `ec2:DescribeNetworkInterfaces` needed on `nyiso-lambda-role` (execution role, not deploy user)
    - ECS service-linked role `AWSServiceRoleForECS` must exist before `create-cluster` with capacity providers

12. **Windows zip path separator** (`build_lambda.py`):
    - `str(Path("models/file.joblib"))` = `"models\\file.joblib"` on Windows — Lambda (Linux) needs forward slashes
    - Fixed: `f.as_posix()` in build_lambda.py for all model files

### Data Backfill

`scripts/backfill_2026.py` — one-time script that fetched Jan 1 – May 17, 2026 for all 4 datasets and inserted into Supabase. **WARNING: had the zone_col bug** — stored 40,219 load_actual rows with zone='EDT'/'EST' instead of 'N.Y.C.'/'LONGIL'. Fixed by `fix_load_zones.py` (delete + re-ingest).

`fix_load_zones.py` (root, not committed) — deleted 58K corrupted load_actual rows (zone IN ('EDT','EST')), re-fetched 7 months with fixed zone detection.

`backfill_forecasts` Lambda task — invoked manually after fix to write load + LMP forecasts for 2026:
- Jan–Feb 2026: 1416 LMP + 1416 N.Y.C. + 1416 LONGIL load forecast rows
- Mar–Apr 2026: 1464 LMP + 1464 N.Y.C. + 1464 LONGIL load forecast rows
- May 1–17, 2026: 408 LMP + 408 N.Y.C. + 408 LONGIL load forecast rows

## Conventions

- All timestamps are Eastern (NYISO native timezone)
- DuckDB for local/notebook parquet queries; Postgres for cloud/dashboard
- Parquet files are gitignored; `src/nyiso_client.py` regenerates locally
- Notebooks use Plotly for interactive charts (nbviewer for rendering)
- Lambda secrets stored in Lambda env vars (DATABASE_URL, CF_API_TOKEN, CF_ZONE_ID)
- Docker image tagged with git SHA for traceability
- Lambda layer uploaded to S3 first (>50MB), then published via `--content S3Bucket=...`
- `zip` not available on Windows Git Bash — use Python `zipfile` module to build zips

## Deploying Lambda Updates

```bash
# 1. Rebuild function zip using build_lambda.py (handles POSIX paths for models/)
conda activate nyiso
python build_lambda.py

# 2. Push code update
aws lambda update-function-code \
  --function-name nyiso-ingestor \
  --zip-file fileb://nyiso-function.zip \
  --region us-east-1

aws lambda wait function-updated --function-name nyiso-ingestor --region us-east-1

# 3. Update env vars if needed
aws lambda update-function-configuration \
  --function-name nyiso-ingestor \
  --environment "Variables={DATABASE_URL=...,MODEL_VERSION=...,CF_API_TOKEN=...,CF_ZONE_ID=...,CF_RECORD_NAME=nyiso.rahishah.com}" \
  --region us-east-1
```

## Getting Current Dashboard IP

```python
import boto3
ecs = boto3.client("ecs", region_name="us-east-1")
ec2 = boto3.client("ec2", region_name="us-east-1")
tasks = ecs.list_tasks(cluster="nyiso-dashboard", serviceName="nyiso-dashboard")["taskArns"]
detail = ecs.describe_tasks(cluster="nyiso-dashboard", tasks=tasks)["tasks"][0]
eni_id = next(d["value"] for att in detail["attachments"] for d in att["details"] if d["name"] == "networkInterfaceId")
ip = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])["NetworkInterfaces"][0]["Association"]["PublicIp"]
print(f"http://{ip}:8501")
```

## TODO

### Project 04 — Live Dashboard
- [x] Design Supabase Postgres schema (`sql/schema.sql`)
- [x] Write migration script: parquets → Postgres (`src/migrate_to_postgres.py`)
- [x] Lambda backend: handler, ingest, features, inference, bess_dispatch
- [x] Package XGBoost models as Lambda artifact (`scripts/export_models.py`)
- [x] Dockerize Streamlit app (`Dockerfile`)
- [x] Build dashboard (load vs forecast, LMP heatmap, spread tracker, BESS dispatch) in `app/app.py`
- [x] Deploy Lambda + EventBridge cron rules (`scripts/deploy_lambda_zip.sh`)
- [x] Deploy ECS Fargate cluster + service + security group (`scripts/setup_ecs.sh`)
- [x] CI/CD via GitHub Actions (`.github/workflows/deploy.yml`)
- [x] Backfill 2026 data (`scripts/backfill_2026.py`)
- [x] Stable HTTPS URL at `nyiso.rahishah.com` via Cloudflare + Lambda DNS auto-update
- [ ] Fix `use_container_width` Streamlit deprecation warnings in `app/app.py` (removed after 2025-12-31)
- [ ] Tighten Postgres query window back to 90 days once Lambda has 3 months of live data
- [ ] Retrain models on 2025+2026 data and redeploy Lambda

### Forecasting Improvements
- [ ] Add solar generation feature for duck curve
- [ ] Retrain LMP model with 2026 data (covariate shift from 2024 training)

### Other
- [ ] Fix Plotly rendering in GitHub notebook viewer
- [ ] Project 05: Electrification structural break detection (heat pumps, EVs)
