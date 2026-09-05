<div align="center">

# ⚡ UPI RADAR

### AI-Powered Payment Outage Predictor & Smart Router

**Predict UPI outages 2 hours before they happen · Route transactions intelligently within Razorpay's ecosystem · Convert failures into queues, never silent drops**

[![Tests](https://img.shields.io/badge/tests-72_passed-brightgreen)](#-testing)
[![Python](https://img.shields.io/badge/python-3.10_%7C_3.11-blue)](#-prerequisites)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)](#-api-reference)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49+-FF4B4B?logo=streamlit&logoColor=white)](#-dashboard)
[![Docker](https://img.shields.io/badge/Docker-compose_ready-2496ED?logo=docker&logoColor=white)](#option-c--docker-full-stack)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

*Razorpay AI Buildathon 2026 · Track 5 (Open Track) · AI Agent System · Fintech / Payment Infrastructure*

</div>

---

## 🎯 The Problem

| Statistic | Value |
|---|---|
| UPI transactions | **23.2 billion/month** (~591M/day, May 2026) |
| Major UPI outages in 3 weeks | **3** (26 Mar – 12 Apr 2026) |
| Transactions lost on 26 Mar outage | **41 million** (550M vs 591M avg) |
| Customers who never return after a failed payment | **62%** |
| When outages strike | Predictably: **FY-end, salary days, festivals, 6–9 PM peak** |

Today's systems are **reactive** — they detect failure *after* it happens, then scramble.
UPI Radar is **proactive**: it forecasts, prevents, and routes around failures before
merchants feel them — while keeping every transaction **inside Razorpay**
(no competitor routing, no revenue leakage).

---

## ✨ Highlights

- 🔮 **2-hour-ahead outage forecasting** — weighted ensemble: LSTM (50%) + Prophet (30%) + Isolation Forest (20%)
- 🏦 **Bank Health Scoring (0–100)** — deterministic, Redis-cached, weighted 15-min / 1-hr / historical technical-decline
- 🔀 **Smart routing ladder** — bank→bank failover, UPI→card/netbanking/wallet/NEFT-IMPS rail switching, cascade retries
- ⏳ **Graceful degradation** — transactions are queued with exponential backoff and replayed, never silently dropped
- 📜 **Fully auditable** — every decision persisted to PostgreSQL *and* JSON-lines audit logs; compliance rules are hard-coded, never ML
- 📄 **Auto-generated post-outage PDF reports**
- 🤖 **Six cooperating agents** — Monitor · Router · Alert · Queue · Report · Retrainer (self-heals when rolling accuracy < 80%)
- 🧪 **Runs anywhere** — demo mode needs zero infrastructure: no Kafka/Postgres/Redis/keys/models required (all have built-in fallbacks)

---

## 🏗️ System Architecture (5 layers)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 · DATA COLLECTION                                                 │
│  Razorpay Downtime API │ NPCI CSVs │ Downdetector │ Synthetic Generator   │
│            └──────────────┬───────────────────────────────┘               │
│                    Apache Kafka ──▶ PostgreSQL                           │
├──────────────────────────────────────────────────────────────────────────┤
│ LAYER 2 · AI MODEL LAYER                                                  │
│  LSTM (24h sequences) │ Prophet (seasonality) │ Isolation Forest (stream)│
│            └─────────────── Ensemble: 0.5·LSTM + 0.3·Prophet + 0.2·IF ──┘│
│  BankHealthScorer (0-100, Redis-cached, weighted 15m/1h/historical)       │
├──────────────────────────────────────────────────────────────────────────┤
│ LAYER 3 · DETERMINISTIC POLICY ENGINE (no ML — fully auditable)           │
│  Bank Rules (B1-B5) │ Gateway Rules (G1-G3) │ Queue Rules │ SRE Alerts    │
├──────────────────────────────────────────────────────────────────────────┤
│ LAYER 4 · AGENT ACTION LAYER                                              │
│  MonitorAgent │ RouterAgent (cascade) │ AlertAgent │ QueueAgent │         │
│  ReportAgent │ RetrainerAgent                                             │
├──────────────────────────────────────────────────────────────────────────┤
│ LAYER 5 · DASHBOARD + API                                                 │
│  FastAPI REST · Streamlit + Plotly dashboard · Redis cache                │
└──────────────────────────────────────────────────────────────────────────┘
```

**The intelligent routing ladder (all within Razorpay):**

```
Bank A critical → Bank B/C (healthiest)      (Bank Rules B1–B5)
UPI at risk    → Card / Netbanking / Wallet  (Gateway Rules G1–G3)
Everything busy → Queue with exponential backoff → retry → alternate rail
Still failing  → SRE page + merchant alert + post-outage PDF report
```

### 🧠 Model choices (why these models)

| Signal | Model | Why |
|---|---|---|
| Past 24 h per-bank telemetry | **LSTM** (3 layers, 128→64→32) | Sequence → risk; catches ramp-ups before spikes |
| Calendar seasonality | **Prophet** (salary/FY-end/festival regressors) | Outages are *scheduled by the calendar* — learn the pattern |
| Live stream deviation | **Isolation Forest** | 5-feature anomaly score every prediction tick |
| Final probability | **Weighted ensemble** | 50/30/20 blend → GREEN/YELLOW/RED risk levels |
| Bank health 0–100 | **Deterministic scorer** | 0.5×15min + 0.3×1hr + 0.2×historical TD% — no black box |
| Compliance decisions | **Hard-coded policy rules** | AI never makes audit/compliance decisions (B1–B5, G1–G3) |

---

## ✅ Prerequisites

| Requirement | Needed for | Notes |
|---|---|---|
| **Python 3.10 or 3.11** | Everything | ⚠️ **3.12/3.13 cannot install the full ML stack** (`tensorflow==2.15.0`, `prophet==1.1.5` have no wheels). Demo mode works on newer Pythons with `requirements-runtime.txt`. |
| Git | Clone | — |
| ~1 GB disk (demo) / ~4 GB (full ML) | pip installs | TensorFlow CPU is enough — **no GPU required** |
| Docker Desktop | Option C only | Full Kafka + PostgreSQL + Redis stack |
| Razorpay **test** keys | "Live keys" mode only | Free: [dashboard.razorpay.com](https://dashboard.razorpay.com) → Settings → API Keys |
| MSVC Build Tools (Windows) | Prophet training on Windows | Or use WSL2, where `pip install prophet` just works |

---

## 🚀 Quick Start

### Option A — Demo mode (no keys, no infra, no training — ~2 minutes)

Works out of the box: every external dependency (Kafka, PostgreSQL, Redis,
Razorpay, trained models) has a built-in fallback.

```bash
git clone https://github.com/senthilpandian-21/upi-radar.git
cd upi-radar

# 1. Virtual environment (Python 3.10/3.11)
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. Dependencies (lightweight runtime set)
pip install -r requirements-runtime.txt

# 3. Database + data + quick model
python db/setup.py                              # SQLite fallback auto-used
python data/synthetic/synthetic_generator.py    # 70,080 hourly rows, 1 year
python -m models.anomaly_detector.train         # IsolationForest (~3 s)

# 4. Run the three services (one per terminal, venv activated in each)
uvicorn api.main:app --reload --port 8000       # Terminal 1 — REST API
python -m agent.monitor_agent                   # Terminal 2 — live agent + audit feed
streamlit run dashboard/app.py --server.port 8501  # Terminal 3 — dashboard
```

| URL | What |
|---|---|
| **http://localhost:8501** | Streamlit dashboard (works even with everything else off) |
| **http://localhost:8000/docs** | Interactive API docs (Swagger UI) |
| **http://localhost:8000/health** | Component health (PG/Redis/Kafka/models) |

The monitor agent runs a **self-demonstrating boot scenario** (critical bank →
reroute → cascade → queue), so the dashboard, audit logs and history endpoints
show live activity within seconds.

### Option B — Full pipeline (train all 3 models)

Requires **Python 3.10/3.11** and the full dependency set:

```bash
pip install -r requirements.txt                 # incl. tensorflow + prophet (~3 GB)

python data/synthetic/synthetic_generator.py
python -m models.lstm_forecaster.train          # → saved_model/lstm_outage_model.h5
python -m models.prophet_forecaster.train       # → saved_model/prophet_model.pkl
python -m models.anomaly_detector.train         # → saved_model/isolation_forest.pkl
```

The ensemble auto-loads trained artifacts and switches off demo fallbacks —
verify with `GET /predict/outage` → `"models_loaded": {"lstm": true, …}`.
LSTM training takes ~10–40 min on a laptop CPU (no GPU needed).

### Option C — Docker (full stack: Kafka + PostgreSQL + Redis)

```bash
cp .env.example .env                            # edit values as needed
docker compose up -d zookeeper kafka postgres redis   # infra only (optional)
docker compose up --build                             # everything
```

Services start in dependency order with healthchecks — the API/agent wait for
PostgreSQL/Redis/Kafka to be **ready**, so the full stack never silently
falls back to SQLite.

> ℹ️ The Docker image installs `requirements-runtime.txt` (fast, CPU-friendly).
> Uncomment the full-requirements lines in the `Dockerfile` to train in-container.

### Live Razorpay test keys (optional)

Set `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` (`rzp_test_…`) in `.env` and
`DEMO_MODE=false`. The pipeline then polls `GET /v1/payments/downtimes` every
5 minutes and payment attempts execute against the Razorpay test API.

---

## 🌐 API Reference

Base URL: `http://localhost:8000` · Interactive docs: `/docs`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/predict/outage` | Ensemble outage probability (2 h horizon) + recommendation |
| GET | `/predict/history?limit=` | Recent model predictions (DB, falls back to audit log) |
| POST | `/route/transaction` | Smart-route one transaction (see below) |
| GET | `/route/history?limit=` | Audited routing decisions |
| GET | `/banks/health` | All bank health scores (0–100) + best/critical/warning lists |
| GET | `/banks/health/{bank}` | Single bank health detail (404 for unknown banks) |
| GET | `/alerts/active` · `/alerts/history` | SRE/merchant alerts |
| GET | `/reports/list` · `/reports/download/{file}` | Generated PDF reports |
| POST | `/reports/generate?hours_ago=2` | Build a post-outage PDF from the audit trail |
| GET | `/health` · `/healthz` | Component-level and liveness health |

### `POST /route/transaction`

```jsonc
// Request
{
  "transaction": {
    "transaction_id": "txn_demo_001",     // optional
    "amount": 500000,                      // paise (₹5,000.00)
    "currency": "INR",
    "method": "upi",                       // upi | card | netbanking | wallet | neft_imps
    "bank": "BOB"                          // remitter bank / PSP handle
  },
  "simulate": false                        // true → decide & report, never execute/queue
}
```

```jsonc
// Response (200)
{
  "transaction_id": "txn_demo_001",
  "original_bank": "BOB",
  "original_method": "upi",
  "action": "ROUTE",                       // PROCEED | ROUTE | QUEUE
  "routed_to_bank": "KOTAK",
  "routed_to_method": "upi",
  "reason": "RULE B4: BOB health score 55.5% < 70% → route via KOTAK",
  "rule_id": "B4",
  "outage_probability": 0.1815,
  "success": true,
  "queued": false,
  "risk_level": "GREEN",
  "severity": "MEDIUM",
  "timestamp": "2026-09-05T07:44:27.126075"
}
```

---

## 📊 Dashboard

Five Streamlit pages, live against the API (with an offline simulation mode if
the API is down):

| Page | Shows |
|---|---|
| 📊 **Overview** | Live outage gauge + ensemble model breakdown |
| 🏦 **Bank Health** | Per-bank scores, best fallback banks, critical/warning lists |
| 📈 **Predictions** | Outage probability history + risk episodes |
| 🔄 **Routing Log** | Every smart-routing decision, fully audited |
| 📄 **Reports** | Auto-generated post-outage PDFs (download in one click) |

---

## 🧪 Testing

```bash
pytest tests/ -v        # 72 tests · no infra required · ~50 s
```

Coverage: synthetic generator (row counts, columns, outage injection, known
outage dates) · health scorer formula & statuses · sequence shapes · ensemble
risk bands & determinism · **every policy rule** (B1–B5, G1–G3, S1–S3, queue
backoff/abandon) · router cascade + queue fallback · alert/report/retrainer
agents · all API endpoints incl. the `/route/transaction` request contract and
`simulate` semantics.

---

## 📁 Project Structure

```
upi-radar/
├── config.py                  # env, paths, thresholds (single source of truth)
├── data/synthetic/            # UPIOutageSimulator → generated_data.csv (70,080 rows)
├── data_pipeline/             # Razorpay/NPCI/Downdetector collectors, Kafka, features
├── models/                    # lstm/ prophet/ anomaly_detector/ bank_health_scorer/ ensemble.py
├── policy/                    # bank_rules B1-B5 · gateway_rules G1-G3 · queue · sre_alerts
├── agent/                     # monitor · router · alert · queue · report · retrainer
├── api/                       # FastAPI: /predict /route /banks /alerts /reports /health
├── dashboard/                 # Streamlit: overview · bank health · predictions · routing · reports
├── db/                        # SQLAlchemy models + 001_initial_schema.sql + setup.py
├── audit_logs/                # JSON-lines audit: routing_decisions / alerts / predictions
├── reports/                   # auto-generated post-outage PDFs
├── tests/                     # 72 pytest cases (models/policy/routing/api/agents)
└── notebooks/                 # 01 EDA → 05 evaluation
```

---

## ⚙️ Configuration

All settings are environment variables (`.env` supported — copy `.env.example`).
Everything has a safe default; the system boots in demo mode with an empty `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `DEMO_MODE` | `true` | Simulate stream + fallbacks; no keys/infra needed |
| `RAZORPAY_KEY_ID` / `_SECRET` | — | `rzp_test_…` keys enable the live pipeline |
| `DATABASE_URL` | local PG DSN | Any SQLAlchemy URL; unreachable PG → SQLite fallback |
| `REDIS_HOST` / `REDIS_PORT` | `localhost:6379` | Health cache; unreachable → in-process cache |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Stream; unreachable → spool/demo stream |
| `OUTAGE_PROBABILITY_THRESHOLD` | `0.70` | RED risk / SRE-page threshold |
| `BANK_HEALTH_CRITICAL_THRESHOLD` | `30` | Rule B1 (critical) — read at request time |
| `BANK_HEALTH_WARNING_THRESHOLD` | `70` | Rule B4 (degraded) — read at request time |
| `QUEUE_MAX_WAIT_SECONDS` | `120` | Max queue wait before abandon + alternate-method advice |
| `PREDICTION_INTERVAL_SECONDS` | `300` | Agent prediction cadence (10 s in demo mode) |
| `SMTP_*` / `TWILIO_*` | — | Real email/SMS alerts; unset → logged, never sent |

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install -r requirements.txt` fails on `tensorflow==2.15.0` | You're on Python 3.12+. Use **3.10/3.11** for the full stack, or `requirements-runtime.txt` (demo mode) on any version. |
| Prophet install fails on Windows | Install *MSVC C++ Build Tools* (Visual Studio Installer → “Desktop development with C++”), or use WSL2. |
| `IsADirectoryError: … generated_data.csv` | A stale directory from older revisions: `rm -rf data/synthetic/generated_data.csv` (Windows: `rmdir /s`), then re-run the generator. Fixed in current `config.py`. |
| Port already in use | `uvicorn … --port 8001` / `streamlit run … --server.port 8502`, or free the port. |
| Dashboard shows old demo rows | Delete `data/upi_radar_demo.db` (runtime artifact, git-ignored) and restart — `db/setup.py` recreates + re-seeds it. |
| `/health` says `postgres: SQLite (demo fallback)` | Expected without Postgres. For the real stack use Docker Option C or set `DATABASE_URL`. |

---

## 🛡️ Reliability & Audit

| Criterion | Implementation |
|---|---|
| Failure recovery | Cascade routing (3 attempts) → queuing w/ exponential backoff → alternate-method suggestion |
| Auditability | Every routing decision, alert and prediction → DB **and** `audit_logs/*.log` JSON, recording the exact health score & outage probability the rules used |
| Deterministic policy | Rules B1–B5 / G1–G3 are pure thresholds — zero ML in compliance paths |
| No competitor routing | Routes only between Razorpay PSP banks & rails (UPI/card/NB/wallet/NEFT-IMPS) |
| Self-healing | RetrainerAgent auto-retrains when rolling accuracy < 80% |
| Graceful degradation | No Kafka → spool/demo stream · no Redis → in-process cache · no PG → SQLite · no models → demo signal |
| 62%-customer-loss defence | Failures converted to queues + alerts, never silent drops |

---

## 🗺️ Roadmap

- [ ] Real-time NPCI bank-wise telemetry ingestion (contract pending)
- [ ] Per-bank LSTM heads (currently one system-wide sequence model)
- [ ] Webhook delivery of merchant routing-change alerts
- [ ] Load-testing harness (target: 5k decisions/s on the policy path)
- [ ] Grafana/Prometheus exporter for the SRE stack

---

## ⚠️ Disclaimer

UPI Radar is a **buildathon submission / research prototype**. It uses a
realistic synthetic dataset (seeded from public NPCI statistics & reported
outage events) and Razorpay **Test Mode** APIs. Production deployment requires
the Razorpay SRE team's risk review, real telemetry contracts, and
load-testing — this is a demonstration of the *architecture and ML approach*,
not production code.

---

## 📄 License

MIT © 2026 UPI Radar Team — see [LICENSE](LICENSE).

## 👥 Team

| | |
|---|---|
| Team | UPI Radar |
| Track | Razorpay AI Buildathon 2026 — Track 5 (Open Track) |
| Repository | [github.com/senthilpandian-21/upi-radar](https://github.com/senthilpandian-21/upi-radar) |
| Issues & feedback | [Open an issue](https://github.com/senthilpandian-21/upi-radar/issues) |
