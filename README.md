# ⚡ UPI RADAR — AI-Powered Payment Outage Predictor & Smart Router

> **Predict UPI outages 2 hours BEFORE they happen. Route transactions intelligently WITHIN Razorpay's ecosystem. Achieve near-zero payment failures.**

**Razorpay AI Buildathon 2026 · Track 5 (Open Track)** · AI Agent System · Fintech / Payment Infrastructure

---

## 🎯 Problem Statement

| Statistic | Value |
|---|---|
| UPI transactions | **23.2 billion/month** (~591M/day, May 2026) |
| Major UPI outages in 3 weeks | **3** (26 Mar – 12 Apr 2026) |
| Transactions lost on 26 Mar outage | **41 million** (550M vs 591M avg) |
| Customers who never return after a failed payment | **62%** |
| When outages strike | Predictably: **FY-end, salary days, festivals, 6–9 PM peak** |

**Current systems are REACTIVE** — they detect failure *after* it happens, then scramble. UPI Radar is **PROACTIVE**: it forecasts, prevents, and routes around failures before merchants feel them — while keeping every transaction **inside Razorpay** (no competitor routing, no revenue leakage).

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
│            └─────────────── Ensemble: 0.5·LSTM + 0.3·Prophet + 0.2·IF ──┘
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

---

## 🧠 AI Judgment (why these models)

| Signal | Model | Why |
|---|---|---|
| Past 24 h per-bank telemetry | **LSTM** (3 layers, 128→64→32) | Sequence → sequence risk; catches ramp-ups before spikes |
| Calendar seasonality | **Prophet** (salary/FY-end/festival regressors) | Outages are *scheduled by the calendar* — learn the pattern |
| Live stream deviation | **Isolation Forest** | 5-feature anomaly score every prediction tick |
| Final probability | **Weighted ensemble** | 50/30/20 blend + GREEN/YELLOW/RED risk levels |
| Bank health 0–100 | **Deterministic scorer** | 0.5×15min + 0.3×1hr + 0.2×historical TD% — no black box |
| Compliance decisions | **Hard-coded policy rules** | AI never makes audit/compliance decisions (B1–B5, G1–G3) |

---

## 🚀 Quick Start

### Option A — Demo mode (no keys, no infra, no training — 2 minutes)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # or the lighter requirements-runtime.txt

python db/setup.py                     # SQLite fallback auto-used
python data/synthetic/synthetic_generator.py     # 70,080 rows
python -m models.anomaly_detector.train         # quick sklearn model (fast)

# Terminal 1 — API
uvicorn api.main:app --reload --port 8000
# Terminal 2 — Agent (feeds audit logs with live demo traffic)
python -m agent.monitor_agent
# Terminal 3 — Dashboard
streamlit run dashboard/app.py --server.port 8501
```

Open **http://localhost:8501** → the dashboard works even with everything off (offline simulation), and lights up with live data once the API/agent run.

### Option B — Full pipeline (train all 3 models)

```bash
pip install -r requirements.txt        # includes tensorflow + prophet (Python 3.10/3.11)
python data/synthetic/synthetic_generator.py
python models/lstm_forecaster/train.py          # → saved_model/lstm_outage_model.h5
python models/prophet_forecaster/train.py       # → saved_model/prophet_model.pkl
python models/anomaly_detector/train.py         # → saved_model/isolation_forest.pkl
```

The ensemble automatically loads the trained artifacts and switches off demo fallbacks (`GET /predict/outage` shows `models_loaded`).

### Option C — Docker (full stack)

```bash
cp .env.example .env
docker-compose up -d zookeeper kafka postgres redis   # infra only
docker-compose up --build                              # everything
```

> ℹ️ Heavy ML deps (tensorflow/prophet) are optional at runtime — `Dockerfile` installs `requirements-runtime.txt` by default so containers boot fast. Uncomment the full `requirements.txt` line for in-container training.

### Test mode — real Razorpay keys

Set `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` (rzp_test_…) in `.env`, set `DEMO_MODE=false`. The pipeline polls `GET /v1/payments/downtimes` every 5 min and routing executes against the Razorpay test API.

---

## 📁 Project Map

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
├── tests/                     # 50+ pytest cases (models/policy/routing/api/agents)
└── notebooks/                 # 01 EDA → 05 evaluation
```

---

## 🌐 API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/predict/outage` | Ensemble outage probability (2 h horizon) + recommendation |
| GET | `/predict/history?limit=` | Recent model predictions |
| POST | `/route/transaction` | `{transaction:{amount, method, bank, …}}` → routing decision + execution |
| GET | `/route/history?limit=` | Audited routing decisions |
| GET | `/banks/health` · `/banks/health/{bank}` | Bank health scores 0–100 |
| GET | `/alerts/active` · `/alerts/history` | SRE/merchant alerts |
| GET | `/reports/list` · `/reports/download/{file}` | PDF reports |
| POST | `/reports/generate` | Demo post-outage report |
| GET | `/health` | Component health (PG/Redis/Kafka/models) |

Interactive docs: **http://localhost:8000/docs**

---

## 🧪 Tests

```bash
pytest tests/ -v          # no infra required — everything falls back to demo mode
```

Coverage: generator row-count/columns/outage injection · scorer formula/status · sequence shapes ·
ensemble mock keys/risk bands · every policy rule (B1–B5, G1–G3, S1–S3, queue) ·
router cascade + queue fallback · alert/report/retrainer agents · all API endpoints.

---

## 🐳 docker-compose Services

`zookeeper` · `kafka` (confluent 7.5) · `postgres:16` · `redis:7` ·
`api` (uvicorn :8000) · `dashboard` (streamlit :8501) · `agent` (monitor agent)

---

## 🛡️ Reliability & Audit (evaluation-criteria aligned)

| Criterion | Implementation |
|---|---|
| Failure recovery | Cascade routing (3 attempts) → queuing w/ exponential backoff → alternate-method suggestion |
| Auditability | Every routing decision, alert and prediction → DB **and** `audit_logs/*.log` JSON |
| Deterministic policy | Rules B1–B5 / G1–G3 are pure thresholds — zero ML in compliance paths |
| No competitor routing | Routes only between Razorpay PSP banks & rails (UPI/card/NB/wallet/NEFT-IMPS) |
| Self-healing | RetrainerAgent auto-retrains when rolling accuracy < 80% |
| Graceful degradation | No Kafka → spool/demo stream · no Redis → in-process cache · no PG → SQLite · no models → demo signal |
| 62%-customer-loss defence | Failures converted to queues + alerts, never silent drops |

---

## ⚠️ Disclaimer

UPI Radar is a **buildathon submission / research prototype**. It uses a realistic
synthetic dataset (seeded from public NPCI statistics & reported outage events) and
Razorpay **Test Mode** APIs. Production deployment requires the Razorpay SRE team's
risk review, real telemetry contracts, and load-testing — this is a demo of the
*architecture and ML approach*, not production code.

---

## 👥 Team

| | |
|---|---|
| Team | UPI Radar |
| Track | Razorpay AI Buildathon 2026 — Track 5 (Open Track) |
| Contact | your-team@example.com |
