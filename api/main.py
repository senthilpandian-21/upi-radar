"""UPI Radar API — FastAPI entry point.

Run:
    uvicorn api.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
for _ in range(4):
    if (ROOT / "config.py").exists():
        break
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from api.routes import alerts, banks, health, predict, reports, route
from config import APP_NAME, APP_VERSION

__version__ = APP_VERSION


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Boot: initialise DB (with demo fallback) + seed demo rows once."""
    try:
        from db.models import init_db

        init_db()
    except Exception as exc:
        logger.warning(f"DB init failed at startup: {exc}")
    try:
        from db.setup import seed_demo_data

        seed_demo_data()
    except Exception as exc:
        logger.debug(f"Demo seed skipped: {exc}")
    logger.info(f"{APP_NAME} API v{__version__} started")
    yield


app = FastAPI(
    title="UPI Radar API",
    description=("AI-powered UPI outage prediction and smart routing — "
                 "predict payment outages BEFORE they happen, route "
                 "transactions intelligently inside Razorpay's ecosystem."),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(route.router, prefix="/route", tags=["Routing"])
app.include_router(banks.router, prefix="/banks", tags=["Banks"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(reports.router, prefix="/reports", tags=["Reports"])


@app.get("/", tags=["Meta"])
async def root() -> dict:
    return {"message": "UPI Radar API Running",
            "status": "ok",
            "version": __version__,
            "docs": "/docs"}


@app.get("/healthz", tags=["Meta"])
async def healthz() -> dict:
    return {"status": "ok"}


def run() -> None:
    """python -m api.main → uvicorn server (for console_scripts)."""
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
