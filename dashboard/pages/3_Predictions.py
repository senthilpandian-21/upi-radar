"""Page 3 — Prediction history over time."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard.api_client import predict_outage, prediction_history  # noqa: E402
from dashboard.components.charts import history_line  # noqa: E402
from dashboard.components.metrics import risk_emoji  # noqa: E402

st.header("📈 Outage Probability Over Time")
st.caption("Ensemble predictions as they happened — red zone triggers "
           "SRE alerting + smart routing")

limit = st.select_slider("History window", options=[24, 48, 100, 200],
                         value=100)
records = prediction_history(limit=limit)
if records:
    st.plotly_chart(history_line(records), width="stretch")

    probs = [float(r.get("outage_probability", 0)) for r in records]
    reds = sum(1 for p in probs if p >= 0.70)
    yellows = sum(1 for p in probs if 0.30 <= p < 0.70)
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples shown", len(probs))
    c2.metric("🔴 Red episodes (≥70%)", reds)
    c3.metric("🟡 Yellow episodes", yellows)

    current = predict_outage()
    st.info(f"{risk_emoji(float(current['final_probability']))} Current risk: "
            f"**{float(current['final_probability']):.1%}** "
            f"({current.get('risk_level')})")
else:
    st.info("No prediction history yet — the monitor agent appends a "
            "prediction every 5 minutes.")
