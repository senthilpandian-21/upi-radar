"""Page 1 — Overview: live outage gauge + model breakdown."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard.api_client import predict_outage  # noqa: E402
from dashboard.components.charts import component_bars, risk_gauge  # noqa: E402
from dashboard.components.metrics import (info_banner, risk_colour,  # noqa: E402
                                          risk_emoji)

st.header("📊 Current Outage Probability")
st.caption("Ensemble forecast: LSTM (50%) + Prophet (30%) + Isolation Forest (20%)")

auto = st.toggle("🔄 Auto-refresh every 30 seconds", value=False)
if st.button("🔄 Refresh now", use_container_width=False):
    st.rerun()

prediction = predict_outage()
prob = float(prediction["final_probability"])
risk = prediction.get("risk_level", "GREEN")
source = prediction.get("source", "api")
info_banner(source, demo=prediction.get("demo", False))

# ── metric row ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"<div style='background:#f7f9fc;border-left:6px solid "
        f"{risk_colour(prob)};border-radius:10px;padding:14px;'>"
        f"<div style='color:#5b6b7b;font-size:13px'>"
        f"{risk_emoji(prob)} SYSTEM OUTAGE PROBABILITY</div>"
        f"<div style='font-size:34px;font-weight:800;color:{risk_colour(prob)}'>"
        f"{prob:.1%}</div></div>", unsafe_allow_html=True)
for col, key, fmt in [
    (c2, "lstm_probability", "LSTM MODEL"),
    (c3, "prophet_probability", "PROPHET SEASONAL"),
    (c4, "anomaly_probability", "ANOMALY SCORE"),
]:
    value = float(prediction.get(key, 0))
    with col:
        st.markdown(
            f"<div style='background:#f7f9fc;border-radius:10px;padding:14px;'>"
            f"<div style='color:#5b6b7b;font-size:13px'>{fmt}</div>"
            f"<div style='font-size:26px;font-weight:700;color:#0a2b45'>"
            f"{value:.1%}</div></div>", unsafe_allow_html=True)

st.markdown("---")
left, right = st.columns([3, 2], gap="large")

with left:
    st.plotly_chart(risk_gauge(prob), use_container_width=True)

with right:
    st.subheader("Model Breakdown")
    st.plotly_chart(component_bars(prediction), use_container_width=True)
    risk_box = {"RED": st.error, "YELLOW": st.warning, "GREEN": st.success}[risk]
    risk_box(f"{risk_emoji(prob)} **{risk}** — {prediction.get('recommendation', '')}")
    st.caption(f"Horizon: {prediction.get('horizon_minutes', 120)} minutes · "
               f"models live: {prediction.get('models_loaded') or 'demo'}")

st.caption(f"Last updated: {prediction.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))[:19]}")

if auto:
    time.sleep(30)
    st.rerun()
