"""KPI metric-card helpers (dashboard)."""
from __future__ import annotations

import streamlit as st


def risk_colour(probability: float) -> str:
    if probability >= 0.70:
        return "#ff5252"
    if probability >= 0.30:
        return "#ffb300"
    return "#00c851"


def risk_emoji(probability: float) -> str:
    if probability >= 0.70:
        return "🔴"
    if probability >= 0.30:
        return "🟡"
    return "🟢"


def status_emoji(status: str) -> str:
    return {"EXCELLENT": "🟢", "GOOD": "✅", "DEGRADED": "🟠",
            "CRITICAL": "🔴"}.get(status.upper(), "⚪")


def metric_card(label: str, value: str, probability: float | None = None,
                help_text: str | None = None):
    """Colour-coded st.metric (probability in [0,1] drives the delta colour)."""
    if probability is None:
        st.metric(label, value, help=help_text)
        return
    colour = risk_colour(probability)
    st.markdown(
        f"""<div style="background:#f7f9fc;border-left:5px solid {colour};
        border-radius:8px;padding:12px 14px;margin-bottom:6px">
        <div style="font-size:13px;color:#5b6b7b">{label}</div>
        <div style="font-size:26px;font-weight:700;color:{colour}">{value}</div>
        </div>""",
        unsafe_allow_html=True)


def info_banner(source: str, demo: bool = False) -> None:
    if source == "offline":
        st.caption("⚠️ API offline — showing simulated telemetry (demo mode). "
                   "Start the FastAPI backend for live data.")
    elif demo:
        st.caption("Demo-mode prediction (models not trained yet). "
                   "Run model training for real ML inference.")
