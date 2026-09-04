"""⚡ UPI RADAR — Streamlit dashboard (entry point).

    streamlit run dashboard/app.py --server.port 8501

Multi-page: pages/1_Overview.py … 5_Reports.py appear in the sidebar.
Works fully in demo mode (no API, no models, no infra required).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard.components.metrics import risk_emoji  # noqa: E402

st.set_page_config(
    page_title="UPI Radar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar header ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:8px 0">
          <div style="font-size:38px">⚡</div>
          <div style="font-size:20px;font-weight:800;color:#0a2b45">UPI RADAR</div>
          <div style="font-size:11px;color:#5b6b7b">AI-Powered Payment Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Razorpay AI Buildathon 2026 · Track 5**")
    st.caption("Predict outages 2h ahead · route smart · queue gracefully")

st.title("⚡ UPI RADAR — Real-Time Payment Intelligence")
st.caption("AI-powered UPI outage prediction and smart routing — "
           "built for Razorpay's SRE team & merchants")

# landing hint (pages render their own content when selected)
st.markdown("---")
col_note, col_alert = st.columns([3, 2])
with col_note:
    st.markdown(
        """
        ### 👋 Welcome

        Use the **sidebar** to navigate:

        | Page | What it shows |
        |---|---|
        | 📊 **Overview** | live outage gauge + ensemble model breakdown |
        | 🏦 **Bank Health** | per-bank health scores, best fallback banks |
        | 📈 **Predictions** | outage probability over time |
        | 🔄 **Routing Log** | every smart-routing decision, audited |
        | 📄 **Reports** | auto-generated post-outage PDFs |
        """,
        unsafe_allow_html=True,
    )
with col_alert:
    try:
        from dashboard.api_client import predict_outage

        pred = predict_outage()
        prob = float(pred["final_probability"])
        st.markdown(
            f"### {risk_emoji(prob)} Live Risk: **{prob:.1%}**\n\n"
            f"_{pred.get('recommendation', '')}_",
            unsafe_allow_html=True,
        )
        st.caption(f"Updated {pred.get('timestamp', '')[:19]} · "
                   f"source: {pred.get('source', 'api')}")
    except Exception:
        st.info("Start the FastAPI backend for live risk telemetry.")
