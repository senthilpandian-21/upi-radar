"""Page 5 — Post-outage PDF reports."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.api_client import API_BASE, reports_list  # noqa: E402

st.header("📄 Post-Outage Reports")
st.caption("Auto-generated PDF after every outage — RCA-ready summaries "
           "with the full routing audit trail.")

if st.button("🧪 Generate demo report now"):
    with st.spinner("Building PDF…"):
        try:
            response = requests.post(f"{API_BASE}/reports/generate",
                                     timeout=10)
            if response.status_code == 200:
                st.success(f"Report generated: "
                           f"{response.json().get('filename')}")
            else:
                st.warning(f"API returned {response.status_code} — "
                           "falling back to offline generation…")
        except Exception:
            st.warning("API offline — falling back to offline generation…")
        try:
            from datetime import datetime, timedelta

            from agent.report_agent import ReportAgent

            path = ReportAgent().generate_post_outage_report(
                datetime.now() - timedelta(hours=2), datetime.now())
            st.success(f"Report generated: {path.split('/')[-1]}")
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

reports = reports_list()
if reports:
    st.markdown(f"**{len(reports)} report(s)** generated:")
    for report in reports:
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.markdown(f"📎 `{report['filename']}`  \n"
                        f"<small>{report['created']} · "
                        f"{report['size_bytes'] / 1024:.1f} KB</small>",
                        unsafe_allow_html=True)
        with c2:
            st.link_button("View", f"{API_BASE}/reports/download/"
                                   f"{report['filename']}",
                           width="stretch")
        with c3:
            try:
                payload = requests.get(
                    f"{API_BASE}/reports/download/{report['filename']}",
                    timeout=5)
                st.download_button("⬇", data=payload.content,
                                   file_name=report["filename"], mime="application/pdf",
                                   width="stretch")
            except Exception:
                st.button("⬇", disabled=True, width="stretch")
        st.divider()
else:
    st.info("No reports yet — click 'Generate demo report' above or wait "
            "for the next simulated outage.")
