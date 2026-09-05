"""Page 2 — Bank Health: per-bank scores, fallback ranking."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.api_client import bank_health  # noqa: E402
from dashboard.components.charts import bank_health_bars  # noqa: E402
from dashboard.components.metrics import info_banner, status_emoji  # noqa: E402

st.header("🏦 Bank Health Scores (0–100)")
payload = bank_health()
scores = payload["bank_health_scores"]
info_banner(payload.get("source", "api"))

st.plotly_chart(bank_health_bars(scores), width="stretch")

critical = payload.get("critical_banks", [])
warning = payload.get("warning_banks", [])
best = payload.get("best_available_banks", [])

if critical:
    st.error(f"🚨 CRITICAL BANKS: {', '.join(critical)} — "
             "auto-routing to healthy banks is ACTIVE (Rule B1)")
elif warning:
    st.warning(f"⚠️ Degraded banks: {', '.join(warning)} — "
               "medium-risk routing active")
else:
    st.success("✅ All banks healthy — no routing interventions needed")

if best:
    st.info(f"🛡️ Fallback order: {' → '.join(best[:4])}")

# table
rows = [{"Bank": bank, "Health": float(data["score"]),
         "Status": data.get("status", ""),
         "Updated": (data.get("updated_at") or "")[:19]}
        for bank, data in scores.items()]
frame = pd.DataFrame(rows).sort_values("Health", ascending=False)
frame.insert(0, "", frame["Status"].map(status_emoji))
st.dataframe(frame.style.format({"Health": "{:.1f}"}),
             width="stretch", hide_index=True)

st.caption(f"Source: {payload.get('source')} · "
           f"updated {payload.get('timestamp', '')[:19]}")
