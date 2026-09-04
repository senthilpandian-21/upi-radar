"""Page 4 — Routing log: every smart-routing decision, audited."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.api_client import routing_history  # noqa: E402

st.header("🔄 Routing Decisions — Audit Trail")
st.caption("Every decision logged with timestamp, reason and outcome "
           "(DB + audit_logs/routing_decisions.log)")

rows = routing_history(limit=200)

if not rows:
    st.info("No routing decisions recorded yet. Run the **monitor agent** "
            "or POST to `/route/transaction` to generate traffic.")
else:
    frame = pd.DataFrame(rows)
    frame["success_icon"] = frame.get("success", pd.Series(dtype=bool)) \
        .map({True: "✅", False: "❌"})
    frame["queued_icon"] = frame.get("queued", pd.Series(dtype=bool)) \
        .fillna(False).map({True: "⏳", False: ""})

    # filters
    f1, f2, f3 = st.columns(3)
    bank_options = ["All"] + sorted(frame["original_bank"].dropna().unique().tolist())
    bank_filter = f1.selectbox("Original bank", bank_options)
    status_options = ["All", "Success", "Failed", "Queued"]
    status_filter = f2.selectbox("Outcome", status_options)
    rows_filter = f3.selectbox("Max rows shown", [20, 50, 100, 200])

    filtered = frame
    if bank_filter != "All":
        filtered = filtered[filtered["original_bank"] == bank_filter]
    if status_filter == "Success":
        filtered = filtered[filtered["success"] == True]  # noqa: E712
    elif status_filter == "Failed":
        filtered = filtered[filtered["success"] == False]  # noqa: E712
    elif status_filter == "Queued":
        filtered = filtered[filtered.get("queued", False) == True]  # noqa: E712

    total = len(frame)
    successes = int(frame["success"].fillna(True).astype(bool).sum()) \
        if "success" in frame else total
    success_rate = successes / total if total else 0.0
    rerouted = int((frame["routed_to_bank"] != frame["original_bank"]).sum()) \
        if "routed_to_bank" in frame and "original_bank" in frame else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total decisions", total)
    m2.metric("Success rate", f"{success_rate:.1%}")
    m3.metric("Rerouted transactions", rerouted)
    m4.metric("Active queue ⏳",
              int(frame["queued"].fillna(False).sum()) if "queued" in frame else 0)

    show = filtered.tail(rows_filter).iloc[::-1]
    columns = [c for c in ["timestamp", "transaction_id", "original_bank",
                           "routed_to_bank", "original_method", "routed_to_method",
                           "rule_id", "reason", "outage_prob_at_time",
                           "success_icon", "queued_icon"]
               if c in show.columns]
    st.dataframe(show[columns].rename(columns={
        "timestamp": "Time", "transaction_id": "Txn ID",
        "original_bank": "Original", "routed_to_bank": "Routed To",
        "original_method": "Method", "routed_to_method": "New Method",
        "rule_id": "Rule", "reason": "Reason",
        "outage_prob_at_time": "P(outage)", "success_icon": "Status",
        "queued_icon": "Queue"}), use_container_width=True, hide_index=True)
