"""Reusable Plotly chart components (dashboard)."""
from __future__ import annotations

import plotly.graph_objects as go

RISK_THRESHOLD = 70.0   # % — the SRE alert line
WARNING_LEVEL = 30.0


def risk_gauge(probability: float) -> go.Figure:
    """0-100% outage-risk dial with green/yellow/red zones."""
    percent = probability * 100
    color = ("#ff5252" if percent >= RISK_THRESHOLD
             else "#ffb300" if percent >= WARNING_LEVEL else "#00c851")
    figure = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(percent, 1),
        number={"suffix": "%", "font": {"size": 44, "color": color}},
        title={"text": "Outage Risk (next 2 hrs)", "font": {"size": 18}},
        delta={"reference": RISK_THRESHOLD, "position": "bottom",
               "increasing": {"color": "#ff5252"},
               "decreasing": {"color": "#00c851"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#ddd",
            "steps": [
                {"range": [0, WARNING_LEVEL], "color": "#d9f6e1"},
                {"range": [WARNING_LEVEL, RISK_THRESHOLD], "color": "#fff3cd"},
                {"range": [RISK_THRESHOLD, 100], "color": "#ffd7d5"},
            ],
            "threshold": {"line": {"color": "red", "width": 4},
                          "thickness": 0.85, "value": RISK_THRESHOLD},
        }))
    figure.update_layout(height=330, margin={"t": 60, "b": 10, "l": 30, "r": 30})
    return figure


def history_line(records: list[dict]) -> go.Figure:
    """Prediction history with colour zones + threshold lines."""
    probs = [float(r.get("outage_probability") or r.get("final_probability") or 0)
             for r in records]
    labels = []
    for r in records:
        stamp = r.get("predicted_at") or r.get("timestamp") or ""
        labels.append(stamp[11:19] if len(stamp) > 19 else stamp)

    figure = go.Figure()
    figure.add_hrect(y0=0, y1=WARNING_LEVEL / 100, fillcolor="#d9f6e1",
                     opacity=0.4, line_width=0, layer="below")
    figure.add_hrect(y0=WARNING_LEVEL / 100, y1=RISK_THRESHOLD / 100,
                     fillcolor="#fff3cd", opacity=0.4, line_width=0,
                     layer="below")
    figure.add_hrect(y0=RISK_THRESHOLD / 100, y1=1.05, fillcolor="#ffd7d5",
                     opacity=0.4, line_width=0, layer="below")
    figure.add_hline(y=RISK_THRESHOLD / 100, line_dash="dash",
                     line_color="red",
                     annotation_text="SRE alert threshold (70%)")
    figure.add_hline(y=WARNING_LEVEL / 100, line_dash="dot",
                     line_color="orange",
                     annotation_text="warning (30%)")
    figure.add_trace(go.Scatter(
        x=list(range(len(probs))), y=probs, mode="lines+markers",
        line={"color": "#123c5f", "width": 2.5},
        marker={"size": 6, "color": probs,
                "colorscale": [[0, "#00c851"], [0.3, "#ffb300"], [1, "#ff5252"]]},
        hovertemplate="%{y:.1%} @ %{text}<extra></extra>",
        text=labels))
    figure.update_layout(
        height=360, yaxis={"range": [0, 1.05], "tickformat": ".0%",
                           "title": "Outage probability"},
        xaxis={"title": f"Predictions (n={len(probs)})"},
        margin={"t": 20, "b": 30, "l": 50, "r": 20},
        showlegend=False)
    return figure


def bank_health_bars(scores: dict) -> go.Figure:
    """Horizontal bar chart, colour-coded by health band."""
    names = list(scores.keys())
    values = [float(d["score"]) for d in scores.values()]
    colours = []
    for value in values:
        if value >= 90:
            colours.append("#00c851")
        elif value >= 70:
            colours.append("#ffb300")
        elif value >= 50:
            colours.append("#ff8f00")
        else:
            colours.append("#ff5252")

    figure = go.Figure(go.Bar(
        y=names, x=values, orientation="h",
        marker={"color": colours, "line": {"color": "#333", "width": 0.6}},
        text=[f"{v:.1f}" for v in values], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}/100<extra></extra>"))
    figure.add_vline(x=RISK_THRESHOLD, line_dash="dash", line_color="#ff5252",
                     annotation_text=" critical 30")
    figure.add_vline(x=70, line_dash="dot", line_color="#ffb300",
                     annotation_text=" warning 70")
    figure.update_layout(
        height=380, xaxis={"range": [0, 110], "title": "Health score"},
        yaxis={"autorange": "reversed"},
        margin={"t": 20, "b": 30, "l": 20, "r": 40})
    return figure


def component_bars(prediction: dict) -> go.Figure:
    """LSTM vs Prophet vs Anomaly — the ensemble breakdown."""
    components = [
        ("LSTM", float(prediction.get("lstm_probability", 0))),
        ("Prophet", float(prediction.get("prophet_probability", 0))),
        ("Anomaly", float(prediction.get("anomaly_probability", 0))),
        ("Ensemble", float(prediction.get("final_probability", 0))),
    ]
    names = [c[0] for c in components]
    values = [c[1] for c in components]
    colours = ["#123c5f", "#0a6ebd", "#9c27b0", "#ff5252"]
    figure = go.Figure(go.Bar(
        x=names, y=[v * 100 for v in values], marker_color=colours,
        text=[f"{v:.1%}" for v in values], textposition="outside"))
    figure.update_layout(
        height=280, yaxis={"range": [0, 100], "title": "%"},
        margin={"t": 20, "b": 20, "l": 40, "r": 20},
        showlegend=False)
    return figure
