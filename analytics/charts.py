"""Plotly charts for the dashboards.

Colour rules: Present/Late/Absent and risk levels use the reserved status
palette (always alongside a text label); subjects use the categorical
palette in a fixed order keyed by subject id, so filtering never repaints a
subject. Single-axis charts only.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from core import config

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS_COLORS = {"present": "#0ca30c", "late": "#fab219", "absent": "#d03b3b"}
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
STATUS_ORDER = ["present", "late", "absent"]
MUTED = "#898781"


def subject_color(subject_id: int) -> str:
    return CATEGORICAL[(int(subject_id) - 1) % len(CATEGORICAL)]


def _layout(fig: go.Figure, title: str | None = None, height: int = 340, y_title: str | None = None,
            y_range: list | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)) if title else None,
        height=height, margin=dict(l=8, r=8, t=44 if title else 16, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hoverlabel=dict(font_size=12), bargap=0.35,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False, title=y_title, range=y_range)
    return fig


def _threshold_line(fig: go.Figure) -> None:
    thr = float(config.get("attendance.required_percentage", 75))
    fig.add_hline(y=thr, line=dict(color=MUTED, width=1, dash="dash"),
                  annotation_text=f"Required {thr:.0f}%", annotation_position="top left",
                  annotation_font_color=MUTED)


def status_bar(kpis: dict, title: str = "Present / Late / Absent") -> go.Figure:
    vals = [kpis.get(s, 0) for s in STATUS_ORDER]
    total = sum(vals) or 1
    fig = go.Figure(go.Bar(
        x=[s.title() for s in STATUS_ORDER], y=vals, marker_color=[STATUS_COLORS[s] for s in STATUS_ORDER],
        text=[f"{v} ({100 * v / total:.0f}%)" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y} records<extra></extra>", marker=dict(cornerradius=4)))
    return _layout(fig, title, y_title="Records")


def subject_bar(subj: pd.DataFrame, title: str = "Attendance by subject") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=subj["subject_code"], y=subj["attendance_pct"],
        marker_color=[subject_color(i) for i in subj["subject_id"]], marker=dict(cornerradius=4),
        text=[f"{v:.1f}%" for v in subj["attendance_pct"]], textposition="outside", cliponaxis=False,
        customdata=subj[["subject_name", "conducted"]].to_numpy(),
        hovertemplate="%{x} - %{customdata[0]}<br>Attendance %{y:.1f}%<br>%{customdata[1]} records<extra></extra>"))
    _threshold_line(fig)
    return _layout(fig, title, y_title="Attendance %", y_range=[0, 105])


def status_stack_by_subject(subj: pd.DataFrame, title: str = "Status mix by subject") -> go.Figure:
    fig = go.Figure()
    for s in STATUS_ORDER:
        fig.add_bar(name=s.title(), x=subj["subject_code"], y=subj[s], marker_color=STATUS_COLORS[s],
                    hovertemplate=f"%{{x}}: %{{y}} {s}<extra></extra>")
    fig.update_layout(barmode="stack")
    return _layout(fig, title, y_title="Records")


def trend_lines(trend: pd.DataFrame, x: str, y: str, title: str, y_title: str = "Attendance %") -> go.Figure:
    fig = go.Figure()
    for (code, sid), g in trend.groupby(["subject_code", "subject_id"], sort=True):
        fig.add_scatter(x=g[x], y=g[y], name=code, mode="lines+markers",
                        line=dict(width=2, color=subject_color(sid)), marker=dict(size=8),
                        hovertemplate=f"{code}<br>%{{x}}<br>%{{y:.1f}}%<extra></extra>")
    _threshold_line(fig)
    fig.update_layout(hovermode="x unified")
    return _layout(fig, title, y_title=y_title, y_range=[0, 105])


def monthly_bars(monthly: pd.DataFrame, title: str = "Month-wise attendance") -> go.Figure:
    fig = go.Figure()
    for (code, sid), g in monthly.groupby(["subject_code", "subject_id"], sort=True):
        fig.add_bar(name=code, x=g["month"], y=g["attendance_pct"], marker_color=subject_color(sid),
                    marker=dict(cornerradius=4), hovertemplate=f"{code} %{{x}}: %{{y:.1f}}%<extra></extra>")
    _threshold_line(fig)
    fig.update_layout(barmode="group", bargroupgap=0.08)
    return _layout(fig, title, y_title="Attendance %", y_range=[0, 105])


def risk_level_bar(risk: pd.DataFrame, title: str = "Risk levels (student x subject)") -> go.Figure:
    levels = ["Low", "Medium", "High"]
    counts = [int((risk["level"] == lv).sum()) for lv in levels]
    fig = go.Figure(go.Bar(x=levels, y=counts, marker_color=[RISK_COLORS[lv] for lv in levels],
                           marker=dict(cornerradius=4), text=counts, textposition="outside", cliponaxis=False,
                           hovertemplate="%{x} risk: %{y}<extra></extra>"))
    return _layout(fig, title, y_title="Student-subject pairs")


def roc_chart(metrics: dict) -> go.Figure:
    fig = go.Figure()
    for i, (key, name) in enumerate([("random_forest", "Random Forest"), ("logistic_regression", "Logistic Regression")]):
        m = metrics[key]
        fig.add_scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"], mode="lines",
                        name=f"{name} (AUC {m['roc_auc']:.3f})", line=dict(width=2, color=CATEGORICAL[i]),
                        hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color=MUTED, dash="dot", width=1))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    return _layout(fig, "ROC curve on held-out students", y_title="True positive rate", y_range=[0, 1.02])


def importance_bar(importance: dict) -> go.Figure:
    from ml.features import FEATURE_LABELS

    items = sorted(importance.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[FEATURE_LABELS.get(k, k) for k, _ in items],
                           orientation="h", marker_color=CATEGORICAL[0], marker=dict(cornerradius=4),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Importance")
    return _layout(fig, "Random Forest feature importance", height=360)


def anomaly_scatter(records: pd.DataFrame, title: str = "Kiosk records: match distance vs liveness") -> go.Figure:
    fig = go.Figure()
    for flagged, name, color, size in [(False, "Normal", MUTED, 8), (True, "Flagged", "#d03b3b", 11)]:
        g = records[records["flagged"] == flagged]
        fig.add_scatter(x=g["match_distance"], y=g["liveness_score"], mode="markers", name=name,
                        marker=dict(size=size, color=color, opacity=0.85, symbol="x" if flagged else "circle"),
                        customdata=g[["full_name", "check_in_time"]].to_numpy(),
                        hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>distance %{x:.3f}"
                                      "<br>liveness %{y:.2f}<extra></extra>")
    thr = float(config.get("recognition.match_threshold", 0.6))
    fig.add_vline(x=thr, line=dict(color=MUTED, dash="dash", width=1), annotation_text="match threshold",
                  annotation_font_color=MUTED)
    fig.update_xaxes(title="Match distance (lower = closer)")
    return _layout(fig, title, y_title="Liveness score")
