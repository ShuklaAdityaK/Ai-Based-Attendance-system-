"""Reusable UI pieces: filter bar, risk badges, factor lists."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from core import db
from core.access import Actor

LEVEL_BADGE = {"Low": ("green", ":material/check_circle:"), "Medium": ("orange", ":material/warning:"),
               "High": ("red", ":material/error:")}
STATUS_BADGE = {"present": ("green", ":material/check:"), "late": ("orange", ":material/schedule:"),
                "absent": ("red", ":material/close:")}


def filter_bar(actor: Actor, key: str, with_student: bool = False) -> dict:
    """Date range + subject (+ student for admins) in one row."""
    subjects = db.list_subjects(actor)
    sub_opts = {None: "All subjects", **{s["id"]: f"{s['code']} - {s['name']}" for s in subjects}}
    with st.container(horizontal=True, vertical_alignment="bottom"):
        rng = st.date_input("Date range", value=(date.today() - timedelta(days=180), date.today()),
                            key=f"{key}_dates", width=260)
        subject_id = st.selectbox("Subject", list(sub_opts), format_func=sub_opts.get, key=f"{key}_subject",
                                  width=280)
        user_id = None
        if with_student and actor.is_admin:
            studs = db.students_with_roles(actor)
            stu_opts = {None: "All students", **{s["id"]: f"{s['roll_no'] or '-'} - {s['full_name']}" for s in studs}}
            user_id = st.selectbox("Student", list(stu_opts), format_func=stu_opts.get, key=f"{key}_student",
                                   width=280)
    d_from, d_to = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (None, None))
    return {"date_from": d_from, "date_to": d_to, "subject_id": subject_id, "user_id": user_id}


def level_badge(level: str, prob: float | None = None, method: str | None = None) -> None:
    color, icon = LEVEL_BADGE.get(level, ("gray", ":material/help:"))
    label = f"{level} risk" + (f" · {prob:.2f}" if prob is not None else "")
    if method == "rule":
        label += " (rule-based)"
    st.badge(label, icon=icon, color=color)


def factors_text(factors: list[dict]) -> str:
    if not factors:
        return "-"
    return "; ".join(f"{f['label']}: {f['value']} (typical {f['typical']})" for f in factors)


def status_label(s: str | None) -> str:
    return {"present": "Present", "late": "Late", "absent": "Absent"}.get(s or "", "-")


def empty_state(msg: str, icon: str = ":material/info:") -> None:
    st.info(msg, icon=icon)


def df_or_empty(df: pd.DataFrame, msg: str, **kwargs) -> None:
    if df is None or df.empty:
        empty_state(msg)
    else:
        st.dataframe(df, hide_index=True, **kwargs)
