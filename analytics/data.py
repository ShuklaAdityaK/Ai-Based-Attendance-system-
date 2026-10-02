"""Attendance analytics with Pandas (Section 2.9).

All frames are built from role-scoped queries, so a student's analytics can
only ever contain their own records. Percentages use closed sessions only
(an open session is not yet final).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from core import attendance, config
from core.access import Actor


def records_frame(actor: Actor, user_id: int | None = None, subject_id: int | None = None,
                  date_from: date | None = None, date_to: date | None = None) -> pd.DataFrame:
    df = pd.DataFrame(attendance.get_records(actor, user_id, subject_id, date_from, date_to, closed_only=True))
    if df.empty:
        return df
    w = float(config.get("attendance.late_weight", 1.0))
    df["credit"] = df["status"].map({"present": 1.0, "late": w, "absent": 0.0})
    df["date_dt"] = pd.to_datetime(df["date"])
    df["month"] = df["date_dt"].dt.to_period("M").astype(str)
    return df


def _summarise(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    p, l, a = (g["status"] == "present").sum(), (g["status"] == "late").sum(), (g["status"] == "absent").sum()
    credit = g["credit"].sum()
    return pd.Series({"present": int(p), "late": int(l), "absent": int(a), "conducted": int(n),
                      "attendance_pct": round(100 * credit / n, 1) if n else 0.0,
                      "classes_needed": attendance.classes_needed(credit, n)})


INT_COLS = ["present", "late", "absent", "conducted", "classes_needed"]


def _ints(df: pd.DataFrame) -> pd.DataFrame:
    for c in INT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df


def student_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(["user_id", "roll_no", "full_name"]).apply(_summarise, include_groups=False).reset_index()
    return _ints(out).sort_values("attendance_pct")


def student_subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (df.groupby(["user_id", "roll_no", "full_name", "subject_id", "subject_code", "subject_name"])
           .apply(_summarise, include_groups=False).reset_index())
    return _ints(out).sort_values(["subject_code", "attendance_pct"])


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    thr = float(config.get("attendance.required_percentage", 75))
    base = df.groupby(["subject_id", "subject_code", "subject_name"]).apply(_summarise, include_groups=False)
    per_student = student_subject_summary(df)
    below = per_student[per_student["attendance_pct"] < thr].groupby("subject_id").size()
    sessions = df.groupby("subject_id")["session_id"].nunique()
    base = base.reset_index()
    base["sessions"] = base["subject_id"].map(sessions).fillna(0).astype(int)
    base["students_below"] = base["subject_id"].map(below).fillna(0).astype(int)
    return _ints(base).drop(columns=["classes_needed"])


def overall_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"pct": 0.0, "present": 0, "late": 0, "absent": 0, "records": 0, "sessions": 0}
    return {"pct": round(100 * df["credit"].sum() / len(df), 1),
            "present": int((df["status"] == "present").sum()), "late": int((df["status"] == "late").sum()),
            "absent": int((df["status"] == "absent").sum()), "records": len(df),
            "sessions": int(df["session_id"].nunique())}


def session_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Attendance % per session, ordered by date, per subject."""
    if df.empty:
        return pd.DataFrame()
    t = (df.groupby(["subject_code", "subject_id", "session_id", "date_dt"])["credit"].mean().mul(100)
         .round(1).reset_index(name="attendance_pct").sort_values("date_dt"))
    return t


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (df.groupby(["month", "subject_code", "subject_id"])["credit"].mean().mul(100).round(1)
            .reset_index(name="attendance_pct").sort_values("month"))


def cumulative_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Running attendance % per subject (used on the student dashboard)."""
    if df.empty:
        return pd.DataFrame()
    d = df.sort_values(["date_dt", "start_time"]).copy()
    g = d.groupby("subject_code")
    d["cum_pct"] = (g["credit"].cumsum() / (g.cumcount() + 1) * 100).round(1)
    return d[["subject_code", "subject_id", "date_dt", "cum_pct", "status"]]


def low_attendance(df: pd.DataFrame) -> pd.DataFrame:
    thr = float(config.get("attendance.required_percentage", 75))
    s = student_subject_summary(df)
    if s.empty:
        return s
    return s[s["attendance_pct"] < thr].sort_values("attendance_pct")
