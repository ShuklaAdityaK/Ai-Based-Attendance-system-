"""Shared report data assembly (role-scoped)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from analytics import data as an
from core import config, db
from core.access import Actor, scope_user_id
from ml import risk_model


@dataclass
class ReportFilters:
    date_from: date | None = None
    date_to: date | None = None
    subject_id: int | None = None
    user_id: int | None = None


@dataclass
class ReportData:
    filters: ReportFilters
    subject_label: str
    student_label: str
    generated_by: str
    generated_at: str
    records: pd.DataFrame
    students: pd.DataFrame
    subjects: pd.DataFrame
    student_subjects: pd.DataFrame
    kpis: dict
    low: pd.DataFrame
    risk: pd.DataFrame = field(default_factory=pd.DataFrame)


def build(actor: Actor, f: ReportFilters) -> ReportData:
    uid = scope_user_id(actor, f.user_id)       # students are pinned to themselves
    f = ReportFilters(f.date_from, f.date_to, f.subject_id, uid)
    recs = an.records_frame(actor, uid, f.subject_id, f.date_from, f.date_to)
    subject_label, student_label = "All subjects", "All students"
    with db.tx() as con:
        if f.subject_id:
            s = db.one(con, "SELECT code, name FROM subjects WHERE id=?", (f.subject_id,))
            subject_label = f"{s['code']} - {s['name']}" if s else str(f.subject_id)
        if uid:
            u = db.one(con, "SELECT full_name, roll_no FROM users WHERE id=?", (uid,))
            student_label = f"{u['full_name']} ({u['roll_no'] or '-'})" if u else str(uid)
    risk = risk_model.stored_predictions(actor, uid)
    if not risk.empty and f.subject_id:
        risk = risk[risk["subject_id"] == f.subject_id]
    return ReportData(
        filters=f, subject_label=subject_label, student_label=student_label,
        generated_by=actor.full_name or actor.username,
        generated_at=datetime.now().strftime("%d %b %Y, %H:%M"),
        records=recs, students=an.student_summary(recs), subjects=an.subject_summary(recs),
        student_subjects=an.student_subject_summary(recs), kpis=an.overall_kpis(recs),
        low=an.low_attendance(recs), risk=risk)


def date_range_label(f: ReportFilters) -> str:
    a = f.date_from.strftime("%d %b %Y") if f.date_from else "start"
    b = f.date_to.strftime("%d %b %Y") if f.date_to else "today"
    return f"{a} to {b}"


def report_filename(prefix: str, ext: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"


def save_copy(content: bytes, filename: str) -> None:
    d = config.path("reports")
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_bytes(content)
