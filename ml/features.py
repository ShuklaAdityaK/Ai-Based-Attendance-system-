"""Feature engineering from attendance history (Section 2.7).

One feature vector per student per subject, computed from the ordered list
of that student's records in *closed* sessions of the subject.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core import attendance, config, db
from core.access import Actor, require_admin

FEATURES = [
    "pct_to_date",
    "pct_last5",
    "pct_last10",
    "late_ratio",
    "longest_absent_run",
    "current_absent_run",
    "dow_absence_concentration",
    "sessions_remaining",
    "max_achievable_pct",
]

FEATURE_LABELS = {
    "pct_to_date": "Attendance to date",
    "pct_last5": "Attendance over last 5 sessions",
    "pct_last10": "Attendance over last 10 sessions",
    "late_ratio": "Share of sessions arrived late",
    "longest_absent_run": "Longest absence streak",
    "current_absent_run": "Current absence streak",
    "dow_absence_concentration": "Absences concentrated on one weekday",
    "sessions_remaining": "Sessions remaining",
    "max_achievable_pct": "Maximum achievable attendance",
}


def _credit(status: str, late_weight: float) -> float:
    return 1.0 if status == "present" else (late_weight if status == "late" else 0.0)


def compute_features(statuses: list[str], weekdays: list[int], planned_sessions: int,
                     late_weight: float | None = None) -> dict[str, float]:
    """Features from an ordered status list ('present'/'late'/'absent')."""
    w = float(config.get("attendance.late_weight", 1.0) if late_weight is None else late_weight)
    n = len(statuses)
    credits = np.array([_credit(s, w) for s in statuses], dtype=float)
    absent = np.array([s == "absent" for s in statuses], dtype=bool)

    def pct(arr: np.ndarray) -> float:
        return float(100 * arr.mean()) if len(arr) else 0.0

    longest = cur = 0
    for a in absent:
        cur = cur + 1 if a else 0
        longest = max(longest, cur)

    dow_conc = 0.0
    if n and absent.any():
        overall = absent.mean()
        rates = [absent[np.array(weekdays) == d].mean() for d in set(weekdays) if (np.array(weekdays) == d).sum() >= 2]
        dow_conc = float(max(rates) - overall) if rates else 0.0

    planned = max(int(planned_sessions), n)
    remaining = planned - n
    return {
        "pct_to_date": pct(credits),
        "pct_last5": pct(credits[-5:]),
        "pct_last10": pct(credits[-10:]),
        "late_ratio": float(np.mean([s == "late" for s in statuses])) if n else 0.0,
        "longest_absent_run": float(longest),
        "current_absent_run": float(cur),
        "dow_absence_concentration": dow_conc,
        "sessions_remaining": float(remaining),
        "max_achievable_pct": float(100 * (credits.sum() + remaining) / planned) if planned else 0.0,
    }


def student_subject_histories(actor: Actor, user_id: int | None = None) -> list[dict]:
    """Ordered closed-session history for every (student, subject) pair.

    Pairs with an enrollment but no closed sessions are included (n = 0)."""
    from core.access import scope_user_id

    uid = scope_user_id(actor, user_id)
    sql = ("SELECT e.user_id, e.subject_id, sub.code, sub.name AS subject_name, sub.planned_sessions, "
           "u.full_name, u.roll_no FROM enrollments e JOIN subjects sub ON sub.id=e.subject_id "
           "JOIN users u ON u.id=e.user_id WHERE u.role='student'")
    params: list = []
    if uid is not None:
        sql += " AND e.user_id=?"
        params.append(uid)
    with db.tx() as con:
        pairs = db.rows(con, sql, params)
        recs = db.rows(con, "SELECT a.user_id, s.subject_id, s.date, s.start_time, a.status FROM attendance a "
                            "JOIN sessions s ON s.id=a.session_id WHERE s.status='closed' "
                            + ("AND a.user_id=? " if uid is not None else "")
                            + "ORDER BY s.date, s.start_time", [uid] if uid is not None else [])
    by_key: dict[tuple[int, int], list[dict]] = {}
    for r in recs:
        by_key.setdefault((r["user_id"], r["subject_id"]), []).append(r)
    out = []
    for p in pairs:
        hist = by_key.get((p["user_id"], p["subject_id"]), [])
        p["statuses"] = [h["status"] for h in hist]
        p["weekdays"] = [date.fromisoformat(h["date"]).weekday() for h in hist]
        p["dates"] = [h["date"] for h in hist]
        out.append(p)
    return out


def feature_frame(actor: Actor, user_id: int | None = None) -> pd.DataFrame:
    w = float(config.get("attendance.late_weight", 1.0))
    rows = []
    for h in student_subject_histories(actor, user_id):
        f = compute_features(h["statuses"], h["weekdays"], h["planned_sessions"], w)
        rows.append({"user_id": h["user_id"], "subject_id": h["subject_id"], "code": h["code"],
                     "subject_name": h["subject_name"], "full_name": h["full_name"], "roll_no": h["roll_no"],
                     "n_sessions": len(h["statuses"]), "planned_sessions": h["planned_sessions"], **f})
    return pd.DataFrame(rows)


def completed_real_samples(actor: Actor) -> pd.DataFrame:
    """Training samples from real subjects whose semester is complete
    (conducted >= planned). Label = final attendance below the threshold.
    Several checkpoints per history are produced, mirroring the synthetic set."""
    require_admin(actor)
    w = float(config.get("attendance.late_weight", 1.0))
    thr = float(config.get("attendance.required_percentage", 75))
    min_n = int(config.get("ml.min_sessions_for_model", 5))
    rows = []
    for h in student_subject_histories(actor):
        n, planned = len(h["statuses"]), int(h["planned_sessions"])
        if n < planned or n < min_n:
            continue
        final = attendance.attendance_percentage(h["statuses"].count("present"), h["statuses"].count("late"), n, w)
        for t in range(min_n, n, 3):
            f = compute_features(h["statuses"][:t], h["weekdays"][:t], planned, w)
            rows.append({**f, "label": int(final < thr), "group": f"real-{h['user_id']}-{h['subject_id']}",
                         "source": "real"})
    return pd.DataFrame(rows)
