"""Session-based attendance rules (Section 2.5) and admin corrections (2.6).

Status rules for a check-in at time t, session start S:
    Present : t <= S + grace
    Late    : S + grace < t <= S + late_cutoff
    (closed): t >  S + late_cutoff -> no record; becomes Absent on close
Absent is never produced by recognition, only by ``close_session``.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from core import config, db
from core.access import Actor, PermissionDenied, require_admin, require_actor, scope_user_id

PRESENT, LATE, ABSENT = "present", "late", "absent"
STATUSES = (PRESENT, LATE, ABSENT)


# --------------------------------------------------------------------------- #
# Pure rule functions (unit-tested at the boundaries)
# --------------------------------------------------------------------------- #

def session_start(session: dict) -> datetime:
    return datetime.fromisoformat(f"{session['date']}T{session['start_time']}")


def classify_check_in(check_in: datetime, start: datetime, grace_minutes: int,
                      late_cutoff_minutes: int) -> str | None:
    """Return 'present', 'late', or None when the check-in window is closed."""
    if check_in <= start + timedelta(minutes=grace_minutes):
        return PRESENT
    if check_in <= start + timedelta(minutes=late_cutoff_minutes):
        return LATE
    return None


def attendance_percentage(present: int, late: int, conducted: int, late_weight: float | None = None) -> float:
    """(Present + Late * late_weight) / sessions conducted, as a percentage."""
    if conducted <= 0:
        return 0.0
    w = float(config.get("attendance.late_weight", 1.0) if late_weight is None else late_weight)
    return 100.0 * (present + late * w) / conducted


def classes_needed(attended_weighted: float, conducted: int, required_pct: float | None = None) -> int:
    """Consecutive classes that must be attended to reach the threshold."""
    thr = float(config.get("attendance.required_percentage", 75) if required_pct is None else required_pct) / 100
    if conducted == 0 or attended_weighted / conducted >= thr:
        return 0
    if thr >= 1.0:
        return -1  # unreachable once any class is missed
    return max(0, math.ceil((thr * conducted - attended_weighted) / (1 - thr) - 1e-9))


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

def create_session(actor: Actor, subject_id: int, session_date: date | str, start_time: str,
                   grace_minutes: int | None = None, late_cutoff_minutes: int | None = None) -> int:
    require_admin(actor)
    grace = int(config.get("attendance.grace_minutes", 10) if grace_minutes is None else grace_minutes)
    cutoff = int(config.get("attendance.late_cutoff_minutes", 30) if late_cutoff_minutes is None
                 else late_cutoff_minutes)
    if cutoff < grace:
        raise ValueError("Late cutoff must be greater than or equal to the grace period.")
    d = session_date.isoformat() if isinstance(session_date, date) else str(session_date)
    start_time = datetime.strptime(start_time.strip()[:5], "%H:%M").strftime("%H:%M")
    with db.tx() as con:
        if not con.execute("SELECT 1 FROM subjects WHERE id=?", (subject_id,)).fetchone():
            raise ValueError("Unknown subject.")
        cur = con.execute(
            "INSERT INTO sessions(subject_id, date, start_time, grace_minutes, late_cutoff_minutes, status, "
            "created_by) VALUES (?,?,?,?,?, 'scheduled', ?)", (subject_id, d, start_time, grace, cutoff, actor.id))
        sid = int(cur.lastrowid)
        db.log_admin_action(con, actor, "create_session", f"session:{sid}",
                            {"subject_id": subject_id, "date": d, "start": start_time})
    return sid


def get_session(actor: Actor, session_id: int) -> dict | None:
    actor = require_actor(actor)
    with db.tx() as con:
        s = db.one(con, "SELECT s.*, sub.code, sub.name AS subject_name FROM sessions s "
                        "JOIN subjects sub ON sub.id=s.subject_id WHERE s.id=?", (session_id,))
        if s and not actor.is_admin and s["subject_id"] not in db.enrolled_subject_ids(con, actor.id):
            raise PermissionDenied("Not your session.")
        return s


def list_sessions(actor: Actor, subject_id: int | None = None, status: str | None = None,
                  date_from: date | None = None, date_to: date | None = None) -> list[dict]:
    actor = require_actor(actor)
    sql = ("SELECT s.*, sub.code, sub.name AS subject_name, "
           "(SELECT COUNT(*) FROM attendance a WHERE a.session_id=s.id AND a.status!='absent') AS checked_in, "
           "(SELECT COUNT(*) FROM enrollments e WHERE e.subject_id=s.subject_id) AS enrolled "
           "FROM sessions s JOIN subjects sub ON sub.id=s.subject_id WHERE 1=1")
    params: list = []
    if not actor.is_admin:
        sql += " AND s.subject_id IN (SELECT subject_id FROM enrollments WHERE user_id=?)"
        params.append(actor.id)
    if subject_id:
        sql += " AND s.subject_id=?"
        params.append(subject_id)
    if status:
        sql += " AND s.status=?"
        params.append(status)
    if date_from:
        sql += " AND s.date>=?"
        params.append(str(date_from))
    if date_to:
        sql += " AND s.date<=?"
        params.append(str(date_to))
    with db.tx() as con:
        return db.rows(con, sql + " ORDER BY s.date DESC, s.start_time DESC", params)


def open_session(actor: Actor, session_id: int) -> None:
    require_admin(actor)
    with db.tx() as con:
        s = db.one(con, "SELECT status FROM sessions WHERE id=?", (session_id,))
        if not s:
            raise ValueError("Session not found.")
        if s["status"] != "scheduled":
            raise ValueError(f"Session is already {s['status']}.")
        con.execute("UPDATE sessions SET status='open', opened_by=?, opened_at=? WHERE id=?",
                    (actor.id, db.now(), session_id))
        db.log_admin_action(con, actor, "open_session", f"session:{session_id}")


def close_session(actor: Actor, session_id: int, closed_at: datetime | None = None) -> int:
    """Close a session and auto-mark Absent for every enrolled student without
    a record. Returns the number of Absent records created."""
    require_admin(actor)
    ts = (closed_at or datetime.now()).isoformat(timespec="seconds")
    with db.tx() as con:
        s = db.one(con, "SELECT * FROM sessions WHERE id=?", (session_id,))
        if not s:
            raise ValueError("Session not found.")
        if s["status"] == "closed":
            raise ValueError("Session is already closed.")
        missing = [r[0] for r in con.execute(
            "SELECT e.user_id FROM enrollments e JOIN users u ON u.id=e.user_id "
            "WHERE e.subject_id=? AND u.role='student' AND e.user_id NOT IN "
            "(SELECT user_id FROM attendance WHERE session_id=?)", (s["subject_id"], session_id))]
        con.executemany(
            "INSERT OR IGNORE INTO attendance(session_id, user_id, status, source, created_at) "
            "VALUES (?,?,'absent','system',?)", [(session_id, uid, ts) for uid in missing])
        con.execute("UPDATE sessions SET status='closed', closed_at=? WHERE id=?", (ts, session_id))
        db.log_admin_action(con, actor, "close_session", f"session:{session_id}", {"auto_absent": len(missing)})
    return len(missing)


def delete_session(actor: Actor, session_id: int) -> None:
    """Only sessions that were never opened can be deleted."""
    require_admin(actor)
    with db.tx() as con:
        s = db.one(con, "SELECT status FROM sessions WHERE id=?", (session_id,))
        if not s:
            return
        if s["status"] != "scheduled":
            raise ValueError("Only scheduled (never opened) sessions can be deleted.")
        con.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        db.log_admin_action(con, actor, "delete_session", f"session:{session_id}")


# --------------------------------------------------------------------------- #
# Kiosk check-in
# --------------------------------------------------------------------------- #

@dataclass
class CheckInResult:
    ok: bool
    code: str            # marked | already | not_enrolled | window_closed | session_not_open
    message: str
    status: str | None = None


def mark_from_kiosk(actor: Actor, session_id: int, user_id: int, match_distance: float,
                    liveness_score: float, failed_attempts: int, check_in: datetime | None = None) -> CheckInResult:
    """Record a recognised, live student against an open session."""
    require_admin(actor)  # the kiosk runs under the admin who unlocked it
    t = check_in or datetime.now()
    with db.tx() as con:
        s = db.one(con, "SELECT * FROM sessions WHERE id=?", (session_id,))
        if not s or s["status"] != "open":
            return CheckInResult(False, "session_not_open", "This session is not open.")
        u = db.one(con, "SELECT full_name FROM users WHERE id=?", (user_id,))
        name = u["full_name"] if u else f"user {user_id}"
        if s["subject_id"] not in db.enrolled_subject_ids(con, user_id):
            return CheckInResult(False, "not_enrolled", f"{name} is not enrolled in this subject.")
        status = classify_check_in(t, session_start(s), s["grace_minutes"], s["late_cutoff_minutes"])
        if status is None:
            return CheckInResult(False, "window_closed",
                                 f"Check-in window closed for {name} (after the late cutoff).")
        try:
            con.execute(
                "INSERT INTO attendance(session_id, user_id, status, check_in_time, match_distance, "
                "liveness_score, failed_attempts, source, created_at) VALUES (?,?,?,?,?,?,?, 'kiosk', ?)",
                (session_id, user_id, status, t.isoformat(timespec="seconds"), float(match_distance),
                 float(liveness_score), int(failed_attempts), db.now()))
        except sqlite3.IntegrityError:
            prev = db.one(con, "SELECT status FROM attendance WHERE session_id=? AND user_id=?",
                          (session_id, user_id))
            return CheckInResult(False, "already", f"{name} is already marked "
                                                   f"({(prev or {}).get('status', '?').title()}).",
                                 (prev or {}).get("status"))
    config.log.info("Kiosk check-in session=%s user=%s status=%s dist=%.3f live=%.3f fails=%s",
                    session_id, user_id, status, match_distance, liveness_score, failed_attempts)
    return CheckInResult(True, "marked", f"{name}: marked {status.title()}.", status)


# --------------------------------------------------------------------------- #
# Admin corrections
# --------------------------------------------------------------------------- #

def correct_attendance(actor: Actor, session_id: int, user_id: int, new_status: str, reason: str) -> None:
    """Manually mark or change a status. A reason is mandatory; every change is
    written to the immutable ``attendance_audit`` table."""
    require_admin(actor)
    if new_status not in STATUSES:
        raise ValueError("Invalid status.")
    if not reason or not reason.strip():
        raise ValueError("A reason is required for every correction.")
    with db.tx() as con:
        s = db.one(con, "SELECT * FROM sessions WHERE id=?", (session_id,))
        if not s:
            raise ValueError("Session not found.")
        if s["status"] == "scheduled":
            raise ValueError("Open the session before marking attendance.")
        if s["subject_id"] not in db.enrolled_subject_ids(con, user_id):
            raise ValueError("Student is not enrolled in this subject.")
        rec = db.one(con, "SELECT * FROM attendance WHERE session_id=? AND user_id=?", (session_id, user_id))
        ts = db.now()
        if rec:
            if rec["status"] == new_status:
                raise ValueError("The record already has that status.")
            con.execute("UPDATE attendance SET status=?, source='manual', updated_at=? WHERE id=?",
                        (new_status, ts, rec["id"]))
            att_id, old = rec["id"], rec["status"]
        else:
            check_in = ts if new_status != ABSENT else None
            cur = con.execute("INSERT INTO attendance(session_id, user_id, status, check_in_time, source, "
                              "created_at) VALUES (?,?,?,?, 'manual', ?)",
                              (session_id, user_id, new_status, check_in, ts))
            att_id, old = int(cur.lastrowid), None
        con.execute("INSERT INTO attendance_audit(attendance_id, session_id, user_id, old_status, new_status, "
                    "changed_by, reason, changed_at) VALUES (?,?,?,?,?,?,?,?)",
                    (att_id, session_id, user_id, old, new_status, actor.id, reason.strip(), ts))
    config.log.info("Correction session=%s user=%s %s->%s by %s", session_id, user_id, old, new_status,
                    actor.username)


def list_corrections(actor: Actor, limit: int = 1000) -> list[dict]:
    require_admin(actor)
    with db.tx() as con:
        return db.rows(con, "SELECT aa.*, u.full_name, u.roll_no, adm.username AS changed_by_name, "
                            "sub.code, s.date FROM attendance_audit aa "
                            "LEFT JOIN users u ON u.id=aa.user_id LEFT JOIN users adm ON adm.id=aa.changed_by "
                            "LEFT JOIN sessions s ON s.id=aa.session_id LEFT JOIN subjects sub ON sub.id=s.subject_id "
                            "ORDER BY aa.id DESC LIMIT ?", (limit,))


# --------------------------------------------------------------------------- #
# Record queries (scoped)
# --------------------------------------------------------------------------- #

def get_records(actor: Actor, user_id: int | None = None, subject_id: int | None = None,
                date_from: date | None = None, date_to: date | None = None,
                closed_only: bool = False, session_id: int | None = None) -> list[dict]:
    """Attendance records joined with session/subject/user info.

    Students are pinned to their own records here, in the data layer."""
    uid = scope_user_id(actor, user_id)
    sql = ("SELECT a.*, s.date, s.start_time, s.status AS session_status, s.subject_id, "
           "sub.code AS subject_code, sub.name AS subject_name, u.full_name, u.roll_no, u.username "
           "FROM attendance a JOIN sessions s ON s.id=a.session_id JOIN subjects sub ON sub.id=s.subject_id "
           "JOIN users u ON u.id=a.user_id WHERE 1=1")
    params: list = []
    if uid is not None:
        sql += " AND a.user_id=?"
        params.append(uid)
    if subject_id:
        sql += " AND s.subject_id=?"
        params.append(subject_id)
    if session_id:
        sql += " AND a.session_id=?"
        params.append(session_id)
    if date_from:
        sql += " AND s.date>=?"
        params.append(str(date_from))
    if date_to:
        sql += " AND s.date<=?"
        params.append(str(date_to))
    if closed_only:
        sql += " AND s.status='closed'"
    with db.tx() as con:
        return db.rows(con, sql + " ORDER BY s.date, s.start_time, u.roll_no", params)


def session_roster(actor: Actor, session_id: int) -> list[dict]:
    """Every enrolled student of a session with their current record (if any)."""
    require_admin(actor)
    with db.tx() as con:
        return db.rows(con, "SELECT u.id AS user_id, u.full_name, u.roll_no, a.id AS attendance_id, a.status, "
                            "a.check_in_time, a.source, a.match_distance, a.liveness_score "
                            "FROM sessions s JOIN enrollments e ON e.subject_id=s.subject_id "
                            "JOIN users u ON u.id=e.user_id "
                            "LEFT JOIN attendance a ON a.session_id=s.id AND a.user_id=u.id "
                            "WHERE s.id=? AND u.role='student' ORDER BY u.roll_no, u.full_name", (session_id,))
