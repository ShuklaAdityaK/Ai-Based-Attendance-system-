"""SQLite schema and role-checked queries for users, subjects, enrollments
and face data.

Attendance-specific queries live in ``core.attendance``; all public functions
take the calling ``Actor`` and enforce access in this layer.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

import numpy as np

from core import config
from core.access import (
    ROLE_ADMIN,
    ROLE_STUDENT,
    Actor,
    PermissionDenied,
    require_actor,
    require_admin,
    require_self_or_admin,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name TEXT NOT NULL,
    roll_no TEXT UNIQUE,
    email TEXT,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('student','admin')),
    is_active INTEGER NOT NULL DEFAULT 0,
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE COLLATE NOCASE,
    name TEXT NOT NULL,
    planned_sessions INTEGER NOT NULL DEFAULT 40
);

CREATE TABLE IF NOT EXISTS enrollments (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, subject_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    date TEXT NOT NULL,
    start_time TEXT NOT NULL,
    grace_minutes INTEGER NOT NULL,
    late_cutoff_minutes INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'scheduled' CHECK (status IN ('scheduled','open','closed')),
    created_by INTEGER,
    opened_by INTEGER,
    opened_at TEXT,
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS face_enrollments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','approved','rejected','superseded','blocked')),
    consent_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    frame_count INTEGER NOT NULL DEFAULT 0,
    quality_summary TEXT,
    reviewed_by INTEGER,
    reviewed_at TEXT,
    review_note TEXT
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    enrollment_id INTEGER NOT NULL REFERENCES face_enrollments(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('present','late','absent')),
    check_in_time TEXT,
    match_distance REAL,
    liveness_score REAL,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL CHECK (source IN ('kiosk','manual','system')),
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE (session_id, user_id)
);

CREATE TABLE IF NOT EXISTS attendance_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attendance_id INTEGER,
    session_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    old_status TEXT,
    new_status TEXT NOT NULL,
    changed_by INTEGER NOT NULL,
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    changed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS anomaly_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attendance_id INTEGER NOT NULL UNIQUE REFERENCES attendance(id) ON DELETE CASCADE,
    score REAL NOT NULL,
    top_features TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending','confirmed','dismissed')),
    reviewed_by INTEGER,
    reviewed_at TEXT,
    notes TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    probability REAL NOT NULL,
    level TEXT NOT NULL,
    method TEXT NOT NULL,
    top_factors TEXT NOT NULL,
    model_version TEXT,
    generated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_by INTEGER,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS admin_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id INTEGER,
    actor_name TEXT,
    action TEXT NOT NULL,
    target TEXT,
    details TEXT,
    at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    related_user_id INTEGER,
    is_read INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_att_user ON attendance(user_id);
CREATE INDEX IF NOT EXISTS ix_att_session ON attendance(session_id);
CREATE INDEX IF NOT EXISTS ix_sess_subject ON sessions(subject_id, date);
CREATE INDEX IF NOT EXISTS ix_risk_user ON risk_predictions(user_id, subject_id, generated_at);

-- Audit logs are immutable (append-only).
CREATE TRIGGER IF NOT EXISTS trg_att_audit_no_update BEFORE UPDATE ON attendance_audit
BEGIN SELECT RAISE(ABORT, 'attendance_audit is immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_att_audit_no_delete BEFORE DELETE ON attendance_audit
BEGIN SELECT RAISE(ABORT, 'attendance_audit is immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_admin_audit_no_update BEFORE UPDATE ON admin_audit
BEGIN SELECT RAISE(ABORT, 'admin_audit is immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_admin_audit_no_delete BEFORE DELETE ON admin_audit
BEGIN SELECT RAISE(ABORT, 'admin_audit is immutable'); END;
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    db_path = config.path("database")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=15, detect_types=0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    return con


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    """Connection inside a transaction; commits on success, rolls back on error."""
    con = connect()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with tx() as con:
        con.executescript(SCHEMA)


def rows(con: sqlite3.Connection, sql: str, params: tuple | list = ()) -> list[dict]:
    return [dict(r) for r in con.execute(sql, params).fetchall()]


def one(con: sqlite3.Connection, sql: str, params: tuple | list = ()) -> dict | None:
    r = con.execute(sql, params).fetchone()
    return dict(r) if r else None


# --------------------------------------------------------------------------- #
# Admin audit + notifications
# --------------------------------------------------------------------------- #

def log_admin_action(con: sqlite3.Connection, actor: Actor | None, action: str,
                     target: str = "", details: dict | str | None = None) -> None:
    if isinstance(details, dict):
        details = json.dumps(details, default=str)
    con.execute(
        "INSERT INTO admin_audit(actor_id, actor_name, action, target, details, at) VALUES (?,?,?,?,?,?)",
        (actor.id if actor else None, actor.username if actor else "system", action, target, details, now()),
    )
    config.log.info("AUDIT %s by %s target=%s %s", action, actor.username if actor else "system", target, details or "")


def notify_admins(con: sqlite3.Connection, kind: str, message: str, related_user_id: int | None = None) -> None:
    con.execute(
        "INSERT INTO notifications(kind, message, related_user_id, created_at) VALUES (?,?,?,?)",
        (kind, message, related_user_id, now()),
    )


def list_admin_audit(actor: Actor, limit: int = 500) -> list[dict]:
    require_admin(actor)
    with tx() as con:
        return rows(con, "SELECT * FROM admin_audit ORDER BY id DESC LIMIT ?", (limit,))


def list_notifications(actor: Actor, unread_only: bool = False) -> list[dict]:
    require_admin(actor)
    sql = "SELECT * FROM notifications"
    if unread_only:
        sql += " WHERE is_read = 0"
    with tx() as con:
        return rows(con, sql + " ORDER BY id DESC LIMIT 200")


def mark_notifications_read(actor: Actor) -> None:
    require_admin(actor)
    with tx() as con:
        con.execute("UPDATE notifications SET is_read = 1 WHERE is_read = 0")


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #

USER_PUBLIC_COLS = ("id, username, full_name, roll_no, email, role, is_active, failed_logins, "
                    "locked_until, must_change_password, created_at")


def admin_exists() -> bool:
    with tx() as con:
        return con.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1").fetchone() is not None


def get_user(actor: Actor, user_id: int) -> dict | None:
    require_self_or_admin(actor, user_id)
    with tx() as con:
        return one(con, f"SELECT {USER_PUBLIC_COLS} FROM users WHERE id=?", (user_id,))


def list_users(actor: Actor, role: str | None = None) -> list[dict]:
    require_admin(actor)
    sql = f"SELECT {USER_PUBLIC_COLS} FROM users"
    params: list = []
    if role:
        sql += " WHERE role=?"
        params.append(role)
    with tx() as con:
        return rows(con, sql + " ORDER BY role, full_name", params)


def set_user_active(actor: Actor, user_id: int, active: bool) -> None:
    require_admin(actor)
    if int(user_id) == actor.id and not active:
        raise PermissionDenied("You cannot deactivate your own account.")
    with tx() as con:
        con.execute("UPDATE users SET is_active=?, failed_logins=0, locked_until=NULL WHERE id=?",
                    (1 if active else 0, user_id))
        log_admin_action(con, actor, "approve_user" if active else "deactivate_user", f"user:{user_id}")


def delete_user(actor: Actor, user_id: int) -> None:
    """Hard-delete a user. Cascades remove embeddings, face enrollments,
    attendance, risk predictions and anomaly flags (DPDP: deletion on request)."""
    require_admin(actor)
    if int(user_id) == actor.id:
        raise PermissionDenied("You cannot delete your own account.")
    with tx() as con:
        u = one(con, "SELECT username, role FROM users WHERE id=?", (user_id,))
        if not u:
            return
        if u["role"] == ROLE_ADMIN:
            n_admins = con.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
            if n_admins <= 1:
                raise PermissionDenied("Cannot delete the last administrator.")
        n_emb = con.execute(
            "SELECT COUNT(*) FROM face_embeddings fe JOIN face_enrollments f ON f.id=fe.enrollment_id "
            "WHERE f.user_id=?", (user_id,)).fetchone()[0]
        con.execute("DELETE FROM users WHERE id=?", (user_id,))
        log_admin_action(con, actor, "delete_user", f"user:{user_id}",
                         {"username": u["username"], "embeddings_removed": n_emb})
    raw = config.path("raw_faces") / str(int(user_id))
    if raw.exists():
        shutil.rmtree(raw, ignore_errors=True)


def enrolled_subject_ids(con: sqlite3.Connection, user_id: int) -> set[int]:
    return {r[0] for r in con.execute("SELECT subject_id FROM enrollments WHERE user_id=?", (user_id,))}


# --------------------------------------------------------------------------- #
# Subjects & enrollments
# --------------------------------------------------------------------------- #

def list_subjects(actor: Actor) -> list[dict]:
    """Admins see all subjects; students see the subjects they are enrolled in."""
    actor = require_actor(actor)
    with tx() as con:
        if actor.is_admin:
            return rows(con, "SELECT * FROM subjects ORDER BY code")
        return rows(con, "SELECT s.* FROM subjects s JOIN enrollments e ON e.subject_id=s.id "
                         "WHERE e.user_id=? ORDER BY s.code", (actor.id,))


def create_subject(actor: Actor, code: str, name: str, planned_sessions: int | None = None) -> int:
    require_admin(actor)
    code, name = code.strip().upper(), name.strip()
    if not code or not name:
        raise ValueError("Subject code and name are required.")
    planned = int(planned_sessions or config.get("attendance.planned_sessions_per_subject", 40))
    with tx() as con:
        try:
            cur = con.execute("INSERT INTO subjects(code, name, planned_sessions) VALUES (?,?,?)",
                              (code, name, planned))
        except sqlite3.IntegrityError:
            raise ValueError(f"Subject code '{code}' already exists.") from None
        log_admin_action(con, actor, "create_subject", f"subject:{cur.lastrowid}", {"code": code, "name": name})
        return int(cur.lastrowid)


def update_subject(actor: Actor, subject_id: int, name: str, planned_sessions: int) -> None:
    require_admin(actor)
    with tx() as con:
        con.execute("UPDATE subjects SET name=?, planned_sessions=? WHERE id=?",
                    (name.strip(), int(planned_sessions), subject_id))
        log_admin_action(con, actor, "update_subject", f"subject:{subject_id}",
                         {"name": name, "planned_sessions": planned_sessions})


def delete_subject(actor: Actor, subject_id: int) -> None:
    require_admin(actor)
    with tx() as con:
        n = con.execute("SELECT COUNT(*) FROM sessions WHERE subject_id=?", (subject_id,)).fetchone()[0]
        if n:
            raise ValueError("This subject already has sessions and cannot be deleted.")
        con.execute("DELETE FROM subjects WHERE id=?", (subject_id,))
        log_admin_action(con, actor, "delete_subject", f"subject:{subject_id}")


def list_enrollments(actor: Actor, subject_id: int | None = None) -> list[dict]:
    require_admin(actor)
    sql = ("SELECT e.user_id, e.subject_id, u.full_name, u.roll_no, s.code FROM enrollments e "
           "JOIN users u ON u.id=e.user_id JOIN subjects s ON s.id=e.subject_id")
    params: list = []
    if subject_id:
        sql += " WHERE e.subject_id=?"
        params.append(subject_id)
    with tx() as con:
        return rows(con, sql + " ORDER BY s.code, u.roll_no", params)


def set_subject_enrollment(actor: Actor, subject_id: int, user_ids: list[int]) -> None:
    """Replace the student list of a subject."""
    require_admin(actor)
    with tx() as con:
        before = {r[0] for r in con.execute("SELECT user_id FROM enrollments WHERE subject_id=?", (subject_id,))}
        after = {int(u) for u in user_ids}
        for uid in after - before:
            con.execute("INSERT OR IGNORE INTO enrollments(user_id, subject_id) VALUES (?,?)", (uid, subject_id))
        for uid in before - after:
            con.execute("DELETE FROM enrollments WHERE user_id=? AND subject_id=?", (uid, subject_id))
        if after != before:
            log_admin_action(con, actor, "update_enrollment", f"subject:{subject_id}",
                             {"added": sorted(after - before), "removed": sorted(before - after)})


# --------------------------------------------------------------------------- #
# Face enrollment
# --------------------------------------------------------------------------- #

def emb_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def face_enrollment_status(actor: Actor, user_id: int) -> dict | None:
    """Latest face enrollment record of a user."""
    require_self_or_admin(actor, user_id)
    with tx() as con:
        return one(con, "SELECT * FROM face_enrollments WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,))


def can_start_enrollment(actor: Actor, user_id: int) -> tuple[bool, str]:
    """A first enrollment is always allowed. Re-enrollment needs admin approval
    of the new capture, so it is allowed but goes back to the pending queue.
    While a capture is pending, a new one is not allowed."""
    latest = face_enrollment_status(actor, user_id)
    if latest is None:
        return True, "first"
    if latest["status"] == "pending":
        return False, "Your face enrollment is awaiting administrator review."
    if latest["status"] == "blocked":
        return False, "Your enrollment was blocked by the duplicate-face check. Contact the administrator."
    return True, "re-enroll" if latest["status"] == "approved" else "retry"


def all_embeddings(con: sqlite3.Connection, statuses: tuple[str, ...] = ("approved",),
                   exclude_user: int | None = None, students_only: bool = True) -> list[tuple[int, np.ndarray]]:
    """Enrolled embeddings. Only student accounts are used for kiosk matching and
    the duplicate check: admins never appear on a class roster."""
    q = ",".join("?" * len(statuses))
    sql = (f"SELECT f.user_id, fe.embedding FROM face_embeddings fe "
           f"JOIN face_enrollments f ON f.id=fe.enrollment_id "
           f"JOIN users u ON u.id=f.user_id WHERE f.status IN ({q})")
    params: list = list(statuses)
    if students_only:
        sql += " AND u.role='student'"
    if exclude_user is not None:
        sql += " AND f.user_id != ?"
        params.append(exclude_user)
    return [(r[0], blob_to_emb(r[1])) for r in con.execute(sql, params)]


def save_face_enrollment(actor: Actor, user_id: int, embeddings: list[np.ndarray], consent_at: str,
                         model_name: str, quality_summary: dict, duplicate_of: int | None,
                         duplicate_distance: float | None) -> tuple[int, str]:
    """Store a new enrollment (embeddings only). Returns (enrollment_id, status)."""
    require_self_or_admin(actor, user_id)
    ok, msg = can_start_enrollment(actor, user_id)
    if not ok:
        raise PermissionDenied(msg)
    status = "blocked" if duplicate_of is not None else "pending"
    with tx() as con:
        cur = con.execute(
            "INSERT INTO face_enrollments(user_id, status, consent_at, created_at, frame_count, quality_summary) "
            "VALUES (?,?,?,?,?,?)",
            (user_id, status, consent_at, now(), len(embeddings), json.dumps(quality_summary)))
        enr_id = int(cur.lastrowid)
        con.executemany("INSERT INTO face_embeddings(enrollment_id, embedding, model_name) VALUES (?,?,?)",
                        [(enr_id, emb_to_blob(e), model_name) for e in embeddings])
        u = one(con, "SELECT username, full_name, roll_no FROM users WHERE id=?", (user_id,))
        who = f"{u['full_name']} ({u['roll_no'] or u['username']})"
        if duplicate_of is not None:
            d = one(con, "SELECT full_name, roll_no, username FROM users WHERE id=?", (duplicate_of,))
            other = f"{d['full_name']} ({d['roll_no'] or d['username']})" if d else f"user {duplicate_of}"
            notify_admins(con, "duplicate_face",
                          f"Duplicate face blocked: enrollment by {who} matches existing user {other} "
                          f"(distance {duplicate_distance:.3f}).", user_id)
            config.log.warning("Duplicate face: user %s matches user %s (%.3f)", user_id, duplicate_of,
                               duplicate_distance or -1)
        else:
            notify_admins(con, "face_pending", f"New face enrollment awaiting approval: {who}.", user_id)
    return enr_id, status


def list_face_enrollments(actor: Actor, status: str | None = None) -> list[dict]:
    require_admin(actor)
    sql = ("SELECT f.*, u.full_name, u.username, u.roll_no FROM face_enrollments f "
           "JOIN users u ON u.id=f.user_id")
    params: list = []
    if status:
        sql += " WHERE f.status=?"
        params.append(status)
    with tx() as con:
        return rows(con, sql + " ORDER BY f.id DESC", params)


def review_face_enrollment(actor: Actor, enrollment_id: int, approve: bool, note: str = "") -> None:
    """Approve or reject a pending (or blocked) enrollment. Approving a
    re-enrollment supersedes the user's previously approved one."""
    require_admin(actor)
    with tx() as con:
        f = one(con, "SELECT * FROM face_enrollments WHERE id=?", (enrollment_id,))
        if not f or f["status"] not in ("pending", "blocked"):
            raise ValueError("Enrollment is not awaiting review.")
        if approve:
            con.execute("UPDATE face_enrollments SET status='superseded' WHERE user_id=? AND status='approved'",
                        (f["user_id"],))
        con.execute("UPDATE face_enrollments SET status=?, reviewed_by=?, reviewed_at=?, review_note=? WHERE id=?",
                    ("approved" if approve else "rejected", actor.id, now(), note, enrollment_id))
        if not approve:
            # Embeddings of rejected captures are not kept (minimal retention).
            con.execute("DELETE FROM face_embeddings WHERE enrollment_id=?", (enrollment_id,))
        # Superseded embeddings are no longer needed either.
        con.execute("DELETE FROM face_embeddings WHERE enrollment_id IN "
                    "(SELECT id FROM face_enrollments WHERE user_id=? AND status='superseded')", (f["user_id"],))
        log_admin_action(con, actor, "approve_face" if approve else "reject_face",
                         f"face_enrollment:{enrollment_id}", {"user_id": f["user_id"], "note": note})


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

def get_settings(actor: Actor) -> dict[str, str]:
    require_admin(actor)
    with tx() as con:
        return {r["key"]: r["value"] for r in rows(con, "SELECT key, value FROM settings")}


def set_setting(actor: Actor, key: str, value) -> None:
    require_admin(actor)
    if key not in config.EDITABLE_SETTINGS:
        raise ValueError(f"'{key}' is not an editable setting.")
    typ, lo, hi, _ = config.EDITABLE_SETTINGS[key]
    v = typ(value)
    if not (lo <= v <= hi):
        raise ValueError(f"{key} must be between {lo} and {hi}.")
    with tx() as con:
        old = one(con, "SELECT value FROM settings WHERE key=?", (key,))
        con.execute("INSERT INTO settings(key, value, updated_by, updated_at) VALUES (?,?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_by=excluded.updated_by, "
                    "updated_at=excluded.updated_at", (key, str(v), actor.id, now()))
        log_admin_action(con, actor, "change_setting", key,
                         {"old": old["value"] if old else config._yaml_get(key), "new": v})


def students_with_roles(actor: Actor) -> list[dict]:
    """Active students (for pick lists)."""
    require_admin(actor)
    with tx() as con:
        return rows(con, "SELECT id, full_name, roll_no, username FROM users "
                         "WHERE role=? AND is_active=1 ORDER BY roll_no, full_name", (ROLE_STUDENT,))
