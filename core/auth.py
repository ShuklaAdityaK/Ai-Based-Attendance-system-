"""Authentication: bcrypt hashing, registration, first-run admin setup,
login lockout and password management."""
from __future__ import annotations

import re
import secrets
import sqlite3
import string
from dataclasses import dataclass
from datetime import datetime, timedelta

import bcrypt

from core import config, db
from core.access import ROLE_ADMIN, ROLE_STUDENT, Actor, PermissionDenied, require_admin, require_actor


class AuthError(Exception):
    pass


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


def validate_password(password: str) -> None:
    min_len = int(config.get("auth.min_password_length", 8))
    if len(password) < min_len:
        raise AuthError(f"Password must be at least {min_len} characters.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise AuthError("Password must contain both letters and digits.")


def _validate_username(username: str) -> str:
    username = username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
        raise AuthError("Username must be 3-32 characters: letters, digits, '_', '.', '-'.")
    return username


def _insert_user(con: sqlite3.Connection, username: str, full_name: str, roll_no: str | None,
                 email: str | None, password: str, role: str, active: bool) -> int:
    try:
        cur = con.execute(
            "INSERT INTO users(username, full_name, roll_no, email, password_hash, role, is_active, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (username, full_name.strip(), (roll_no or "").strip().upper() or None, (email or "").strip() or None,
             hash_password(password), role, 1 if active else 0, db.now()))
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        if "username" in msg:
            raise AuthError("That username is already taken.") from None
        if "roll_no" in msg:
            raise AuthError("That roll number is already registered.") from None
        raise
    return int(cur.lastrowid)


def create_first_admin(username: str, full_name: str, email: str, password: str) -> int:
    """First-run setup. Only allowed while no administrator exists."""
    username = _validate_username(username)
    validate_password(password)
    if not full_name.strip():
        raise AuthError("Full name is required.")
    with db.tx() as con:
        con.execute("BEGIN IMMEDIATE")
        if con.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():
            raise AuthError("An administrator already exists.")
        uid = _insert_user(con, username, full_name, None, email, password, ROLE_ADMIN, True)
        db.log_admin_action(con, Actor(uid, username, ROLE_ADMIN), "first_run_admin_created", f"user:{uid}")
    return uid


def register_student(username: str, full_name: str, roll_no: str, email: str, password: str) -> int:
    """Self-registration. The account stays inactive until an admin approves it."""
    username = _validate_username(username)
    validate_password(password)
    if not full_name.strip():
        raise AuthError("Full name is required.")
    if not roll_no.strip():
        raise AuthError("Roll number is required.")
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
        raise AuthError("Email address looks invalid.")
    with db.tx() as con:
        uid = _insert_user(con, username, full_name, roll_no, email, password, ROLE_STUDENT, False)
        db.notify_admins(con, "user_pending", f"New student registration awaiting approval: "
                                              f"{full_name.strip()} ({roll_no.strip().upper()}).", uid)
    config.log.info("Student registered: %s (pending approval)", username)
    return uid


def create_user_by_admin(actor: Actor, username: str, full_name: str, roll_no: str | None, email: str,
                         password: str, role: str) -> int:
    require_admin(actor)
    if role not in (ROLE_STUDENT, ROLE_ADMIN):
        raise AuthError("Invalid role.")
    username = _validate_username(username)
    validate_password(password)
    with db.tx() as con:
        uid = _insert_user(con, username, full_name, roll_no, email, password, role, True)
        con.execute("UPDATE users SET must_change_password=1 WHERE id=?", (uid,))
        db.log_admin_action(con, actor, "create_user", f"user:{uid}", {"username": username, "role": role})
    return uid


_DUMMY_HASH: str | None = None


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_hex(8))
    return _DUMMY_HASH


@dataclass
class LoginResult:
    ok: bool
    message: str
    actor: Actor | None = None
    must_change_password: bool = False


def login(username: str, password: str) -> LoginResult:
    max_fail = int(config.get("auth.max_failed_logins", 5))
    lock_min = int(config.get("auth.lockout_minutes", 15))
    generic = "Invalid username or password."
    with db.tx() as con:
        u = db.one(con, "SELECT * FROM users WHERE username=?", (username.strip(),))
        if not u:
            # Equalise timing a little so usernames are harder to enumerate.
            verify_password(password, _dummy_hash())
            return LoginResult(False, generic)
        now = datetime.now()
        if u["locked_until"] and datetime.fromisoformat(u["locked_until"]) > now:
            mins = int((datetime.fromisoformat(u["locked_until"]) - now).total_seconds() // 60) + 1
            return LoginResult(False, f"Account locked after repeated failed logins. Try again in {mins} min.")
        if not verify_password(password, u["password_hash"]):
            fails = int(u["failed_logins"]) + 1
            if fails >= max_fail:
                until = (now + timedelta(minutes=lock_min)).isoformat(timespec="seconds")
                con.execute("UPDATE users SET failed_logins=0, locked_until=? WHERE id=?", (until, u["id"]))
                config.log.warning("Account locked: %s", u["username"])
                return LoginResult(False, f"Too many failed attempts. Account locked for {lock_min} minutes.")
            con.execute("UPDATE users SET failed_logins=?, locked_until=NULL WHERE id=?", (fails, u["id"]))
            left = max_fail - fails
            return LoginResult(False, f"{generic} {left} attempt(s) left before lockout.")
        con.execute("UPDATE users SET failed_logins=0, locked_until=NULL WHERE id=?", (u["id"],))
        if not u["is_active"]:
            return LoginResult(False, "Your account is awaiting administrator approval or has been deactivated.")
        actor = Actor(u["id"], u["username"], u["role"], u["full_name"])
        config.log.info("Login: %s (%s)", u["username"], u["role"])
        return LoginResult(True, "Welcome!", actor, bool(u["must_change_password"]))


def change_password(actor: Actor, old_password: str, new_password: str) -> None:
    actor = require_actor(actor)
    validate_password(new_password)
    with db.tx() as con:
        u = db.one(con, "SELECT password_hash FROM users WHERE id=?", (actor.id,))
        if not u or not verify_password(old_password, u["password_hash"]):
            raise AuthError("Current password is incorrect.")
        if old_password == new_password:
            raise AuthError("New password must differ from the current one.")
        con.execute("UPDATE users SET password_hash=?, must_change_password=0 WHERE id=?",
                    (hash_password(new_password), actor.id))


def reset_password(actor: Actor, user_id: int) -> str:
    """Admin reset: sets a random temporary password the user must change."""
    require_admin(actor)
    alphabet = string.ascii_letters + string.digits
    while True:
        temp = "".join(secrets.choice(alphabet) for _ in range(10))
        if re.search(r"\d", temp) and re.search(r"[A-Za-z]", temp):
            break
    with db.tx() as con:
        if not con.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone():
            raise AuthError("User not found.")
        con.execute("UPDATE users SET password_hash=?, must_change_password=1, failed_logins=0, "
                    "locked_until=NULL WHERE id=?", (hash_password(temp), user_id))
        db.log_admin_action(con, actor, "reset_password", f"user:{user_id}")
    return temp


def verify_admin_password(actor: Actor, password: str) -> bool:
    """Used to unlock (exit) kiosk mode."""
    require_admin(actor)
    with db.tx() as con:
        u = db.one(con, "SELECT password_hash FROM users WHERE id=?", (actor.id,))
    return bool(u) and verify_password(password, u["password_hash"])


def refresh_actor(actor: Actor) -> Actor | None:
    """Re-read the user so deactivation/deletion takes effect immediately."""
    with db.tx() as con:
        u = db.one(con, "SELECT id, username, role, full_name, is_active FROM users WHERE id=?", (actor.id,))
    if not u or not u["is_active"]:
        return None
    return Actor(u["id"], u["username"], u["role"], u["full_name"])


__all__ = ["AuthError", "LoginResult", "PermissionDenied", "login", "register_student", "create_first_admin"]
