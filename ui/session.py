"""Login session handling for the Streamlit UI: current actor, inactivity
timeout and kiosk lock. Authorisation itself is enforced in the data layer."""
from __future__ import annotations

import time

import streamlit as st

from core import auth, config
from core.access import Actor, PermissionDenied


def init_state() -> None:
    defaults = {"actor": None, "last_activity": time.time(), "must_change_password": False,
                "kiosk_session_id": None, "flash": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def actor() -> Actor | None:
    return st.session_state.get("actor")


def require_actor() -> Actor:
    a = actor()
    if a is None:
        st.error("Please log in.")
        st.stop()
    return a


def require_admin() -> Actor:
    a = require_actor()
    if not a.is_admin:
        st.error("Administrator privileges required.")
        st.stop()
    return a


def login(a: Actor, must_change: bool) -> None:
    st.session_state.actor = a
    st.session_state.must_change_password = must_change
    st.session_state.last_activity = time.time()


def logout(message: str | None = None) -> None:
    for k in list(st.session_state.keys()):
        if k not in ("flash",):
            del st.session_state[k]
    init_state()
    st.session_state.flash = message


def enforce_timeout_and_refresh() -> None:
    """Log out after inactivity; re-validate the user against the database so
    deactivation / deletion takes effect immediately."""
    a = actor()
    if a is None:
        return
    if st.session_state.kiosk_session_id is None:
        timeout = int(config.get("auth.session_timeout_minutes", 30)) * 60
        if time.time() - st.session_state.last_activity > timeout:
            logout("You were logged out after a period of inactivity.")
            return
    st.session_state.last_activity = time.time()
    fresh = auth.refresh_actor(a)
    if fresh is None:
        logout("Your account is no longer active.")
    else:
        st.session_state.actor = fresh


def flash(msg: str, kind: str = "success") -> None:
    st.session_state.flash = (kind, msg)


def show_flash() -> None:
    f = st.session_state.get("flash")
    if not f:
        return
    kind, msg = f if isinstance(f, tuple) else ("info", f)
    getattr(st, kind, st.info)(msg)
    st.session_state.flash = None


def guarded(fn, *args, success: str | None = None, rerun: bool = True, **kwargs):
    """Run a data-layer call, report the outcome as a flash message (which
    survives the rerun) and rerun the page unless ``rerun`` is False."""
    try:
        out = fn(*args, **kwargs)
    except PermissionDenied as e:
        flash(f"Access denied: {e}", "error")
        out = None
    except (ValueError, auth.AuthError) as e:
        flash(str(e), "error")
        out = None
    else:
        if success:
            flash(success)
        out = out if out is not None else True
    if rerun:
        st.rerun()
    return out
