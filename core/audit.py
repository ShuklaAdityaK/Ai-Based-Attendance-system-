"""Audit trail access.

Two append-only tables are protected by SQLite triggers that abort any
UPDATE or DELETE (see ``core.db.SCHEMA``):

* ``attendance_audit`` - every manual attendance change (old/new status,
  admin, reason, timestamp)
* ``admin_audit`` - every other administrative action (approvals,
  deletions, settings, sessions, backups, model retraining)
"""
from __future__ import annotations

import pandas as pd

from core import attendance, db
from core.access import Actor, require_admin

log_admin_action = db.log_admin_action


def attendance_audit_frame(actor: Actor) -> pd.DataFrame:
    require_admin(actor)
    df = pd.DataFrame(attendance.list_corrections(actor))
    if df.empty:
        return df
    return df[["changed_at", "changed_by_name", "full_name", "roll_no", "code", "date",
               "old_status", "new_status", "reason"]]


def admin_audit_frame(actor: Actor) -> pd.DataFrame:
    require_admin(actor)
    df = pd.DataFrame(db.list_admin_audit(actor))
    if df.empty:
        return df
    return df[["at", "actor_name", "action", "target", "details"]]
