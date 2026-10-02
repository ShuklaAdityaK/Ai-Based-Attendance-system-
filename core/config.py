"""Configuration loading.

Defaults come from ``config.yaml``. A subset of keys is admin-editable; those
are overridden by rows in the ``settings`` table (see ``EDITABLE_SETTINGS``).
"""
from __future__ import annotations

import logging
import os
import sqlite3
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(os.environ.get("SA_CONFIG", ROOT / "config.yaml"))

# key -> (type, min, max, label)
EDITABLE_SETTINGS: dict[str, tuple[type, float, float, str]] = {
    "auth.max_failed_logins": (int, 1, 20, "Failed logins before lockout"),
    "auth.lockout_minutes": (int, 1, 1440, "Lockout duration (minutes)"),
    "auth.session_timeout_minutes": (int, 1, 480, "Inactivity timeout (minutes)"),
    "attendance.grace_minutes": (int, 0, 120, "Grace period - Present (minutes)"),
    "attendance.late_cutoff_minutes": (int, 1, 240, "Late cutoff (minutes)"),
    "attendance.late_weight": (float, 0.0, 1.0, "Weight of a Late mark"),
    "attendance.required_percentage": (float, 1, 100, "Required attendance (%)"),
    "attendance.planned_sessions_per_subject": (int, 1, 200, "Default planned sessions per subject"),
    "recognition.match_threshold": (float, 0.05, 1.5, "Recognition threshold (cosine distance)"),
    "recognition.duplicate_threshold": (float, 0.05, 1.5, "Duplicate-face threshold (cosine distance)"),
    "liveness.passive_threshold": (float, 0.05, 0.99, "Passive liveness threshold (real prob.)"),
    "liveness.challenge_timeout_seconds": (int, 3, 60, "Active challenge time limit (s)"),
    "ml.anomaly_contamination": (float, 0.01, 0.4, "Isolation Forest contamination"),
}

_yaml_cache: dict | None = None


def _load_yaml() -> dict:
    global _yaml_cache
    if _yaml_cache is None:
        with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
            _yaml_cache = yaml.safe_load(fh) or {}
    return _yaml_cache


def reload() -> None:
    global _yaml_cache
    _yaml_cache = None


def _yaml_get(key: str, default: Any = None) -> Any:
    node: Any = _load_yaml()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def path(key: str) -> Path:
    """Resolve a ``paths.*`` entry to an absolute path."""
    if key == "database" and os.environ.get("SA_DB_PATH"):
        return Path(os.environ["SA_DB_PATH"])
    p = Path(_yaml_get(f"paths.{key}"))
    return p if p.is_absolute() else ROOT / p


def _db_override(key: str) -> str | None:
    db = path("database")
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(db, timeout=5)
        try:
            row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def get(key: str, default: Any = None) -> Any:
    """Return a setting; admin-editable keys honour the ``settings`` table."""
    if key in EDITABLE_SETTINGS:
        raw = _db_override(key)
        if raw is not None:
            typ = EDITABLE_SETTINGS[key][0]
            try:
                return typ(float(raw)) if typ is int else typ(raw)
            except ValueError:
                pass
    return _yaml_get(key, default)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("smart_attendance")
    if logger.handlers:
        return logger
    log_file = path("log_file")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


log = setup_logging()
