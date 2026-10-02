"""Manual backup export / restore (Section 2.13).

A backup is one .zip containing:
    attendance.db      consistent snapshot via the SQLite online-backup API
    trained/           trained risk / anomaly models
    config.yaml        configuration
    manifest.json      app + schema info and file list
Pretrained ONNX models are not included (they are re-downloadable and large).
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from core import config, db
from core.access import Actor, require_admin

MANIFEST = "manifest.json"
APP_ID = "smart-attendance-backup"


def create_backup(actor: Actor, label: str = "manual") -> Path:
    require_admin(actor)
    out_dir = config.path("backups")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = out_dir / f"backup_{label}_{stamp}.zip"
    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp) / "attendance.db"
        src = db.connect()
        dst = sqlite3.connect(snap)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        files = ["attendance.db", "config.yaml"]
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(snap, "attendance.db")
            z.write(config.CONFIG_FILE, "config.yaml")
            trained = config.path("trained_models")
            if trained.exists():
                for f in trained.rglob("*"):
                    if f.is_file():
                        arc = f"trained/{f.relative_to(trained).as_posix()}"
                        z.write(f, arc)
                        files.append(arc)
            z.writestr(MANIFEST, json.dumps({"app": APP_ID, "created_at": db.now(), "created_by": actor.username,
                                             "files": files}, indent=2))
    with db.tx() as con:
        db.log_admin_action(con, actor, "backup_export", target.name)
    return target


def list_backups(actor: Actor) -> list[dict]:
    require_admin(actor)
    d = config.path("backups")
    if not d.exists():
        return []
    return [{"file": p.name, "size_kb": round(p.stat().st_size / 1024, 1),
             "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"), "path": str(p)}
            for p in sorted(d.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)]


def validate_backup(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        if MANIFEST not in names or "attendance.db" not in names:
            raise ValueError("Not a valid Smart Attendance backup (manifest or database missing).")
        for n in names:
            if n.startswith("/") or ".." in Path(n).parts:
                raise ValueError("Backup contains unsafe paths.")
        manifest = json.loads(z.read(MANIFEST))
        if manifest.get("app") != APP_ID:
            raise ValueError("Backup was not produced by this application.")
        with tempfile.TemporaryDirectory() as tmp:
            z.extract("attendance.db", tmp)
            con = sqlite3.connect(Path(tmp) / "attendance.db")
            try:
                if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("Database in the backup failed the integrity check.")
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            finally:
                con.close()
        missing = {"users", "sessions", "attendance", "face_embeddings"} - tables
        if missing:
            raise ValueError(f"Backup database is missing tables: {', '.join(sorted(missing))}")
    return manifest


def restore_backup(actor: Actor, zip_path: Path) -> Path:
    """Restore from a backup. A safety backup of the current state is taken
    first and its path returned."""
    require_admin(actor)
    manifest = validate_backup(zip_path)
    safety = create_backup(actor, label="pre-restore")
    db_path = config.path("database")
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp)
        restored = sqlite3.connect(Path(tmp) / "attendance.db")
        live = sqlite3.connect(db_path)
        try:
            restored.backup(live)            # page-level copy into the live file (handles WAL)
        finally:
            live.close()
            restored.close()
        trained = config.path("trained_models")
        if (Path(tmp) / "trained").exists():
            shutil.rmtree(trained, ignore_errors=True)
            shutil.copytree(Path(tmp) / "trained", trained)
        if (Path(tmp) / "config.yaml").exists():
            shutil.copy2(Path(tmp) / "config.yaml", config.CONFIG_FILE)
            config.reload()
    db.init_db()  # make sure schema objects of this version exist
    with db.tx() as con:
        db.log_admin_action(con, actor, "backup_restore", zip_path.name,
                            {"backup_created_at": manifest.get("created_at"), "safety_backup": safety.name})
    return safety
