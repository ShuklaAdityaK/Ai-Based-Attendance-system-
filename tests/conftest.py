import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import auth, config, db  # noqa: E402
from core.access import Actor  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated database + output folders for each test."""
    monkeypatch.setenv("SA_DB_PATH", str(tmp_path / "test.db"))
    real_path = config.path

    def fake_path(key):
        if key in ("trained_models", "reports", "backups", "raw_faces"):
            return tmp_path / key
        return real_path(key)

    monkeypatch.setattr(config, "path", fake_path)
    cfg_copy = tmp_path / "config.yaml"
    cfg_copy.write_bytes(config.CONFIG_FILE.read_bytes())
    monkeypatch.setattr(config, "CONFIG_FILE", cfg_copy)
    db.init_db()
    admin_id = auth.create_first_admin("admin", "Admin User", "admin@example.com", "Admin1234")
    admin = Actor(admin_id, "admin", "admin", "Admin User")
    students = []
    for i in range(1, 4):
        uid = auth.register_student(f"stu{i}", f"Student {i}", f"R{i:03d}", f"s{i}@example.com", "Passw0rd1")
        db.set_user_active(admin, uid, True)
        students.append(Actor(uid, f"stu{i}", "student", f"Student {i}"))
    return {"admin": admin, "students": students, "tmp": tmp_path}
