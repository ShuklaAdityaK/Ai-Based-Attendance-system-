from datetime import datetime, timedelta

import pytest

from core import auth, config, db


def test_passwords_are_bcrypt_hashed(env):
    with db.tx() as con:
        row = con.execute("SELECT password_hash FROM users WHERE username='stu1'").fetchone()
    assert row[0].startswith("$2b$") and "Passw0rd1" not in row[0]


def test_no_second_first_admin(env):
    with pytest.raises(auth.AuthError):
        auth.create_first_admin("admin2", "Another", "", "Admin1234")


def test_registration_inactive_until_approved(env):
    uid = auth.register_student("newbie", "New Student", "R999", "", "Passw0rd1")
    r = auth.login("newbie", "Passw0rd1")
    assert not r.ok and "approval" in r.message
    db.set_user_active(env["admin"], uid, True)
    assert auth.login("newbie", "Passw0rd1").ok


def test_lockout_after_failed_attempts(env):
    n = int(config.get("auth.max_failed_logins"))
    for _ in range(n - 1):
        assert not auth.login("stu1", "wrong").ok
    r = auth.login("stu1", "wrong")
    assert "locked" in r.message.lower()
    # Even the correct password is refused while locked.
    r = auth.login("stu1", "Passw0rd1")
    assert not r.ok and "locked" in r.message.lower()
    # After the lock expires, login works again.
    with db.tx() as con:
        con.execute("UPDATE users SET locked_until=? WHERE username='stu1'",
                    ((datetime.now() - timedelta(minutes=1)).isoformat(),))
    assert auth.login("stu1", "Passw0rd1").ok


def test_lockout_is_configurable(env):
    db.set_setting(env["admin"], "auth.max_failed_logins", 2)
    assert not auth.login("stu2", "bad").ok
    assert "locked" in auth.login("stu2", "bad").message.lower()


def test_password_policy():
    for bad in ["short1", "lettersonly", "12345678"]:
        with pytest.raises(auth.AuthError):
            auth.validate_password(bad)


def test_admin_reset_forces_change(env):
    s1 = env["students"][0]
    temp = auth.reset_password(env["admin"], s1.id)
    r = auth.login("stu1", temp)
    assert r.ok and r.must_change_password
    auth.change_password(r.actor, temp, "NewPassw0rd")
    assert not auth.login("stu1", "NewPassw0rd").must_change_password
