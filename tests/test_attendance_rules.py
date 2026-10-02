"""Attendance rules at the grace / cutoff boundaries, auto-absent,
duplicate-mark prevention and audited corrections (Section 12)."""
from datetime import date, datetime, timedelta

import pytest

from core import attendance as A
from core import db

START = datetime(2026, 9, 1, 9, 0)


@pytest.mark.parametrize("offset_s, expected", [
    (-300, "present"),                 # early
    (0, "present"),                    # exactly at start
    (10 * 60, "present"),              # exactly start + grace  -> Present (<=)
    (10 * 60 + 1, "late"),             # one second after grace
    (30 * 60, "late"),                 # exactly the cutoff     -> Late (<=)
    (30 * 60 + 1, None),               # after cutoff -> window closed (Absent on close)
])
def test_classification_boundaries(offset_s, expected):
    assert A.classify_check_in(START + timedelta(seconds=offset_s), START, 10, 30) == expected


def test_zero_grace():
    assert A.classify_check_in(START, START, 0, 15) == "present"
    assert A.classify_check_in(START + timedelta(seconds=1), START, 0, 15) == "late"


def test_attendance_percentage_late_weight():
    assert A.attendance_percentage(6, 2, 10, late_weight=1.0) == pytest.approx(80.0)
    assert A.attendance_percentage(6, 2, 10, late_weight=0.5) == pytest.approx(70.0)
    assert A.attendance_percentage(0, 0, 0) == 0.0


def test_classes_needed():
    assert A.classes_needed(6, 10, 75) == 6      # (6+6)/(10+6) = 75%
    assert A.classes_needed(8, 10, 75) == 0
    assert A.classes_needed(0, 0, 75) == 0
    k = A.classes_needed(5, 12, 75)
    assert (5 + k) / (12 + k) >= 0.75 and (5 + k - 1) / (12 + k - 1) < 0.75


def _setup_session(env, start="09:00"):
    admin = env["admin"]
    sid = db.create_subject(admin, "CS701", "Machine Learning")
    db.set_subject_enrollment(admin, sid, [s.id for s in env["students"]])
    sess = A.create_session(admin, sid, date(2026, 9, 1), start)
    A.open_session(admin, sess)
    return sid, sess


def test_kiosk_marking_and_auto_absent(env):
    admin, (s1, s2, s3) = env["admin"], env["students"]
    _, sess = _setup_session(env)
    r1 = A.mark_from_kiosk(admin, sess, s1.id, 0.3, 0.95, 0, check_in=START + timedelta(minutes=5))
    r2 = A.mark_from_kiosk(admin, sess, s2.id, 0.3, 0.95, 1, check_in=START + timedelta(minutes=20))
    r3 = A.mark_from_kiosk(admin, sess, s3.id, 0.3, 0.95, 0, check_in=START + timedelta(minutes=45))
    assert (r1.status, r2.status) == ("present", "late")
    assert not r3.ok and r3.code == "window_closed"
    created = A.close_session(admin, sess)
    assert created == 1
    recs = {r["user_id"]: r for r in A.get_records(admin, session_id=sess)}
    assert recs[s3.id]["status"] == "absent" and recs[s3.id]["source"] == "system"
    assert recs[s2.id]["failed_attempts"] == 1


def test_duplicate_mark_prevented(env):
    admin, s1 = env["admin"], env["students"][0]
    _, sess = _setup_session(env)
    assert A.mark_from_kiosk(admin, sess, s1.id, 0.3, 0.9, 0, check_in=START).ok
    again = A.mark_from_kiosk(admin, sess, s1.id, 0.3, 0.9, 0, check_in=START + timedelta(minutes=1))
    assert not again.ok and again.code == "already"
    with db.tx() as con:
        n = con.execute("SELECT COUNT(*) FROM attendance WHERE session_id=? AND user_id=?", (sess, s1.id)).fetchone()[0]
        assert n == 1
        with pytest.raises(Exception):  # enforced by UNIQUE(session_id, user_id)
            con.execute("INSERT INTO attendance(session_id, user_id, status, source, created_at) "
                        "VALUES (?,?,'present','manual','x')", (sess, s1.id))


def test_kiosk_rejects_unenrolled_and_closed(env):
    admin, s1 = env["admin"], env["students"][0]
    sid = db.create_subject(admin, "CS702", "Compiler Design")
    sess = A.create_session(admin, sid, date(2026, 9, 1), "09:00")
    assert A.mark_from_kiosk(admin, sess, s1.id, 0.3, 0.9, 0, check_in=START).code == "session_not_open"
    A.open_session(admin, sess)
    assert A.mark_from_kiosk(admin, sess, s1.id, 0.3, 0.9, 0, check_in=START).code == "not_enrolled"


def test_absent_only_from_close(env):
    """Recognition can produce only Present or Late."""
    for off in range(-60, 31):
        assert A.classify_check_in(START + timedelta(minutes=off), START, 10, 30) in ("present", "late")


def test_correction_requires_reason_and_is_audited(env):
    admin, s1 = env["admin"], env["students"][0]
    _, sess = _setup_session(env)
    A.close_session(admin, sess)
    with pytest.raises(ValueError):
        A.correct_attendance(admin, sess, s1.id, "present", "   ")
    A.correct_attendance(admin, sess, s1.id, "present", "Recognition failed; verified in class")
    log = A.list_corrections(admin)
    assert log[0]["old_status"] == "absent" and log[0]["new_status"] == "present"
    assert log[0]["changed_by"] == admin.id and log[0]["reason"].startswith("Recognition")


def test_audit_log_is_immutable(env):
    admin, s1 = env["admin"], env["students"][0]
    _, sess = _setup_session(env)
    A.close_session(admin, sess)
    A.correct_attendance(admin, sess, s1.id, "late", "Arrived late, camera issue")
    with db.tx() as con, pytest.raises(Exception, match="immutable"):
        con.execute("UPDATE attendance_audit SET reason='tampered'")
    with db.tx() as con, pytest.raises(Exception, match="immutable"):
        con.execute("DELETE FROM attendance_audit")
    with db.tx() as con, pytest.raises(Exception, match="immutable"):
        con.execute("DELETE FROM admin_audit")
