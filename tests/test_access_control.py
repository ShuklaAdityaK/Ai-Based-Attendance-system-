"""A student attempting admin and other-user routes must be blocked in the
data layer (Section 12, access control)."""
from datetime import date

import pytest

from analytics import data as an
from core import attendance as A
from core import auth, backup, db
from core.access import Actor, PermissionDenied
from ml import anomaly, risk_model
from reports import excel_report, pdf_report
from reports.common import ReportFilters


def test_student_blocked_from_admin_functions(env):
    stu, other = env["students"][0], env["students"][1]
    admin_calls = [
        lambda: db.list_users(stu),
        lambda: db.set_user_active(stu, other.id, False),
        lambda: db.delete_user(stu, other.id),
        lambda: db.create_subject(stu, "X1", "Hack"),
        lambda: db.list_face_enrollments(stu),
        lambda: db.review_face_enrollment(stu, 1, True),
        lambda: db.set_setting(stu, "attendance.grace_minutes", 60),
        lambda: db.list_admin_audit(stu),
        lambda: A.create_session(stu, 1, date.today(), "09:00"),
        lambda: A.open_session(stu, 1),
        lambda: A.close_session(stu, 1),
        lambda: A.correct_attendance(stu, 1, stu.id, "present", "self-service"),
        lambda: A.mark_from_kiosk(stu, 1, stu.id, 0.1, 0.99, 0),
        lambda: A.list_corrections(stu),
        lambda: auth.reset_password(stu, other.id),
        lambda: auth.create_user_by_admin(stu, "evil", "Evil", None, "", "Passw0rd1", "admin"),
        lambda: anomaly.run_detection(stu),
        lambda: anomaly.list_flags(stu),
        lambda: anomaly.review_flag(stu, 1, "dismissed"),
        lambda: risk_model.train(stu),
        lambda: risk_model.refresh_predictions(stu),
        lambda: backup.create_backup(stu),
        lambda: backup.list_backups(stu),
        lambda: A.session_roster(stu, 1),
    ]
    for call in admin_calls:
        with pytest.raises(PermissionDenied):
            call()


def test_student_cannot_read_other_users_data(env):
    stu, other = env["students"][0], env["students"][1]
    other_user_calls = [
        lambda: db.get_user(stu, other.id),
        lambda: db.face_enrollment_status(stu, other.id),
        lambda: A.get_records(stu, user_id=other.id),
        lambda: an.records_frame(stu, user_id=other.id),
        lambda: risk_model.stored_predictions(stu, other.id),
        lambda: risk_model.predict_frame(stu, other.id),
        lambda: excel_report.generate(stu, ReportFilters(user_id=other.id)),
        lambda: pdf_report.generate(stu, ReportFilters(user_id=other.id)),
    ]
    for call in other_user_calls:
        with pytest.raises(PermissionDenied):
            call()


def test_student_queries_are_pinned_to_self(env):
    admin, (s1, s2, _) = env["admin"], env["students"]
    sid = db.create_subject(admin, "CS703", "Data Mining")
    db.set_subject_enrollment(admin, sid, [s1.id, s2.id])
    sess = A.create_session(admin, sid, date(2026, 9, 2), "10:00")
    A.open_session(admin, sess)
    A.close_session(admin, sess)
    # Passing no user_id must NOT return everyone's records for a student.
    recs = A.get_records(s1)
    assert recs and all(r["user_id"] == s1.id for r in recs)
    assert len(A.get_records(admin)) == 2


def test_unauthenticated_blocked(env):
    with pytest.raises(PermissionDenied):
        A.get_records(None)
    with pytest.raises(PermissionDenied):
        db.list_subjects(None)


def test_forged_actor_role_does_not_help_with_deleted_user(env):
    """Sessions are re-validated against the database (refresh_actor)."""
    admin, s1 = env["admin"], env["students"][0]
    db.delete_user(admin, s1.id)
    assert auth.refresh_actor(s1) is None
    assert auth.refresh_actor(Actor(s1.id, "stu1", "admin")) is None
