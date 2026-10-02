"""Liveness rules and the kiosk state machine.

The neural networks are replaced by small fakes so the tests are fast and
deterministic; everything else (challenge logic, kiosk states, gallery,
attendance rules, database) is the real code.
"""
from datetime import date, datetime, timedelta
from itertools import cycle

import av
import numpy as np
import pytest

from core import attendance as A
from core import config, db
from vision import kiosk as K
from vision.detector import Face
from vision.liveness import ActiveChallenge, MeshMetrics, passive_is_real
from vision.recognizer import Gallery

REAL = np.array([[0.05, 0.90, 0.05], [0.10, 0.85, 0.05]])
SPOOF = np.array([[0.70, 0.20, 0.10], [0.60, 0.30, 0.10]])
OPEN, CLOSED = 0.30, 0.12


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------- passive -- #

def test_passive_rule_requires_real_as_top_class_and_minimum():
    assert passive_is_real(np.array([0.1, 0.8, 0.1]), 0.45)
    assert not passive_is_real(np.array([0.5, 0.4, 0.1]), 0.3)    # real is not the top class
    assert not passive_is_real(np.array([0.3, 0.4, 0.3]), 0.45)   # top class but below the minimum


# ------------------------------------------------------------- challenge -- #

def _feed(ch: ActiveChallenge, metrics: list[MeshMetrics]) -> bool:
    for m in metrics:
        if ch.update(m):
            return True
    return False


def blink_seq(n: int, ear_closed: float = CLOSED, blink_score: float = 0.0) -> list[MeshMetrics]:
    one = [MeshMetrics(OPEN, 0.5)] * 3 + [MeshMetrics(ear_closed, 0.5, blink_score)] + [MeshMetrics(OPEN, 0.5)] * 2
    return one * n


def test_two_blinks_complete_challenge():
    assert _feed(ActiveChallenge("blink"), blink_seq(2))


def test_one_blink_is_not_enough():
    ch = ActiveChallenge("blink")
    assert not _feed(ch, blink_seq(1) + [MeshMetrics(OPEN, 0.5)] * 20)
    assert ch.blinks == 1


def test_blink_detected_by_blendshape_when_ear_barely_moves():
    # Low-res webcams: EAR hardly drops, but MediaPipe's blink score does.
    assert _feed(ActiveChallenge("blink"), blink_seq(2, ear_closed=0.28, blink_score=0.9))


def test_static_photo_never_blinks():
    assert not _feed(ActiveChallenge("blink"), [MeshMetrics(OPEN, 0.5)] * 100)


def test_turn_left_from_frontal():
    seq = [MeshMetrics(OPEN, 0.5)] * 3 + [MeshMetrics(OPEN, 0.8)] * 3
    assert _feed(ActiveChallenge("turn_left"), seq)


def test_turn_requires_frontal_start():
    # A face that is already turned (e.g. an angled photo) does not pass.
    assert not _feed(ActiveChallenge("turn_left"), [MeshMetrics(OPEN, 0.8)] * 30)


def test_turn_wrong_direction_fails():
    seq = [MeshMetrics(OPEN, 0.5)] * 3 + [MeshMetrics(OPEN, 0.2)] * 10
    assert not _feed(ActiveChallenge("turn_left"), seq)
    assert _feed(ActiveChallenge("turn_right"), seq)


def test_challenge_expires(monkeypatch):
    ch = ActiveChallenge("blink")
    ch.started -= float(config.get("liveness.challenge_timeout_seconds")) + 1
    assert ch.expired()
    assert not _feed(ch, blink_seq(3))


# ---------------------------------------------------------------- kiosk --- #

class FakeDetector:
    faces: list = []

    def detect(self, img):
        return list(FakeDetector.faces)


class FakeEmbedder:
    vec = unit(1)
    model_name = "fake"

    def embed(self, img, landmarks):
        return FakeEmbedder.vec


class FakePassive:
    probs = REAL

    def model_probabilities(self, img, box):
        return FakePassive.probs


class FakeMesh:
    seq = cycle(blink_seq(1))

    def metrics(self, img):
        return next(FakeMesh.seq)

    def close(self):
        pass


FACE = Face((200, 150, 160, 160), np.zeros((5, 2), np.float32), 0.99)
FRAME = av.VideoFrame.from_ndarray(np.full((480, 640, 3), 120, np.uint8), format="bgr24")


@pytest.fixture()
def kiosk(env, monkeypatch):
    monkeypatch.setattr(K, "FaceDetector", FakeDetector)
    monkeypatch.setattr(K, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(K, "get_passive", FakePassive)
    monkeypatch.setattr(K, "FaceMesh", FakeMesh)
    monkeypatch.setattr(K.ActiveChallenge, "random", classmethod(lambda cls: cls("blink")))
    FakeDetector.faces, FakeEmbedder.vec, FakePassive.probs = [FACE], unit(1), REAL
    FakeMesh.seq = cycle(blink_seq(1))

    admin, (s1, s2, _) = env["admin"], env["students"]
    sid = db.create_subject(admin, "CS705", "Kiosk Test")
    db.set_subject_enrollment(admin, sid, [s1.id])
    for stu, seed in ((s1, 1), (s2, 2)):
        enr, _ = db.save_face_enrollment(stu, stu.id, [unit(seed)] * 5, db.now(), "fake", {}, None, None)
        db.review_face_enrollment(admin, enr, True)
    sess = A.create_session(admin, sid, date.today(), (datetime.now() - timedelta(minutes=1)).strftime("%H:%M"))
    A.open_session(admin, sess)
    proc = K.KioskProcessor(sess, admin)
    return {"proc": proc, "session": sess, "admin": admin, "s1": s1, "s2": s2}


def run_frames(proc, n=40):
    for _ in range(n):
        proc.recv(FRAME)
        if proc.state == K.RESULT:
            return proc.recent_events()[0]
    return None


def record(k, user):
    recs = A.get_records(k["admin"], user_id=user.id, session_id=k["session"])
    return recs[0] if recs else None


def test_kiosk_marks_live_recognised_student(kiosk):
    ev = run_frames(kiosk["proc"])
    assert ev and ev.ok and ev.status == "present" and ev.user_id == kiosk["s1"].id
    rec = record(kiosk, kiosk["s1"])
    assert rec["status"] == "present" and rec["source"] == "kiosk"
    assert rec["liveness_score"] == pytest.approx(REAL.mean(axis=0)[1])
    assert rec["match_distance"] < 0.01


def test_kiosk_rejects_spoof_and_counts_failed_attempt(kiosk):
    proc = kiosk["proc"]
    FakePassive.probs = SPOOF
    ev = run_frames(proc)
    assert ev and not ev.ok and "Spoof" in ev.message
    assert record(kiosk, kiosk["s1"]) is None
    assert proc.fail_counts == {kiosk["s1"].id: 1}
    # Next, a genuine attempt: the earlier failure is stored with the record.
    FakePassive.probs = REAL
    proc.state, proc.result_until = K.IDLE, 0
    ev = run_frames(proc)
    assert ev.ok and record(kiosk, kiosk["s1"])["failed_attempts"] == 1


def test_kiosk_rejects_unknown_face(kiosk):
    FakeEmbedder.vec = unit(99)                          # nobody enrolled has this face
    ev = run_frames(kiosk["proc"])
    assert ev and not ev.ok and "not recognised" in ev.message


def test_kiosk_rejects_student_not_in_subject(kiosk):
    FakeEmbedder.vec = unit(2)                           # s2 is enrolled (face) but not in the subject
    ev = run_frames(kiosk["proc"])
    assert ev and not ev.ok and "not enrolled" in ev.message
    assert record(kiosk, kiosk["s2"]) is None


def test_kiosk_multiple_faces(kiosk):
    proc = kiosk["proc"]
    FakeDetector.faces = [FACE, FACE]
    proc.recv(FRAME)
    assert proc.state == K.IDLE and "one person" in proc.hint
    # A second face appearing mid-attempt is rejected.
    FakeDetector.faces = [FACE]
    for _ in range(5):
        proc.recv(FRAME)
    assert proc.state in (K.PASSIVE, K.CHALLENGE)
    FakeDetector.faces = [FACE, FACE]
    proc.recv(FRAME)
    assert proc.state == K.RESULT and "More than one face" in proc.recent_events()[0].message


def test_kiosk_no_duplicate_mark(kiosk):
    proc = kiosk["proc"]
    assert run_frames(proc).ok
    proc.state, proc.result_until = K.IDLE, 0
    ev = run_frames(proc)
    assert "already marked" in ev.message


def test_admin_faces_excluded_from_gallery(kiosk):
    admin = kiosk["admin"]
    enr, _ = db.save_face_enrollment(admin, admin.id, [unit(7)] * 5, db.now(), "fake", {}, None, None)
    db.review_face_enrollment(admin, enr, True)
    g = Gallery.from_db()
    assert admin.id not in set(g.user_ids.tolist())
    assert {kiosk["s1"].id, kiosk["s2"].id} <= set(g.user_ids.tolist())
