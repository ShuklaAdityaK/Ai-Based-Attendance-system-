"""Attendance kiosk pipeline running inside the streamlit-webrtc video thread.

Per student:
    IDLE      wait for exactly one, large-enough face (0 or >1 faces -> retry)
    PASSIVE   average MiniFASNet real-probability over N frames
    CHALLENGE random active challenge, verified with MediaPipe face mesh,
              passive scoring continues on every 2nd frame
    (recognise) ArcFace 1:N match; identity must equal the one seen at the
              start of the attempt (prevents swapping faces mid-attempt)
    RESULT    show the outcome, then back to IDLE

Failed attempts (spoof / challenge timeout / unknown) are attributed to the
provisional identity seen at the start of the attempt and stored with the
student's eventual successful record (feature for the anomaly detector).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import av
import cv2
import numpy as np

from core import attendance, config
from core.access import Actor
from vision.detector import Face, FaceDetector
from vision.liveness import ActiveChallenge, FaceMesh, get_passive, passive_is_real
from vision.recognizer import Gallery, cosine_distance, get_embedder

IDLE, PASSIVE, CHALLENGE, RESULT = "idle", "passive", "challenge", "result"

GREEN, RED, AMBER, WHITE, BLUE = (60, 200, 60), (40, 40, 230), (0, 190, 255), (255, 255, 255), (230, 160, 40)


@dataclass
class KioskEvent:
    at: str
    ok: bool
    title: str
    message: str
    user_id: int | None = None
    status: str | None = None
    distance: float | None = None
    liveness: float | None = None
    proc_ms: float | None = None


@dataclass
class _Attempt:
    started: float = field(default_factory=time.monotonic)
    passive_scores: list = field(default_factory=list)       # fused real-probability per frame
    passive_probs: list = field(default_factory=list)        # (n_models, 3) per frame
    probe_start: np.ndarray | None = None
    provisional_uid: int | None = None
    challenge: ActiveChallenge | None = None
    proc_s: float = 0.0          # compute time excluding the challenge
    frame_no: int = 0
    lost_since: float | None = None
    face_px: list = field(default_factory=list)
    brightness: list = field(default_factory=list)


class KioskProcessor:
    """streamlit-webrtc video processor. Created per stream by the factory in
    ``app_pages/kiosk.py`` with the open session and the admin actor."""

    def __init__(self, session_id: int, actor: Actor):
        self.session_id = session_id
        self.actor = actor
        self.detector = FaceDetector()
        self.embedder = get_embedder()
        self.passive = get_passive()
        self.mesh = FaceMesh()
        self.gallery = Gallery.from_db()
        self.lock = threading.Lock()
        self.events: deque[KioskEvent] = deque(maxlen=50)
        self.fail_counts: dict[int, int] = {}
        self.state = IDLE
        self.att: _Attempt | None = None
        self.result: tuple[str, str, tuple] | None = None
        self.result_until = 0.0
        self.stable_frames = 0
        self.hint = "Position your face inside the frame"
        cfg = config.get
        self.passive_frames = int(cfg("liveness.passive_frames", 5))
        self.passive_thr = float(cfg("liveness.passive_threshold", 0.6))
        self.match_thr = float(cfg("recognition.match_threshold", 0.6))
        self.min_face = int(cfg("enrollment.min_face_px", 90))
        self.result_secs = float(cfg("kiosk.result_display_seconds", 3))
        self.debug = False                     # live diagnostic overlay (toggled from the kiosk page)
        self.live: dict = {}                   # latest per-frame values for the overlay
        self._frame_times: deque = deque(maxlen=30)

    # ------------------------------------------------------------ helpers --
    def reload_gallery(self) -> int:
        g = Gallery.from_db()
        with self.lock:
            self.gallery = g
            self.match_thr = float(config.get("recognition.match_threshold", 0.6))
            self.passive_thr = float(config.get("liveness.passive_threshold", 0.6))
        return len(g)

    def _name(self, uid: int | None) -> str:
        if uid is None:
            return "Unknown"
        u = self.gallery.names.get(uid, {})
        return f"{u.get('full_name', uid)} ({u.get('roll_no') or u.get('username', '')})"

    def _emit(self, ev: KioskEvent) -> None:
        with self.lock:
            self.events.appendleft(ev)

    def _finish(self, ok: bool, title: str, msg: str, color: tuple, **kw) -> None:
        self.result = (title, msg, color)
        self.result_until = time.monotonic() + self.result_secs
        self.state = RESULT
        self._emit(KioskEvent(datetime.now().strftime("%H:%M:%S"), ok, title, msg, **kw))
        self.att = None

    def _fail(self, reason: str, title: str = "Rejected") -> None:
        uid = self.att.provisional_uid if self.att else None
        if uid is not None:
            self.fail_counts[uid] = self.fail_counts.get(uid, 0) + 1
        config.log.info("Kiosk reject session=%s provisional=%s reason=%s | %s", self.session_id, uid, reason,
                        self._diag())
        self._finish(False, title, reason, RED, user_id=uid)

    def fps(self) -> float:
        t = self._frame_times
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    def _diag(self) -> str:
        """One-line diagnostics of the current attempt (for tuning thresholds)."""
        att = self.att
        if att is None:
            return ""
        parts = [f"fps={self.fps():.1f}"]
        if att.passive_probs:
            pm = np.mean(att.passive_probs, axis=0)                  # (n_models, 3)
            fused = pm.mean(axis=0)
            parts.append("passive_fused=[" + ",".join(f"{v:.2f}" for v in fused) + "]")
            parts.append("per_model_real=[" + ",".join(f"{v:.2f}" for v in pm[:, 1]) + "]")
            parts.append(f"passive_frames={len(att.passive_probs)}")
        if att.face_px:
            parts.append(f"face_px={np.mean(att.face_px):.0f} brightness={np.mean(att.brightness):.0f}")
        if att.challenge:
            parts.append(att.challenge.summary())
        return " ".join(parts)

    def _passive_step(self, img: np.ndarray, face: Face) -> None:
        att = self.att
        probs = self.passive.model_probabilities(img, face.box)
        att.passive_probs.append(probs)
        att.passive_scores.append(float(probs.mean(axis=0)[1]))
        x, y, w, h = face.box
        att.face_px.append(w)
        att.brightness.append(float(cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).mean()))
        self.live["real"] = att.passive_scores[-1]

    def _passive_ok(self) -> tuple[bool, float]:
        fused = np.mean([p.mean(axis=0) for p in self.att.passive_probs], axis=0)
        return passive_is_real(fused, self.passive_thr), float(fused[1])

    def _embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        return self.embedder.embed(img, face.landmarks)

    # -------------------------------------------------------------- frame --
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_times.append(time.monotonic())
        try:
            face, n_faces = self._step(img)
        except Exception as e:  # never kill the video thread
            config.log.exception("Kiosk frame error: %s", e)
            self.state, self.att = IDLE, None
            face, n_faces = None, 0
        return av.VideoFrame.from_ndarray(self._draw(img, face, n_faces), format="bgr24")

    def _step(self, img: np.ndarray) -> tuple[Face | None, int]:
        now = time.monotonic()
        if self.state == RESULT:
            if now < self.result_until:
                return None, 0
            self.state, self.result, self.stable_frames = IDLE, None, 0

        t0 = time.perf_counter()
        faces = self.detector.detect(img)
        det_s = time.perf_counter() - t0
        n = len(faces)
        face = faces[0] if n == 1 else None

        if self.state == IDLE:
            if n == 0:
                self.hint, self.stable_frames = "Position your face inside the frame", 0
            elif n > 1:
                self.hint, self.stable_frames = "Only one person at a time, please", 0
            elif face.box[2] < self.min_face:
                self.hint, self.stable_frames = "Come closer to the camera", 0
            else:
                self.stable_frames += 1
                self.hint = "Hold still..."
                if self.stable_frames >= 3:
                    self.state, self.att = PASSIVE, _Attempt()
            return face, n

        att = self.att
        assert att is not None
        att.frame_no += 1
        if face is None:
            # Face lost (or a second face appeared) mid-attempt.
            if n > 1:
                self._fail("More than one face in the frame.", "Multiple faces")
                return None, n
            att.lost_since = att.lost_since or now
            if now - att.lost_since > 1.5:
                self.state, self.att, self.stable_frames = IDLE, None, 0
            return None, n
        att.lost_since = None
        att.proc_s += det_s

        if self.state == PASSIVE:
            t0 = time.perf_counter()
            self._passive_step(img, face)
            if att.probe_start is None:
                att.probe_start = self._embed(img, face)
                m = self.gallery.match(att.probe_start, self.match_thr)
                att.provisional_uid = m.user_id
                self.live["dist"] = m.distance
            att.proc_s += time.perf_counter() - t0
            if len(att.passive_scores) >= self.passive_frames:
                ok, score = self._passive_ok()
                if not ok:
                    self._fail(f"Spoof suspected (liveness score {score:.2f}). Use your real face.",
                               "Liveness failed")
                    return face, n
                att.challenge = ActiveChallenge.random()
                self.state = CHALLENGE
            self.hint = "Checking liveness..."
            return face, n

        if self.state == CHALLENGE:
            ch = att.challenge
            assert ch is not None
            if ch.expired():
                self._fail("Challenge not completed in time.", "Liveness failed")
                return face, n
            if att.frame_no % 3 == 0:
                self._passive_step(img, face)
            m = self.mesh.metrics(img)
            if m is not None:
                self.live.update(ear=m.ear, blink=m.blink_score, turn=m.turn_ratio)
            if m is not None and ch.update(m):
                self._recognise(img, face)
            else:
                self.hint = ch.prompt
            return face, n
        return face, n

    def _recognise(self, img: np.ndarray, face: Face) -> None:
        att = self.att
        assert att is not None
        t0 = time.perf_counter()
        ok, live = self._passive_ok()
        if not ok:
            self._fail(f"Spoof suspected (liveness score {live:.2f}).", "Liveness failed")
            return
        probe = self._embed(img, face)
        # The end frame may be a turned face, so allow a looser same-person check
        # and match with both the frontal start probe and the end probe.
        if att.probe_start is not None and cosine_distance(probe, att.probe_start) > self.match_thr + 0.2:
            self._fail("Face changed during the challenge.", "Rejected")
            return
        m_end = self.gallery.match(probe, self.match_thr)
        m = m_end
        if att.probe_start is not None:
            m_start = self.gallery.match(att.probe_start, self.match_thr)
            if m_start.user_id and m_end.user_id and m_start.user_id != m_end.user_id:
                self._fail("Inconsistent identity during the attempt.", "Rejected")
                return
            if m_start.user_id and (m_end.user_id is None or m_start.distance < m_end.distance):
                m = m_start
        att.proc_s += time.perf_counter() - t0
        proc_ms = att.proc_s * 1000
        if m.user_id is None:
            self._fail("Face not recognised. Are you enrolled and approved?", "Unknown face")
            return
        config.log.info("Kiosk accept session=%s user=%s dist=%.3f | %s", self.session_id, m.user_id, m.distance,
                        self._diag())
        fails = self.fail_counts.pop(m.user_id, 0)
        res = attendance.mark_from_kiosk(self.actor, self.session_id, m.user_id, m.distance, live, fails)
        name = self._name(m.user_id)
        if res.ok:
            color = GREEN if res.status == attendance.PRESENT else AMBER
            self._finish(True, f"{res.status.title()}: {name}", res.message, color, user_id=m.user_id,
                         status=res.status, distance=m.distance, liveness=live, proc_ms=proc_ms)
        else:
            color = BLUE if res.code == "already" else RED
            self._finish(res.code == "already", name, res.message, color, user_id=m.user_id,
                         status=res.status, distance=m.distance, liveness=live, proc_ms=proc_ms)

    # ------------------------------------------------------------ drawing --
    def _draw(self, img: np.ndarray, face: Face | None, n_faces: int) -> np.ndarray:
        out = cv2.flip(img, 1)                     # selfie view for the student
        H, W = out.shape[:2]
        if face is not None:
            x, y, w, h = face.box
            color = {IDLE: WHITE, PASSIVE: AMBER, CHALLENGE: AMBER}.get(self.state, WHITE)
            cv2.rectangle(out, (W - x - w, y), (W - x, y + h), color, 2)
        cv2.rectangle(out, (0, 0), (W, 56), (30, 30, 30), -1)
        if self.state == RESULT and self.result:
            title, msg, color = self.result
            cv2.rectangle(out, (0, H - 90), (W, H), color, -1)
            _text(out, title[:48], (12, H - 55), 0.75, WHITE, 2, outline=False)
            _text(out, msg[:70], (12, H - 20), 0.55, WHITE, 1, outline=False)
            _text(out, "Next student, please", (12, 37), 0.8, WHITE, 2)
            return out
        _text(out, self.hint, (12, 37), 0.8, WHITE, 2)
        if self.state == CHALLENGE and self.att and self.att.challenge:
            frac = max(0.0, self.att.challenge.time_left()) / float(config.get("liveness.challenge_timeout_seconds", 8))
            cv2.rectangle(out, (0, 56), (int(W * frac), 64), AMBER, -1)
        elif self.state == PASSIVE and self.att:
            frac = len(self.att.passive_scores) / max(1, self.passive_frames)
            cv2.rectangle(out, (0, 56), (int(W * frac), 64), BLUE, -1)
        if self.debug:
            lv = self.live
            lines = [f"fps {self.fps():.1f}  state {self.state}  face {face.box[2] if face else '-'}px",
                     f"real {lv.get('real', 0):.2f} (min {self.passive_thr:.2f})  dist {lv.get('dist', 9):.2f} "
                     f"(max {self.match_thr:.2f})",
                     f"EAR {lv.get('ear', 0):.3f}  blink {lv.get('blink', 0):.2f}  turn {lv.get('turn', 0.5):.2f}"]
            for i, ln in enumerate(lines):
                _text(out, ln, (10, 90 + 22 * i), 0.5, (255, 255, 0), 1)
        return out

    def recent_events(self) -> list[KioskEvent]:
        with self.lock:
            return list(self.events)

    def __del__(self):
        try:
            self.mesh.close()
        except Exception:
            pass


def _text(img, s, org, scale, color, thick, outline: bool = True):
    if outline:
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
