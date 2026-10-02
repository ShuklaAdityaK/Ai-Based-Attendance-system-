"""Two-layer liveness detection (Section 2.4).

1. Passive: MiniFASNetV2 (scale 2.7) + MiniFASNetV1SE (scale 4.0) from
   Silent-Face-Anti-Spoofing (Minivision), run with ONNX Runtime. Each model
   outputs 3 logits; class 1 is "real". The real-probabilities of the two
   models are averaged (the fusion used by the original project).
2. Active: a random challenge (blink twice / turn head left / turn head
   right) verified with MediaPipe Face Landmarker (478-point face mesh):
   blinks via the Eye Aspect Ratio, head turns via the nose position
   between the cheek contour points (a 2-D head-pose estimate).

Known limitation: targets printed-photo and phone/screen replay attacks;
not designed to stop 3D masks or deepfake injection.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

from core import config

# ---------------------------------------------------------------- passive -- #


def _crop_scaled(img: np.ndarray, box: tuple[int, int, int, int], scale: float, out: int = 80) -> np.ndarray:
    """Silent-Face style crop: enlarge the box around its centre by ``scale``."""
    src_h, src_w = img.shape[:2]
    x, y, bw, bh = box
    scale = min((src_h - 1) / max(bh, 1), (src_w - 1) / max(bw, 1), scale)
    nw, nh = bw * scale, bh * scale
    cx, cy = x + bw / 2, y + bh / 2
    x1, y1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
    x2, y2 = min(src_w - 1, int(cx + nw / 2)), min(src_h - 1, int(cy + nh / 2))
    return cv2.resize(img[y1:y2 + 1, x1:x2 + 1], (out, out))


class PassiveLiveness:
    def __init__(self):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.models = []
        for spec in config.get("liveness.passive_models"):
            path = config.path("models") / spec["file"]
            if not path.exists():
                raise FileNotFoundError(f"Anti-spoofing model missing: {path}. Run scripts/download_models.py")
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            self.models.append((sess, sess.get_inputs()[0].name, float(spec["scale"])))

    def model_probabilities(self, img_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        """Softmax over the 3 classes for each model: shape (n_models, 3); class 1 = real."""
        out = []
        for sess, name, scale in self.models:
            crop = _crop_scaled(img_bgr, box, scale).astype(np.float32)        # BGR, 0..255
            logits = sess.run(None, {name: crop.transpose(2, 0, 1)[None]})[0][0]
            e = np.exp(logits - logits.max())
            out.append(e / e.sum())
        return np.array(out)

    def probabilities(self, img_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        """Fused 3-class probabilities (mean of the models, as in Silent-Face)."""
        return self.model_probabilities(img_bgr, box).mean(axis=0)

    def real_probability(self, img_bgr: np.ndarray, box: tuple[int, int, int, int]) -> float:
        return float(self.probabilities(img_bgr, box)[1])


def passive_is_real(mean_probs: np.ndarray, threshold: float | None = None) -> bool:
    """Silent-Face decision rule: "real" must be the most likely class, and its
    probability must also reach the configured minimum."""
    thr = float(config.get("liveness.passive_threshold", 0.45) if threshold is None else threshold)
    return int(np.argmax(mean_probs)) == 1 and float(mean_probs[1]) >= thr


_passive: PassiveLiveness | None = None
_lock = threading.Lock()


def get_passive() -> PassiveLiveness:
    global _passive
    with _lock:
        if _passive is None:
            _passive = PassiveLiveness()
        return _passive


# ----------------------------------------------------------------- active -- #

# MediaPipe face-mesh indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]    # p1..p6 for EAR
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP, CHEEK_A, CHEEK_B = 1, 234, 454      # 234: image-left contour, 454: image-right


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)  (Soukupova & Cech, 2016)."""
    p1, p2, p3, p4, p5, p6 = pts
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4) + 1e-6))


@dataclass
class MeshMetrics:
    ear: float
    turn_ratio: float        # 0.5 = frontal, >0.5 = nose toward image right (subject's left)
    blink_score: float = 0.0  # MediaPipe eyeBlink blendshape (0 open .. 1 closed), mean of both eyes


class FaceMesh:
    """MediaPipe Face Landmarker (Tasks API). One instance per thread."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mpv

        model = config.path("models") / config.get("liveness.landmarker_model")
        if not model.exists():
            raise FileNotFoundError(f"Face landmarker model missing: {model}. Run scripts/download_models.py")
        opts = mpv.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model.read_bytes()),
            running_mode=mpv.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True)
        self._mp = mp
        self._lm = mpv.FaceLandmarker.create_from_options(opts)

    def metrics(self, img_bgr: np.ndarray) -> MeshMetrics | None:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None
        h, w = img_bgr.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)
        ear = (eye_aspect_ratio(pts[LEFT_EYE]) + eye_aspect_ratio(pts[RIGHT_EYE])) / 2
        a, b, n = pts[CHEEK_A][0], pts[CHEEK_B][0], pts[NOSE_TIP][0]
        ratio = float((n - a) / (b - a)) if abs(b - a) > 1 else 0.5
        blink = 0.0
        if res.face_blendshapes:
            bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
            blink = (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)) / 2
        return MeshMetrics(ear, ratio, float(blink))

    def close(self) -> None:
        try:
            self._lm.close()
        except Exception:
            pass


CHALLENGES = {
    "blink": "Blink twice",
    "turn_left": "Turn your head to YOUR LEFT",
    "turn_right": "Turn your head to YOUR RIGHT",
}


@dataclass
class ActiveChallenge:
    """State machine for one challenge. Feed per-frame ``MeshMetrics``."""
    kind: str
    started: float = field(default_factory=time.monotonic)
    blinks: int = 0
    _eye_closed: bool = False
    _ear_hist: list = field(default_factory=list)
    _frontal_seen: bool = False
    _turn_frames: int = 0
    done: bool = False
    # diagnostics
    frames: int = 0
    ear_min: float = 9.0
    ear_max: float = 0.0
    blink_max: float = 0.0
    ratio_min: float = 9.0
    ratio_max: float = 0.0

    @classmethod
    def random(cls) -> "ActiveChallenge":
        return cls(random.choice(list(CHALLENGES)))

    @property
    def prompt(self) -> str:
        p = CHALLENGES[self.kind]
        if self.kind == "blink":
            p += f"  ({self.blinks}/{int(config.get('liveness.blinks_required', 2))})"
        return p

    def time_left(self) -> float:
        return float(config.get("liveness.challenge_timeout_seconds", 8)) - (time.monotonic() - self.started)

    def expired(self) -> bool:
        return not self.done and self.time_left() <= 0

    def update(self, m: MeshMetrics) -> bool:
        if self.done or self.expired():
            return self.done
        self.frames += 1
        self.ear_min, self.ear_max = min(self.ear_min, m.ear), max(self.ear_max, m.ear)
        self.blink_max = max(self.blink_max, m.blink_score)
        self.ratio_min, self.ratio_max = min(self.ratio_min, m.turn_ratio), max(self.ratio_max, m.turn_ratio)
        if self.kind == "blink":
            # Eye closure = EAR drop below an adaptive fraction of the open-eye
            # baseline, or MediaPipe's blink blendshape (robust on low-res webcams).
            self._ear_hist.append(m.ear)
            hist = self._ear_hist[-45:]
            baseline = float(np.percentile(hist, 90)) if len(hist) >= 3 else max(m.ear, 0.25)
            closed_thr = float(config.get("liveness.ear_closed_ratio", 0.8)) * baseline
            bs_closed = float(config.get("liveness.blink_score_closed", 0.45))
            if m.ear < closed_thr or m.blink_score > bs_closed:
                self._eye_closed = True
            elif self._eye_closed and m.ear > (closed_thr + baseline) / 2 and m.blink_score < bs_closed * 0.6:
                self._eye_closed = False
                self.blinks += 1
            self.done = self.blinks >= int(config.get("liveness.blinks_required", 2))
        else:
            delta = float(config.get("liveness.turn_ratio_delta", 0.12))
            if abs(m.turn_ratio - 0.5) < delta * 0.75:
                self._frontal_seen = True        # must start from a frontal pose
            # Un-mirrored camera image: subject's left -> nose toward image right.
            turned = m.turn_ratio > 0.5 + delta if self.kind == "turn_left" else m.turn_ratio < 0.5 - delta
            self._turn_frames = self._turn_frames + 1 if (turned and self._frontal_seen) else 0
            self.done = self._turn_frames >= int(config.get("liveness.turn_hold_frames", 3))
        return self.done

    def summary(self) -> str:
        return (f"challenge={self.kind} frames={self.frames} blinks={self.blinks} ear=[{self.ear_min:.3f},"
                f"{self.ear_max:.3f}] blink_max={self.blink_max:.2f} turn_ratio=[{self.ratio_min:.2f},"
                f"{self.ratio_max:.2f}] frontal_seen={self._frontal_seen}")
