"""Face detection with OpenCV YuNet plus enrollment frame-quality checks."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core import config


@dataclass
class Face:
    box: tuple[int, int, int, int]          # x, y, w, h
    landmarks: np.ndarray                   # (5, 2): r-eye, l-eye, nose, r-mouth, l-mouth (YuNet order)
    score: float

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        x, y, w, h = self.box
        return x, y, x + w, y + h


class FaceDetector:
    """Thin wrapper around ``cv2.FaceDetectorYN``. Not thread-safe: create one
    instance per thread (the kiosk video thread owns its own)."""

    def __init__(self, score_threshold: float | None = None):
        model = config.path("models") / config.get("recognition.detector_model")
        if not model.exists():
            raise FileNotFoundError(f"YuNet model missing: {model}. Run scripts/download_models.py")
        thr = float(score_threshold or config.get("recognition.detector_score_threshold", 0.8))
        self._det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), thr, 0.3, 5000)
        self._size = (320, 320)

    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        h, w = img_bgr.shape[:2]
        if (w, h) != self._size:
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._det.detect(img_bgr)
        out: list[Face] = []
        if faces is None:
            return out
        for f in faces:
            x, y, bw, bh = (int(round(v)) for v in f[:4])
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw <= 0 or bh <= 0:
                continue
            out.append(Face((x, y, bw, bh), f[4:14].reshape(5, 2).astype(np.float32), float(f[14])))
        out.sort(key=lambda fc: fc.box[2] * fc.box[3], reverse=True)
        return out


def estimate_yaw(face: Face) -> float:
    """Approximate yaw in degrees from the 5 YuNet landmarks.

    Positive = nose toward the image right (the subject turned to their own
    left, for an un-mirrored camera image)."""
    r_eye, l_eye, nose = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    eye_mid = (r_eye + l_eye) / 2
    eye_dist = float(np.linalg.norm(l_eye - r_eye)) or 1.0
    ratio = float((nose[0] - eye_mid[0]) / eye_dist)
    return float(np.degrees(np.arcsin(np.clip(ratio * 1.6, -1, 1))))


@dataclass
class QualityReport:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    face: Face | None = None


def face_crop(img: np.ndarray, face: Face, pad: float = 0.0) -> np.ndarray:
    x, y, w, h = face.box
    px, py = int(w * pad), int(h * pad)
    H, W = img.shape[:2]
    return img[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]


def check_quality(img_bgr: np.ndarray, faces: list[Face]) -> QualityReport:
    """Enrollment frame checks: exactly one face, size, sharpness, lighting, pose."""
    if len(faces) == 0:
        return QualityReport(False, ["No face detected."])
    if len(faces) > 1:
        return QualityReport(False, [f"{len(faces)} faces detected - only you should be in the frame."])
    face = faces[0]
    H, W = img_bgr.shape[:2]
    x, y, w, h = face.box
    crop = cv2.cvtColor(face_crop(img_bgr, face), cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(cv2.resize(crop, (160, 160)), cv2.CV_64F).var())
    bright, contrast = float(crop.mean()), float(crop.std())
    yaw = estimate_yaw(face)
    m = {"face_px": w, "face_ratio": round(w / W, 3), "blur": round(blur, 1), "brightness": round(bright, 1),
         "contrast": round(contrast, 1), "yaw": round(yaw, 1), "det_score": round(face.score, 3)}
    reasons: list[str] = []
    if w < int(config.get("enrollment.min_face_px", 90)) and w / W < float(config.get("enrollment.min_face_ratio", 0.18)):
        reasons.append("Face too small - move closer to the camera.")
    if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
        reasons.append("Face is cut off at the frame edge - centre your face.")
    if blur < float(config.get("enrollment.blur_threshold", 60)):
        reasons.append(f"Image is blurred (sharpness {blur:.0f}) - hold still.")
    if bright < float(config.get("enrollment.brightness_min", 60)):
        reasons.append("Too dark - improve the lighting.")
    elif bright > float(config.get("enrollment.brightness_max", 200)):
        reasons.append("Too bright / over-exposed - avoid direct light.")
    if contrast < float(config.get("enrollment.contrast_min", 25)):
        reasons.append("Low contrast - lighting is too flat or washed out.")
    if abs(yaw) > float(config.get("enrollment.max_abs_yaw_deg", 35)):
        reasons.append("Head turned too far - turn only slightly.")
    return QualityReport(not reasons, reasons, m, face)
