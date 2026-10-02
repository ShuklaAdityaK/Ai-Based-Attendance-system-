"""Face enrollment workflow (Section 2.2): quality-checked multi-frame capture,
embedding generation, duplicate-face check and submission for approval."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from core import config, db
from core.access import Actor, require_self_or_admin
from vision.detector import FaceDetector, QualityReport, check_quality
from vision.recognizer import find_duplicate, get_embedder

# Prompts cycle through these poses until max_frames are accepted.
POSE_PROMPTS = ["Look straight at the camera", "Turn your head SLIGHTLY to your left",
                "Turn your head SLIGHTLY to your right", "Look straight, chin slightly up",
                "Look straight, chin slightly down"]

CONSENT_NOTICE = """**Biometric data consent notice**

By continuing you consent to the following processing of your biometric data,
in line with India's *Digital Personal Data Protection Act, 2023*:

* **Purpose (limited):** your face is used **only** to mark your attendance at the
  classroom kiosk and to stop one person from holding two accounts.
* **What is stored:** numerical face *embeddings* (512 numbers per frame). Your
  photos are **discarded** right after the embeddings are computed.
* **Who can access it:** only this attendance system; it is never shared.
* **Retention & deletion:** the data is kept while you are enrolled. It is deleted
  if your account is deleted, and you may request deletion from the administrator
  at any time.
* **Withdrawal:** you may withdraw consent; attendance will then be marked manually.
"""


@dataclass
class FrameResult:
    report: QualityReport
    embedding: np.ndarray | None = None


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the camera image.")
    return img


def process_frame(img_bgr: np.ndarray, detector: FaceDetector | None = None) -> FrameResult:
    detector = detector or FaceDetector()
    report = check_quality(img_bgr, detector.detect(img_bgr))
    if not report.ok:
        return FrameResult(report)
    emb = get_embedder().embed(img_bgr, report.face.landmarks)
    return FrameResult(report, emb)


def consistency_outliers(embeddings: list[np.ndarray], max_dist: float = 0.5) -> list[int]:
    """Indices of frames whose embedding is far from the others' centroid
    (e.g. a different person stepped in)."""
    if len(embeddings) < 3:
        return []
    m = np.stack(embeddings)
    out = []
    for i in range(len(m)):
        c = np.delete(m, i, axis=0).mean(axis=0)
        c /= np.linalg.norm(c)
        if 1 - float(m[i] @ c) > max_dist:
            out.append(i)
    return out


def submit_enrollment(actor: Actor, user_id: int, embeddings: list[np.ndarray], consent_at: datetime,
                      quality_metrics: list[dict], raw_frames: list[np.ndarray] | None = None) -> dict:
    """Run the duplicate check and store the embeddings as a Pending enrollment.

    Raw frames are written to disk only when ``enrollment.keep_raw_images`` is
    true; otherwise they are dropped here and never persisted."""
    require_self_or_admin(actor, user_id)
    lo, hi = int(config.get("enrollment.min_frames", 5)), int(config.get("enrollment.max_frames", 10))
    if not lo <= len(embeddings) <= hi:
        raise ValueError(f"Between {lo} and {hi} accepted frames are required.")
    if consistency_outliers(embeddings):
        raise ValueError("The captured frames do not all show the same person. Please restart the capture.")
    dup_uid, dup_dist = find_duplicate(embeddings, user_id)
    summary = {
        "frames": len(embeddings),
        "mean_blur": round(float(np.mean([m["blur"] for m in quality_metrics])), 1),
        "mean_brightness": round(float(np.mean([m["brightness"] for m in quality_metrics])), 1),
        "yaw_range": [min(m["yaw"] for m in quality_metrics), max(m["yaw"] for m in quality_metrics)],
        "nearest_other_user_distance": None if dup_dist is None else round(dup_dist, 4),
    }
    enr_id, status = db.save_face_enrollment(
        actor, user_id, embeddings, consent_at.isoformat(timespec="seconds"),
        get_embedder().model_name, summary, dup_uid, dup_dist)
    if raw_frames and config.get("enrollment.keep_raw_images", False):
        folder = config.path("raw_faces") / str(user_id) / str(enr_id)
        folder.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(raw_frames):
            cv2.imwrite(str(folder / f"{i:02d}.jpg"), f)
    config.log.info("Face enrollment %s user=%s status=%s %s", enr_id, user_id, status, json.dumps(summary))
    return {"enrollment_id": enr_id, "status": status, "summary": summary}
