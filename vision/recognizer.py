"""ArcFace embeddings and 1:N cosine matching.

The embedding network is InsightFace's ArcFace ResNet-50 (``w600k_r50``,
from the ``buffalo_l`` model pack, trained on WebFace600K with the ArcFace
additive-angular-margin loss). It is executed directly with ONNX Runtime,
which avoids the native build of the ``insightface`` pip package on Windows
while producing identical embeddings. Faces are aligned to the standard
112x112 ArcFace template using the 5 YuNet landmarks.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from core import config, db

# Standard ArcFace 112x112 landmark template (InsightFace).
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def align_face(img_bgr: np.ndarray, landmarks5: np.ndarray, size: int = 112) -> np.ndarray:
    # YuNet order is (right eye, left eye, nose, right mouth, left mouth) from the
    # subject's perspective, i.e. image-left eye first - same order as the template.
    M, _ = cv2.estimateAffinePartial2D(landmarks5.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        raise ValueError("Face alignment failed.")
    return cv2.warpAffine(img_bgr, M, (size, size), borderValue=0.0)


class FaceEmbedder:
    """ONNX Runtime sessions are thread-safe for ``run``; one shared instance."""

    def __init__(self):
        model = config.path("models") / config.get("recognition.embedding_model")
        if not model.exists():
            raise FileNotFoundError(f"ArcFace model missing: {model}. Run scripts/download_models.py")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.model_name = str(config.get("recognition.model_name", "ArcFace"))

    def embed_aligned(self, aligned_bgr: np.ndarray | list[np.ndarray]) -> np.ndarray:
        imgs = aligned_bgr if isinstance(aligned_bgr, list) else [aligned_bgr]
        blob = cv2.dnn.blobFromImages(imgs, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = self.sess.run(None, {self.input_name: blob})[0]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.astype(np.float32)

    def embed(self, img_bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
        return self.embed_aligned(align_face(img_bgr, landmarks5))[0]


_embedder: FaceEmbedder | None = None
_lock = threading.Lock()


def get_embedder() -> FaceEmbedder:
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = FaceEmbedder()
        return _embedder


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@dataclass
class Match:
    user_id: int | None
    distance: float
    second_distance: float
    accepted: bool


class Gallery:
    """In-memory matrix of approved embeddings for fast 1:N matching."""

    def __init__(self, items: list[tuple[int, np.ndarray]], names: dict[int, dict] | None = None):
        self.user_ids = np.array([u for u, _ in items], dtype=np.int64)
        self.matrix = (np.stack([e / np.linalg.norm(e) for _, e in items]).astype(np.float32)
                       if items else np.zeros((0, 512), np.float32))
        self.names = names or {}

    @classmethod
    def from_db(cls) -> "Gallery":
        with db.tx() as con:
            items = db.all_embeddings(con, ("approved",))
            names = {r["id"]: r for r in db.rows(
                con, "SELECT id, full_name, roll_no, username FROM users WHERE is_active=1 AND role='student'")}
        items = [(u, e) for u, e in items if u in names]   # deactivated users are not recognised
        return cls(items, names)

    def __len__(self) -> int:
        return len(set(self.user_ids.tolist()))

    def per_user_distances(self, probe: np.ndarray) -> dict[int, float]:
        if self.matrix.shape[0] == 0:
            return {}
        d = 1.0 - self.matrix @ (probe / np.linalg.norm(probe))
        best: dict[int, float] = {}
        for uid, dist in zip(self.user_ids.tolist(), d.tolist()):
            if dist < best.get(uid, 9.0):
                best[uid] = dist
        return best

    def match(self, probe: np.ndarray, threshold: float | None = None) -> Match:
        thr = float(config.get("recognition.match_threshold", 0.6) if threshold is None else threshold)
        best = sorted(self.per_user_distances(probe).items(), key=lambda kv: kv[1])
        if not best:
            return Match(None, 9.0, 9.0, False)
        uid, dist = best[0]
        second = best[1][1] if len(best) > 1 else 9.0
        return Match(uid if dist < thr else None, float(dist), float(second), dist < thr)


def find_duplicate(new_embeddings: list[np.ndarray], user_id: int,
                   threshold: float | None = None) -> tuple[int | None, float | None]:
    """Compare a new enrollment with every other enrolled user (approved,
    pending or blocked). Returns (matching user_id, mean distance) or (None, best)."""
    thr = float(config.get("recognition.duplicate_threshold", 0.55) if threshold is None else threshold)
    with db.tx() as con:
        items = db.all_embeddings(con, ("approved", "pending", "blocked"), exclude_user=user_id)
    if not items:
        return None, None
    g = Gallery(items)
    per_user: dict[int, list[float]] = {}
    for e in new_embeddings:
        for uid, dist in g.per_user_distances(e).items():
            per_user.setdefault(uid, []).append(dist)
    scored = sorted(((uid, float(np.mean(ds))) for uid, ds in per_user.items()), key=lambda kv: kv[1])
    uid, dist = scored[0]
    return (uid, dist) if dist < thr else (None, dist)
