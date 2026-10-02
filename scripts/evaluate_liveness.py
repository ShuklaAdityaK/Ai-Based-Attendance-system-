"""Passive-liveness evaluation on real vs printed-photo vs screen-replay samples.

    python scripts/evaluate_liveness.py [--data eval_data/liveness]

Folder layout (images and/or short videos, >= 20 attempts per class):
    eval_data/liveness/real/      genuine live faces
    eval_data/liveness/print/     printed-photo attacks
    eval_data/liveness/replay/    phone / laptop screen replays
Record them with scripts/capture_liveness_samples.py.

Each attempt is scored like the kiosk does: the mean MiniFASNet real-probability
over the first N frames (liveness.passive_frames). Reports, per threshold:
    attack rejection rate  (per attack type and overall)  - higher is better
    false-reject rate      (real users rejected)          - lower is better
The ACTIVE challenge must be evaluated live at the kiosk; use
docs/liveness_trial_sheet.csv to log those trials.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config  # noqa: E402
from vision.detector import FaceDetector  # noqa: E402
from vision.liveness import get_passive  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp"}
VID = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def frames_of(path: Path, n: int) -> list[np.ndarray]:
    if path.suffix.lower() in IMG:
        img = cv2.imread(str(path))
        return [img] if img is not None else []
    cap = cv2.VideoCapture(str(path))
    out = []
    while len(out) < n * 3:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out[::3][:n] if len(out) >= n else out[:n]


def score_attempt(path: Path, det: FaceDetector, n: int) -> float | None:
    scores = []
    for f in frames_of(path, n):
        faces = det.detect(f)
        if len(faces) == 1:
            scores.append(get_passive().real_probability(f, faces[0].box))
    return float(np.mean(scores)) if scores else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "eval_data" / "liveness")
    args = ap.parse_args()
    n = int(config.get("liveness.passive_frames", 5))
    det = FaceDetector(score_threshold=0.6)
    scores: dict[str, list[float]] = {}
    for cls in ("real", "print", "replay"):
        folder = args.data / cls
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG | VID) if folder.exists() else []
        vals = [s for s in (score_attempt(p, det, n) for p in files) if s is not None]
        scores[cls] = vals
        print(f"{cls:7s}: {len(vals)} scored attempts (of {len(files)} files)")
    if not scores.get("real") or not (scores.get("print") or scores.get("replay")):
        print("Need samples in real/ and at least one attack folder. See the docstring.")
        return 1

    def at(thr: float) -> dict:
        r = {"threshold": thr, "false_reject_rate_real": round(float(np.mean(np.array(scores["real"]) < thr)), 4)}
        attacks = []
        for a in ("print", "replay"):
            if scores.get(a):
                arr = np.array(scores[a])
                r[f"attack_rejection_{a}"] = round(float(np.mean(arr < thr)), 4)
                attacks.append(arr)
        r["attack_rejection_overall"] = round(float(np.mean(np.concatenate(attacks) < thr)), 4)
        return r

    sweep = [at(round(t, 2)) for t in np.arange(0.3, 0.96, 0.05)]
    current = at(float(config.get("liveness.passive_threshold")))
    result = {"attempts": {k: len(v) for k, v in scores.items()},
              "mean_score": {k: round(float(np.mean(v)), 4) for k, v in scores.items() if v},
              "at_configured_threshold": current, "sweep": sweep}
    out = ROOT / "eval_results"
    out.mkdir(exist_ok=True)
    (out / "liveness_passive.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("attempts", "mean_score", "at_configured_threshold")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
