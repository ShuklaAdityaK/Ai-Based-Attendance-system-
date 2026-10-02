"""Webcam diagnostics for the kiosk: measures what the liveness and recognition
models actually see on THIS camera, so thresholds can be tuned from data.

    .venv\\Scripts\\python scripts\\kiosk_diagnostics.py [--camera 0]

Sit in front of the camera as a student would at the kiosk. A window guides
you through four 6-second steps:  look straight -> blink a few times ->
turn head to your LEFT -> turn head to your RIGHT.

Only numbers are recorded (no images are saved). Results are printed and
written to eval_results/kiosk_diagnostics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config, db  # noqa: E402
from vision.detector import FaceDetector  # noqa: E402
from vision.liveness import ActiveChallenge, FaceMesh, get_passive, passive_is_real  # noqa: E402
from vision.recognizer import Gallery, get_embedder  # noqa: E402

PHASES = [("straight", "Look straight at the camera"), ("blink", "Blink normally, 3-4 times"),
          ("turn_left", "Slowly turn your head to YOUR LEFT, then back"),
          ("turn_right", "Slowly turn your head to YOUR RIGHT, then back")]
PHASE_SECONDS = 6.0


def stats(v: list[float]) -> dict:
    if not v:
        return {}
    a = np.array(v, dtype=float)
    return {"min": round(float(a.min()), 3), "median": round(float(np.median(a)), 3), "max": round(float(a.max()), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    det, passive, mesh, emb = FaceDetector(), get_passive(), FaceMesh(), get_embedder()
    with db.tx() as con:
        gallery = Gallery(db.all_embeddings(con, ("approved",), students_only=False))
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Could not open the camera. Close other apps using it (browser kiosk, Zoom, Teams) and retry.")
        return 1
    for _ in range(10):          # let auto-exposure settle
        cap.read()

    report: dict = {"camera_resolution": None, "phases": {}}
    win = "Kiosk diagnostics (press q to abort)"
    for key, instruction in PHASES:
        rec = {k: [] for k in ("faces", "face_px", "brightness", "sharpness", "real", "real_v2", "real_v1se",
                               "passive_pass", "ear", "blink", "turn", "dist", "frame_ms")}
        challenge = ActiveChallenge(key) if key != "straight" else None
        t_end = time.monotonic() + PHASE_SECONDS
        countdown_end = time.monotonic() + 2.0
        while time.monotonic() < t_end:
            ok, frame = cap.read()
            if not ok:
                break
            report["camera_resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"
            show = cv2.flip(frame, 1)
            msg = instruction if time.monotonic() > countdown_end else f"Next: {instruction}"
            cv2.rectangle(show, (0, 0), (show.shape[1], 44), (30, 30, 30), -1)
            cv2.putText(show, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 1
            if time.monotonic() < countdown_end:
                continue
            t0 = time.perf_counter()
            faces = det.detect(frame)
            rec["faces"].append(len(faces))
            if len(faces) != 1:
                continue
            f = faces[0]
            x, y, w, h = f.box
            grey = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            rec["face_px"].append(w)
            rec["brightness"].append(float(grey.mean()))
            rec["sharpness"].append(float(cv2.Laplacian(cv2.resize(grey, (160, 160)), cv2.CV_64F).var()))
            pm = passive.model_probabilities(frame, f.box)
            fused = pm.mean(axis=0)
            rec["real"].append(float(fused[1]))
            rec["real_v2"].append(float(pm[0, 1]))
            rec["real_v1se"].append(float(pm[1, 1]))
            rec["passive_pass"].append(bool(passive_is_real(fused)))
            m = mesh.metrics(frame)
            if m:
                rec["ear"].append(m.ear)
                rec["blink"].append(m.blink_score)
                rec["turn"].append(m.turn_ratio)
                if challenge:
                    challenge.update(m)
            if len(gallery):
                rec["dist"].append(gallery.match(emb.embed(frame, f.landmarks)).distance)
            rec["frame_ms"].append((time.perf_counter() - t0) * 1000)
        n = len(rec["faces"])
        out = {
            "frames": n, "one_face_frames": int(sum(1 for c in rec["faces"] if c == 1)),
            "face_px": stats(rec["face_px"]), "brightness": stats(rec["brightness"]),
            "sharpness": stats(rec["sharpness"]),
            "passive_real_fused": stats(rec["real"]), "passive_real_MiniFASNetV2": stats(rec["real_v2"]),
            "passive_real_MiniFASNetV1SE": stats(rec["real_v1se"]),
            "passive_pass_rate": round(float(np.mean(rec["passive_pass"])), 3) if rec["passive_pass"] else None,
            "ear": stats(rec["ear"]), "blink_score": stats(rec["blink"]), "turn_ratio": stats(rec["turn"]),
            "match_distance_to_enrolled": stats(rec["dist"]),
            "processing_ms_per_frame": stats(rec["frame_ms"]),
        }
        if challenge:
            out["challenge_completed"] = challenge.done
            out["challenge_detail"] = challenge.summary()
        report["phases"][key] = out
    cap.release()
    cv2.destroyAllWindows()
    mesh.close()

    report["config"] = {k: config.get(k) for k in ("liveness.passive_threshold", "recognition.match_threshold",
                                                   "liveness.ear_closed_ratio", "liveness.blink_score_closed",
                                                   "liveness.turn_ratio_delta")}
    report["enrolled_faces_in_db"] = len(gallery)
    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "kiosk_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
