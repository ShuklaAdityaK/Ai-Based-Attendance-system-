"""Record webcam clips for the liveness evaluation (Section 12: >= 20 attempts each).

    python scripts/capture_liveness_samples.py

Keys:  r = record a REAL-face clip   p = PRINTED-photo attack
       s = phone/SCREEN replay attack q = quit
Each clip is 3 s and is saved to eval_data/liveness/<real|print|replay>/.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval_data" / "liveness"
KEYS = {ord("r"): "real", ord("p"): "print", ord("s"): "replay"}


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera found.")
        return 1
    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        counts = {k: len(list((OUT / k).glob("*.mp4"))) for k in KEYS.values()}
        cv2.putText(frame, "  ".join(f"{k}:{v}" for k, v in counts.items()), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("liveness capture (r/p/s, q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in KEYS:
            kind = KEYS[key]
            (OUT / kind).mkdir(parents=True, exist_ok=True)
            path = OUT / kind / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            h, w = frame.shape[:2]
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            end = time.time() + 3
            while time.time() < end:
                ok, frame = cap.read()
                if not ok:
                    break
                vw.write(frame)
                cv2.imshow("liveness capture (r/p/s, q to quit)", frame)
                cv2.waitKey(1)
            vw.release()
            print("saved", path)
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
