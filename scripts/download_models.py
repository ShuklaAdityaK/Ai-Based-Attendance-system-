"""Download the pretrained models into ./models.

    python scripts/download_models.py

* YuNet face detector (OpenCV Zoo)
* ArcFace R50 (w600k_r50.onnx from InsightFace's buffalo_l pack)
* MiniFASNetV2 / MiniFASNetV1SE (Silent-Face-Anti-Spoofing, ONNX export)
* MediaPipe Face Landmarker
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

FILES = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "MiniFASNetV2.onnx": "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
    "MiniFASNetV1SE.onnx": "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ARCFACE = "arcface_w600k_r50.onnx"


def _get(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    for name, url in FILES.items():
        if not (MODELS / name).exists():
            _get(url, MODELS / name)
    if not (MODELS / ARCFACE).exists():
        z = MODELS / "buffalo_l.zip"
        _get(BUFFALO_URL, z)
        with zipfile.ZipFile(z) as zf:
            member = next(n for n in zf.namelist() if n.endswith("w600k_r50.onnx"))
            (MODELS / ARCFACE).write_bytes(zf.read(member))
        z.unlink()
    print("All models present:", ", ".join(sorted(p.name for p in MODELS.glob("*.*") if p.is_file())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
