"""Download and verify the pretrained models in ./models.

    python scripts/download_models.py           # download missing files, verify all
    python scripts/download_models.py --verify  # only verify what is on disk

* YuNet face detector (OpenCV Zoo)
* ArcFace R50 (w600k_r50.onnx from InsightFace's buffalo_l pack)
* MiniFASNetV2 / MiniFASNetV1SE (Silent-Face-Anti-Spoofing, ONNX export)
* MediaPipe Face Landmarker

Every file is checked against a pinned SHA-256 hash, so a corrupted or
tampered download is rejected instead of being loaded by the app.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# name -> (url, sha256)
FILES = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "MiniFASNetV2.onnx": (
        "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
        "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907"),
    "MiniFASNetV1SE.onnx": (
        "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx",
        "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676"),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
}
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ARCFACE = ("arcface_w600k_r50.onnx", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _check(path: Path, expected: str) -> bool:
    if not path.exists():
        print(f"  MISSING  {path.name}")
        return False
    ok = sha256(path) == expected
    print(f"  {'OK      ' if ok else 'MISMATCH'} {path.name}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only verify files already on disk")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)

    if not args.verify:
        for name, (url, expected) in FILES.items():
            path = MODELS / name
            if not path.exists():
                _get(url, path)
                if sha256(path) != expected:
                    path.unlink()
                    print(f"Checksum mismatch for {name} - download removed. Try again or check the URL.")
                    return 1
        name, expected = ARCFACE
        if not (MODELS / name).exists():
            z = MODELS / "buffalo_l.zip"
            _get(BUFFALO_URL, z)
            with zipfile.ZipFile(z) as zf:
                member = next(n for n in zf.namelist() if n.endswith("w600k_r50.onnx"))
                (MODELS / name).write_bytes(zf.read(member))
            z.unlink()
            if sha256(MODELS / name) != expected:
                (MODELS / name).unlink()
                print(f"Checksum mismatch for {name} - file removed.")
                return 1

    print("Verifying model files:")
    results = [_check(MODELS / n, h) for n, (_, h) in FILES.items()] + [_check(MODELS / ARCFACE[0], ARCFACE[1])]
    if not all(results):
        print("Some model files are missing or corrupted. Delete them and run this script again.")
        return 1
    print("All models present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
