"""Face-recognition evaluation: FAR / FRR / accuracy and threshold tuning.

    # your own volunteers: eval_data/faces/<person_name>/*.jpg
    python scripts/evaluate_recognition.py --data eval_data/faces

    # or the public LFW dataset (downloaded via scikit-learn, ~230 MB)
    python scripts/evaluate_recognition.py --lfw --per-person 20

Verification protocol: every genuine pair (same person) and an equal-sized
random sample of impostor pairs (different people) are scored with the
ArcFace cosine distance. The recommended threshold is the one with the lowest
FRR subject to FAR <= --target-far (default 1%, Section 10). A 1:N
identification test (gallery = first --gallery images per person, probes =
the rest) mirrors the kiosk.

Outputs go to eval_results/recognition_*.{json,csv,html}.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config  # noqa: E402
from vision.detector import FaceDetector  # noqa: E402
from vision.recognizer import get_embedder  # noqa: E402

OUT = ROOT / "eval_results"


def load_folder(path: Path, per_person: int) -> dict[str, list[np.ndarray]]:
    data = {}
    for d in sorted(p for p in path.iterdir() if p.is_dir()):
        imgs = [cv2.imread(str(f)) for f in sorted(d.iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        imgs = [i for i in imgs if i is not None][:per_person]
        if len(imgs) >= 2:
            data[d.name] = imgs
    return data


def load_lfw(per_person: int, min_faces: int) -> dict[str, list[np.ndarray]]:
    from sklearn.datasets import fetch_lfw_people
    print("Fetching LFW (first run downloads ~230 MB)...", flush=True)
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, color=True, resize=1.0, slice_=None,
                           data_home=str(ROOT / "eval_data" / "lfw"))
    data: dict[str, list[np.ndarray]] = {}
    for img, target in zip(lfw.images, lfw.target):
        name = lfw.target_names[target]
        if len(data.setdefault(name, [])) < per_person:
            rgb = img if img.max() > 1.5 else img * 255
            data[name].append(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return data


def embed_all(data: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    det, emb = FaceDetector(score_threshold=0.6), get_embedder()
    out, skipped = {}, 0
    total = sum(len(v) for v in data.values())
    done = 0
    for name, imgs in data.items():
        vecs = []
        for img in imgs:
            faces = det.detect(img)
            done += 1
            if not faces:
                skipped += 1
                continue
            # LFW images are centred on the subject: take the face nearest the centre.
            h, w = img.shape[:2]
            f = min(faces, key=lambda fc: (fc.box[0] + fc.box[2] / 2 - w / 2) ** 2 + (fc.box[1] + fc.box[3] / 2 - h / 2) ** 2)
            vecs.append(emb.embed(img, f.landmarks))
            if done % 100 == 0:
                print(f"  embedded {done}/{total}", flush=True)
        if len(vecs) >= 2:
            out[name] = vecs
    print(f"Embedded {sum(len(v) for v in out.values())} faces of {len(out)} people ({skipped} without a face).")
    return out


def verification(embs: dict[str, list[np.ndarray]], rng: np.random.Generator):
    genuine = [1 - float(a @ b) for v in embs.values() for a, b in itertools.combinations(v, 2)]
    names = list(embs)
    impostor = []
    while len(impostor) < max(len(genuine), 5000):
        n1, n2 = rng.choice(len(names), 2, replace=False)
        a = embs[names[n1]][rng.integers(len(embs[names[n1]]))]
        b = embs[names[n2]][rng.integers(len(embs[names[n2]]))]
        impostor.append(1 - float(a @ b))
    return np.array(genuine), np.array(impostor)


def rates(gen: np.ndarray, imp: np.ndarray, thr: float) -> dict:
    far = float((imp < thr).mean())
    frr = float((gen >= thr).mean())
    acc = float(((gen < thr).sum() + (imp >= thr).sum()) / (len(gen) + len(imp)))
    return {"threshold": round(thr, 4), "FAR": round(far, 5), "FRR": round(frr, 5), "accuracy": round(acc, 5)}


def identification(embs: dict[str, list[np.ndarray]], gallery_n: int, thr: float) -> dict:
    g_ids, g_vecs, probes = [], [], []
    for name, v in embs.items():
        if len(v) <= gallery_n:
            continue
        g_ids += [name] * gallery_n
        g_vecs += v[:gallery_n]
        probes += [(name, p) for p in v[gallery_n:]]
    if not probes:
        return {}
    G = np.stack(g_vecs)
    correct = wrong = rejected = 0
    for name, p in probes:
        d = 1 - G @ p
        i = int(np.argmin(d))
        if d[i] >= thr:
            rejected += 1
        elif g_ids[i] == name:
            correct += 1
        else:
            wrong += 1
    n = len(probes)
    return {"people": len(set(g_ids)), "probes": n, "gallery_per_person": gallery_n,
            "correct_rate": round(correct / n, 4), "misidentification_rate": round(wrong / n, 4),
            "rejection_rate": round(rejected / n, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="folder with one sub-folder of images per person")
    ap.add_argument("--lfw", action="store_true", help="use the public LFW dataset")
    ap.add_argument("--per-person", type=int, default=20)
    ap.add_argument("--min-faces", type=int, default=20)
    ap.add_argument("--gallery", type=int, default=5)
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.data and not args.lfw:
        ap.error("give --data FOLDER or --lfw")

    data = load_lfw(args.per_person, args.min_faces) if args.lfw else load_folder(args.data, args.per_person)
    embs = embed_all(data)
    gen, imp = verification(embs, np.random.default_rng(args.seed))

    grid = np.round(np.arange(0.20, 1.001, 0.005), 3)
    table = [rates(gen, imp, t) for t in grid]
    ok = [r for r in table if r["FAR"] <= args.target_far]
    # lowest FRR within the FAR budget; ties -> lowest FAR (safest threshold)
    best = min(ok, key=lambda r: (r["FRR"], r["FAR"])) if ok else min(table, key=lambda r: r["FAR"])
    eer = min(table, key=lambda r: abs(r["FAR"] - r["FRR"]))
    current = rates(gen, imp, float(config.get("recognition.match_threshold")))
    ident = identification(embs, args.gallery, best["threshold"])

    result = {
        "dataset": "LFW" if args.lfw else str(args.data), "model": config.get("recognition.model_name"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"), "people": len(embs),
        "images": int(sum(len(v) for v in embs.values())), "genuine_pairs": int(len(gen)),
        "impostor_pairs": int(len(imp)), "target_far": args.target_far,
        "recommended": best, "equal_error_rate": eer, "at_configured_threshold": current,
        "identification_1N_at_recommended": ident,
        "genuine_distance": {"mean": round(float(gen.mean()), 4), "p95": round(float(np.percentile(gen, 95)), 4)},
        "impostor_distance": {"mean": round(float(imp.mean()), 4), "p1": round(float(np.percentile(imp, 1)), 4)},
    }
    OUT.mkdir(exist_ok=True)
    tag = "lfw" if args.lfw else "custom"
    (OUT / f"recognition_{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(OUT / f"recognition_{tag}_thresholds.csv", "w", encoding="utf-8") as fh:
        fh.write("threshold,FAR,FRR,accuracy\n")
        for r in table:
            fh.write(f"{r['threshold']},{r['FAR']},{r['FRR']},{r['accuracy']}\n")
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_histogram(x=gen, name="Genuine pairs", opacity=0.7, marker_color="#2a78d6", histnorm="probability")
        fig.add_histogram(x=imp, name="Impostor pairs", opacity=0.7, marker_color="#eb6834", histnorm="probability")
        fig.add_vline(x=best["threshold"], line_dash="dash", annotation_text=f"threshold {best['threshold']}")
        fig.update_layout(barmode="overlay", title="ArcFace cosine distance distribution",
                          xaxis_title="Cosine distance", yaxis_title="Share of pairs")
        fig.write_html(OUT / f"recognition_{tag}_distances.html", include_plotlyjs="cdn")
    except Exception as e:  # plot is optional
        print("Plot skipped:", e)

    print(json.dumps({k: result[k] for k in ("people", "images", "genuine_pairs", "impostor_pairs", "recommended",
                                             "equal_error_rate", "at_configured_threshold",
                                             "identification_1N_at_recommended")}, indent=2))
    print(f"\nSet recognition.match_threshold = {best['threshold']} (Admin > System > Settings) "
          f"for FAR <= {args.target_far:.0%}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
