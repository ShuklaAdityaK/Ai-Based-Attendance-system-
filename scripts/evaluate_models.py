"""ML evaluation for the project report.

    python scripts/evaluate_models.py

1. Risk model: Random Forest vs Logistic Regression baseline on held-out
   students (precision, recall, F1, ROC-AUC), plus a per-archetype breakdown.
2. Anomaly detector: realistic kiosk records with injected anomalies of four
   kinds; reports the detection rate per kind and the false-flag rate.
Results: eval_results/ml_evaluation.json
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SA_DB_PATH", str(Path(tempfile.gettempdir()) / "sa_eval.db"))

from core import config  # noqa: E402
from ml import anomaly, risk_model  # noqa: E402
from ml.features import FEATURES  # noqa: E402
from ml.synthetic_data import generate_training_set  # noqa: E402


def risk_eval() -> dict:
    m = risk_model.train(None, include_real=False)
    bundle = risk_model.load()
    test = generate_training_set(students_per_archetype=60, semesters=1, seed=999)  # fresh, unseen semester
    p = bundle["rf"].predict_proba(test[FEATURES].to_numpy(float))[:, 1]
    test["pred"] = (p >= 0.5).astype(int)
    per_arch = (test.groupby("archetype").apply(lambda g: pd.Series({
        "samples": len(g), "actual_at_risk_rate": round(g["label"].mean(), 3),
        "predicted_at_risk_rate": round(g["pred"].mean(), 3),
        "accuracy": round((g["pred"] == g["label"]).mean(), 3)}), include_groups=False).to_dict(orient="index"))
    keys = ("precision", "recall", "f1", "roc_auc", "accuracy")
    return {"version": m["version"], "train_samples": m["train_samples"], "test_samples": m["test_samples"],
            "random_forest": {k: m["random_forest"][k] for k in keys},
            "logistic_regression": {k: m["logistic_regression"][k] for k in keys},
            "fresh_semester_by_archetype": per_arch}


def kiosk_records(rng: np.random.Generator, n_sessions=60, per_session=30):
    rows, rid = [], 0
    base = datetime(2026, 8, 3, 9, 0)
    for s in range(n_sessions):
        start = base + timedelta(days=s)
        t = start - timedelta(minutes=6)
        for u in rng.permutation(per_session):
            t += timedelta(seconds=float(rng.uniform(10, 45)))
            rows.append({"id": rid, "user_id": int(u), "session_id": s, "date": start.date().isoformat(),
                         "start_time": "09:00", "check_in_time": t.isoformat(timespec="seconds"),
                         "match_distance": float(np.clip(rng.normal(0.30, 0.06), 0.12, 0.5)),
                         "liveness_score": float(np.clip(rng.beta(18, 1.5), 0.62, 0.999)),
                         "failed_attempts": int(rng.random() < 0.08), "kind": "normal"})
            rid += 1
    return pd.DataFrame(rows), rid


def anomaly_eval(seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    df, rid = kiosk_records(rng)
    inject = []
    for i in range(10):     # borderline match + retries (possible look-alike / proxy)
        r = df.sample(1, random_state=seed + i).iloc[0].to_dict()
        inject.append({**r, "id": rid, "match_distance": rng.uniform(0.54, 0.59), "failed_attempts": int(rng.integers(2, 5)),
                       "kind": "borderline_match"}); rid += 1
    for i in range(10):     # weak liveness that still passed
        r = df.sample(1, random_state=100 + i).iloc[0].to_dict()
        inject.append({**r, "id": rid, "liveness_score": rng.uniform(0.60, 0.68), "failed_attempts": int(rng.integers(1, 4)),
                       "kind": "weak_liveness"}); rid += 1
    for i in range(5):      # burst: 3 check-ins within seconds
        r = df.sample(1, random_state=200 + i).iloc[0].to_dict()
        t0 = datetime.fromisoformat(r["check_in_time"])
        for j in range(3):
            inject.append({**r, "id": rid, "user_id": 900 + 3 * i + j,
                           "check_in_time": (t0 + timedelta(seconds=1 + j)).isoformat(timespec="seconds"),
                           "match_distance": rng.uniform(0.42, 0.55), "kind": "burst"}); rid += 1
    for i in range(10):     # unusual arrival for a habitually early student
        u = int(rng.integers(30))
        hist = df[df["user_id"] == u].iloc[0].to_dict()
        start = datetime.fromisoformat(f"{hist['date']}T09:00")
        inject.append({**hist, "id": rid, "check_in_time": (start + timedelta(minutes=29)).isoformat(timespec="seconds"),
                       "match_distance": rng.uniform(0.45, 0.55), "kind": "odd_time"}); rid += 1
    all_df = pd.concat([df, pd.DataFrame(inject)], ignore_index=True)
    feats = anomaly.record_features(all_df.drop(columns=["kind"]))
    kinds = all_df.set_index("id")["kind"]
    _, _, flagged = anomaly.detect(feats, contamination=float(config.get("ml.anomaly_contamination", 0.05)))
    feats["kind"] = feats["id"].map(kinds)
    feats["flagged"] = flagged
    res = {k: round(float(g["flagged"].mean()), 3) for k, g in feats[feats["kind"] != "normal"].groupby("kind")}
    normal = feats[feats["kind"] == "normal"]
    injected = feats[feats["kind"] != "normal"]
    return {"normal_records": len(normal), "injected_records": len(injected),
            "contamination": config.get("ml.anomaly_contamination"),
            "detection_rate_by_kind": res, "detection_rate_overall": round(float(injected["flagged"].mean()), 3),
            "false_flag_rate_normal": round(float(normal["flagged"].mean()), 4)}


def main() -> int:
    tmp_models = Path(tempfile.mkdtemp(prefix="sa_eval_models_"))
    real_path = config.path
    config.path = lambda key: tmp_models if key == "trained_models" else real_path(key)  # keep app models untouched
    print("Training & evaluating the risk model...")
    risk = risk_eval()
    print(json.dumps({k: risk[k] for k in ("random_forest", "logistic_regression")}, indent=2))
    print("Evaluating the anomaly detector on injected anomalies...")
    anom = anomaly_eval()
    print(json.dumps(anom, indent=2))
    out = ROOT / "eval_results"
    out.mkdir(exist_ok=True)
    (out / "ml_evaluation.json").write_text(json.dumps({"risk_model": risk, "anomaly_detector": anom,
                                                        "evaluated_at": datetime.now().isoformat(timespec="seconds")},
                                                       indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
