"""Feature engineering, synthetic data, risk model and anomaly detector."""
import numpy as np
import pandas as pd

from ml import anomaly, risk_model
from ml.features import FEATURES, compute_features
from ml.synthetic_data import generate_training_set


def test_feature_values():
    st = ["present", "absent", "absent", "late", "present", "absent", "absent", "absent"]
    wd = [0, 2, 0, 2, 0, 2, 0, 2]
    f = compute_features(st, wd, planned_sessions=20, late_weight=1.0)
    assert f["pct_to_date"] == 100 * 3 / 8
    assert f["pct_last5"] == 100 * 2 / 5
    assert f["longest_absent_run"] == 3 and f["current_absent_run"] == 3
    assert f["late_ratio"] == 1 / 8
    assert f["sessions_remaining"] == 12
    assert f["max_achievable_pct"] == 100 * (3 + 12) / 20


def test_synthetic_generator_balanced_enough():
    df = generate_training_set(students_per_archetype=20, semesters=1, seed=1)
    assert set(FEATURES) <= set(df.columns)
    assert 0.2 < df["label"].mean() < 0.8
    assert set(df["archetype"]) == {"regular", "declining", "irregular", "chronic"}
    # Chronic students should be labelled at-risk far more often than regular ones.
    by = df.groupby("archetype")["label"].mean()
    assert by["chronic"] > 0.8 and by["regular"] < 0.3


def test_risk_model_beats_chance_and_explains(env, monkeypatch):
    monkeypatch.setattr("ml.risk_model.generate_training_set",
                        lambda seed=None: generate_training_set(students_per_archetype=40, semesters=1, seed=seed))
    m = risk_model.train()
    for k in ("random_forest", "logistic_regression"):
        assert m[k]["roc_auc"] > 0.8
        assert {"precision", "recall", "f1", "roc_auc"} <= set(m[k])
    bundle = risk_model.load()
    x = np.array(list(compute_features(["absent"] * 6 + ["present"] * 2, [0, 1] * 4, 40).values()))
    p = bundle["rf"].predict_proba(x[None])[0, 1]
    assert p > 0.7 and risk_model.risk_level(p) == "High"
    assert risk_model.explain(bundle, x)


def test_risk_levels():
    assert risk_model.risk_level(0.39) == "Low"
    assert risk_model.risk_level(0.4) == "Medium"
    assert risk_model.risk_level(0.7) == "Medium"
    assert risk_model.risk_level(0.71) == "High"


def _normal_records(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sess = i // 20
        rows.append({"id": i, "user_id": i % 20, "session_id": sess,
                     "date": "2026-09-01", "start_time": "09:00",
                     "check_in_time": f"2026-09-01T09:{int(rng.uniform(0, 12)):02d}:{int(rng.uniform(0, 59)):02d}",
                     "match_distance": rng.uniform(0.15, 0.35), "liveness_score": rng.uniform(0.85, 0.99),
                     "failed_attempts": int(rng.random() < 0.05)})
    return pd.DataFrame(rows)


def test_anomaly_detector_finds_injected_cases():
    df = _normal_records()
    injected = [
        {"id": 900, "user_id": 3, "session_id": 0, "date": "2026-09-01", "start_time": "09:00",
         "check_in_time": "2026-09-01T09:05:00", "match_distance": 0.58, "liveness_score": 0.62, "failed_attempts": 3},
        {"id": 901, "user_id": 4, "session_id": 1, "date": "2026-09-01", "start_time": "09:00",
         "check_in_time": "2026-09-01T09:29:00", "match_distance": 0.57, "liveness_score": 0.65, "failed_attempts": 2},
    ]
    feats = anomaly.record_features(pd.concat([df, pd.DataFrame(injected)], ignore_index=True))
    _, scores, flagged = anomaly.detect(feats, contamination=0.03)
    flagged_ids = set(feats.loc[flagged, "id"])
    assert {900, 901} <= flagged_ids
    expl = anomaly.explain_rows(feats, np.where(feats["id"] == 900)[0])[0]
    assert {"match_distance", "liveness_score", "failed_attempts"} & {e["feature"] for e in expl}


def test_anomaly_skipped_below_minimum(env):
    r = anomaly.run_detection(env["admin"])
    assert r["ran"] is False and "at least" in r["message"]
