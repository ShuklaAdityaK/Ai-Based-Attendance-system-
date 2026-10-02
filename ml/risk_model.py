"""Attendance risk prediction (Section 2.7).

Random Forest classifier with a Logistic Regression baseline. Predicts, per
student per subject, the probability that the final attendance will fall
below the required threshold. Students with fewer than
``ml.min_sessions_for_model`` sessions get a rule-based estimate instead.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core import config, db
from core.access import Actor, require_admin, scope_user_id
from ml.features import FEATURE_LABELS, FEATURES, completed_real_samples, feature_frame
from ml.synthetic_data import generate_training_set

MODEL_FILE = "risk_model.joblib"
METRICS_FILE = "risk_metrics.json"
_cache: dict = {"mtime": None, "bundle": None}
_lock = threading.Lock()


def _dir() -> Path:
    d = config.path("trained_models")
    d.mkdir(parents=True, exist_ok=True)
    return d


def risk_level(p: float) -> str:
    lo, hi = float(config.get("ml.risk_low", 0.4)), float(config.get("ml.risk_high", 0.7))
    return "Low" if p < lo else ("Medium" if p <= hi else "High")


def _metrics(y_true, prob, thr: float = 0.5) -> dict:
    pred = (prob >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y_true, prob)
    idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 60)).astype(int)
    return {
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "roc_curve": {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()},
    }


def train(actor: Actor | None = None, include_real: bool = True, seed: int | None = None) -> dict:
    """Train RF + LR on synthetic (+ completed real) data with a held-out,
    student-grouped split. Saves the model bundle and returns the metrics."""
    if actor is not None:
        require_admin(actor)
    seed = int(config.get("ml.random_seed", 42) if seed is None else seed)
    data = generate_training_set(seed=seed)
    n_real = 0
    if include_real and actor is not None:
        real = completed_real_samples(actor)
        n_real = len(real)
        if n_real:
            data = pd.concat([data, real], ignore_index=True)
    X, y, groups = data[FEATURES].to_numpy(float), data["label"].to_numpy(int), data["group"].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))

    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced",
                                n_jobs=-1, random_state=seed)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    rf.fit(X[tr], y[tr])
    lr.fit(X[tr], y[tr])
    m_rf = _metrics(y[te], rf.predict_proba(X[te])[:, 1])
    m_lr = _metrics(y[te], lr.predict_proba(X[te])[:, 1])

    # Reference point for explanations: median profile of "not at risk" samples.
    reference = np.median(X[tr][y[tr] == 0], axis=0)
    version = "risk-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    importances = dict(zip(FEATURES, np.round(rf.feature_importances_, 4).tolist()))
    metrics = {
        "version": version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_samples": int(len(tr)), "test_samples": int(len(te)),
        "synthetic_samples": int((data["source"] == "synthetic").sum()), "real_samples": int(n_real),
        "positive_rate": round(float(y.mean()), 4),
        "random_forest": m_rf, "logistic_regression": m_lr,
        "feature_importance": importances,
        "split": "GroupShuffleSplit by student (25% held out)",
    }
    bundle = {"rf": rf, "lr": lr, "features": FEATURES, "reference": reference, "version": version,
              "metrics": metrics}
    joblib.dump(bundle, _dir() / MODEL_FILE)
    (_dir() / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if actor is not None:
        with db.tx() as con:
            db.log_admin_action(con, actor, "retrain_risk_model", version,
                                {"rf_f1": m_rf["f1"], "rf_auc": m_rf["roc_auc"], "lr_auc": m_lr["roc_auc"]})
    config.log.info("Risk model trained %s RF AUC=%.3f LR AUC=%.3f", version, m_rf["roc_auc"], m_lr["roc_auc"])
    return metrics


def load() -> dict | None:
    f = _dir() / MODEL_FILE
    if not f.exists():
        return None
    with _lock:
        mtime = f.stat().st_mtime
        if _cache["mtime"] != mtime:
            _cache["bundle"], _cache["mtime"] = joblib.load(f), mtime
        return _cache["bundle"]


def latest_metrics() -> dict | None:
    f = _dir() / METRICS_FILE
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _fmt(feature: str, value: float) -> str:
    if feature.startswith("pct") or feature == "max_achievable_pct":
        return f"{value:.0f}%"
    if feature in ("late_ratio", "dow_absence_concentration"):
        return f"{value * 100:.0f}%"
    return f"{value:.0f}"


def explain(bundle: dict, x: np.ndarray, top: int = 3) -> list[dict]:
    """Local attribution: how much the risk drops when each feature is set to
    the typical value of a not-at-risk student (perturbation analysis)."""
    rf = bundle["rf"]
    base = rf.predict_proba(x[None])[0, 1]
    variants = np.repeat(x[None], len(FEATURES), axis=0)
    for i in range(len(FEATURES)):
        variants[i, i] = bundle["reference"][i]
    deltas = base - rf.predict_proba(variants)[:, 1]
    order = np.argsort(-deltas) if base >= float(config.get("ml.risk_low", 0.4)) else np.argsort(deltas)
    out = []
    for i in order[:top]:
        if abs(deltas[i]) < 0.01:
            continue
        f = FEATURES[i]
        out.append({"feature": f, "label": FEATURE_LABELS[f], "value": _fmt(f, x[i]),
                    "typical": _fmt(f, bundle["reference"][i]), "effect": round(float(deltas[i]), 3),
                    "direction": "raises risk" if deltas[i] > 0 else "lowers risk"})
    return out


def rule_based(row: pd.Series) -> tuple[float, list[dict]]:
    """Estimate for students with too little history for the model."""
    thr = float(config.get("attendance.required_percentage", 75))
    n = int(row["n_sessions"])
    if row["max_achievable_pct"] < thr:
        p = 0.99
    elif n == 0:
        p = 0.2
    else:
        p = float(np.clip((thr - row["pct_to_date"]) / thr * 1.5 + 0.3, 0.05, 0.95))
    factors = [{"feature": "n_sessions", "label": "Too few sessions for the ML model", "value": str(n),
                "typical": f">= {int(config.get('ml.min_sessions_for_model', 5))}", "effect": 0.0,
                "direction": "rule-based estimate"},
               {"feature": "pct_to_date", "label": FEATURE_LABELS["pct_to_date"],
                "value": _fmt("pct_to_date", row["pct_to_date"]), "typical": f">= {thr:.0f}%", "effect": 0.0,
                "direction": "raises risk" if row["pct_to_date"] < thr else "lowers risk"}]
    return p, factors


def predict_frame(actor: Actor, user_id: int | None = None) -> pd.DataFrame:
    """Risk for every visible (student, subject) pair - students see only their own."""
    scope_user_id(actor, user_id)
    df = feature_frame(actor, user_id)
    if df.empty:
        return df
    bundle = load()
    min_n = int(config.get("ml.min_sessions_for_model", 5))
    probs, levels, methods, factors = [], [], [], []
    for _, r in df.iterrows():
        if bundle is not None and r["n_sessions"] >= min_n:
            x = r[FEATURES].to_numpy(float)
            p = float(bundle["rf"].predict_proba(x[None])[0, 1])
            fac, method = explain(bundle, x), "model"
        else:
            p, fac = rule_based(r)
            method = "rule"
        probs.append(round(p, 4))
        levels.append(risk_level(p))
        methods.append(method)
        factors.append(fac)
    df["probability"], df["level"], df["method"], df["top_factors"] = probs, levels, methods, factors
    df["model_version"] = bundle["version"] if bundle else None
    return df


def refresh_predictions(actor: Actor) -> int:
    """Recompute and store predictions for all students (admin)."""
    require_admin(actor)
    df = predict_frame(actor)
    ts = db.now()
    with db.tx() as con:
        con.execute("DELETE FROM risk_predictions")
        con.executemany(
            "INSERT INTO risk_predictions(user_id, subject_id, probability, level, method, top_factors, "
            "model_version, generated_at) VALUES (?,?,?,?,?,?,?,?)",
            [(int(r.user_id), int(r.subject_id), float(r.probability), r.level, r.method,
              json.dumps(r.top_factors), r.model_version, ts) for r in df.itertuples()])
    return len(df)


def stored_predictions(actor: Actor, user_id: int | None = None) -> pd.DataFrame:
    uid = scope_user_id(actor, user_id)
    sql = ("SELECT r.*, u.full_name, u.roll_no, s.code, s.name AS subject_name FROM risk_predictions r "
           "JOIN users u ON u.id=r.user_id JOIN subjects s ON s.id=r.subject_id")
    params: list = []
    if uid is not None:
        sql += " WHERE r.user_id=?"
        params.append(uid)
    with db.tx() as con:
        df = pd.DataFrame(db.rows(con, sql + " ORDER BY r.probability DESC", params))
    if not df.empty:
        df["top_factors"] = df["top_factors"].map(json.loads)
    return df
