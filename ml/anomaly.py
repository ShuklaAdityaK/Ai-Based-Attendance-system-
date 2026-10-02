"""Attendance anomaly detection with Isolation Forest (Section 2.8).

Each kiosk record is described by:
    check_in_offset_min  minutes after the session start
    offset_zscore        deviation from the student's usual offset
    match_distance       ArcFace cosine distance (close to threshold = suspicious)
    liveness_score       passive anti-spoofing real-probability
    failed_attempts      failed attempts before the successful one
    gap_prev_seconds     time since the previous check-in at the kiosk (bursts)

Flags go to a review queue (Pending -> Confirmed / Dismissed). A flag never
changes the attendance status.
"""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core import config, db
from core.access import Actor, require_admin

FEATURES = ["check_in_offset_min", "offset_zscore", "match_distance", "liveness_score",
            "failed_attempts", "gap_prev_seconds"]
LABELS = {
    "check_in_offset_min": "Check-in offset (min)",
    "offset_zscore": "Unusual arrival time for this student",
    "match_distance": "Face match distance",
    "liveness_score": "Liveness score",
    "failed_attempts": "Failed attempts before success",
    "gap_prev_seconds": "Seconds since previous kiosk check-in",
}
# Which direction is suspicious for each feature (+1 high, -1 low, 0 both).
SUSPICIOUS_DIRECTION = {"check_in_offset_min": 0, "offset_zscore": 0, "match_distance": 1,
                        "liveness_score": -1, "failed_attempts": 1, "gap_prev_seconds": -1}
GAP_CAP = 600.0


def record_features(records: pd.DataFrame) -> pd.DataFrame:
    """Compute anomaly features from kiosk records (columns: id, user_id,
    session_id, check_in_time, date, start_time, match_distance,
    liveness_score, failed_attempts)."""
    if records.empty:
        return pd.DataFrame(columns=["id", *FEATURES])
    df = records.copy()
    df["check_in"] = pd.to_datetime(df["check_in_time"])
    df["start"] = pd.to_datetime(df["date"] + "T" + df["start_time"])
    df["check_in_offset_min"] = (df["check_in"] - df["start"]).dt.total_seconds() / 60
    # Leave-one-out mean/std of the student's own offsets.
    g = df.groupby("user_id")["check_in_offset_min"]
    n, s, ss = g.transform("count"), g.transform("sum"), g.transform(lambda v: (v ** 2).sum())
    x = df["check_in_offset_min"]
    loo_n = n - 1
    loo_mean = (s - x) / loo_n.where(loo_n > 0, np.nan)
    loo_var = ((ss - x ** 2) / loo_n.where(loo_n > 0, np.nan)) - loo_mean ** 2
    loo_std = np.sqrt(loo_var.clip(lower=0)).clip(lower=2.0)
    df["offset_zscore"] = ((x - loo_mean) / loo_std).where(loo_n >= 3, 0.0).fillna(0.0)
    df = df.sort_values(["session_id", "check_in"])
    df["gap_prev_seconds"] = (df.groupby("session_id")["check_in"].diff().dt.total_seconds()
                              .fillna(GAP_CAP).clip(upper=GAP_CAP))
    df["failed_attempts"] = df["failed_attempts"].fillna(0)
    return df[["id", *FEATURES]].reset_index(drop=True)


def _load_records() -> pd.DataFrame:
    with db.tx() as con:
        return pd.DataFrame(db.rows(
            con, "SELECT a.id, a.user_id, a.session_id, a.check_in_time, a.match_distance, a.liveness_score, "
                 "a.failed_attempts, s.date, s.start_time FROM attendance a JOIN sessions s ON s.id=a.session_id "
                 "WHERE a.source='kiosk' AND a.check_in_time IS NOT NULL AND a.match_distance IS NOT NULL"))


def model_matrix(feats: pd.DataFrame) -> np.ndarray:
    """Model input. The kiosk gap is log-scaled so that a 1-2 s burst stands
    out from normal 10-60 s queues (and the 600 s cap does not dominate)."""
    X = feats[FEATURES].to_numpy(float).copy()
    g = FEATURES.index("gap_prev_seconds")
    X[:, g] = np.log1p(X[:, g])
    return X


def explain_rows(feats: pd.DataFrame, flagged_idx: np.ndarray, top: int = 3) -> list[list[dict]]:
    """Main contributing features: robust z-score against all records, counted
    only in the suspicious direction."""
    X = model_matrix(feats)
    raw = feats[FEATURES].to_numpy(float)
    med = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    mad = np.where(iqr > 1e-6, iqr / 1.349, X.std(axis=0) + 1e-6)
    out = []
    for i in flagged_idx:
        z = (X[i] - med) / mad
        contrib = []
        for j, f in enumerate(FEATURES):
            d = SUSPICIOUS_DIRECTION[f]
            score = abs(z[j]) if d == 0 else max(0.0, d * z[j])
            contrib.append((score, f, raw[i, j]))
        contrib.sort(reverse=True)
        out.append([{"feature": f, "label": LABELS[f], "value": round(float(v), 3),
                     "typical": round(float(np.median(raw[:, FEATURES.index(f)])), 3), "z": round(float(s), 2)}
                    for s, f, v in contrib[:top] if s > 0.5])
    return out


def detect(feats: pd.DataFrame, contamination: float | None = None, seed: int = 42):
    """Fit Isolation Forest; returns (model, anomaly_scores, is_flagged)."""
    c = float(config.get("ml.anomaly_contamination", 0.05) if contamination is None else contamination)
    X = model_matrix(feats)
    model = IsolationForest(n_estimators=300, contamination=c, random_state=seed)
    model.fit(X)
    scores = -model.score_samples(X)           # higher = more anomalous
    return model, scores, model.predict(X) == -1


def run_detection(actor: Actor) -> dict:
    require_admin(actor)
    min_records = int(config.get("ml.anomaly_min_records", 30))
    records = _load_records()
    if len(records) < min_records:
        return {"ran": False, "records": len(records),
                "message": f"Anomaly detection needs at least {min_records} kiosk records "
                           f"({len(records)} so far). Skipped."}
    feats = record_features(records)
    model, scores, flagged = detect(feats)
    idx = np.where(flagged)[0]
    explanations = explain_rows(feats, idx)
    version = "anomaly-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    d = config.path("trained_models")
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "version": version}, d / "anomaly_model.joblib")
    new = 0
    with db.tx() as con:
        for i, expl in zip(idx, explanations):
            cur = con.execute(
                "INSERT OR IGNORE INTO anomaly_flags(attendance_id, score, top_features, model_version, created_at) "
                "VALUES (?,?,?,?,?)", (int(feats.loc[i, "id"]), float(scores[i]), json.dumps(expl), version, db.now()))
            new += cur.rowcount
        db.log_admin_action(con, actor, "run_anomaly_detection", version,
                            {"records": len(feats), "flagged": int(len(idx)), "new_flags": new})
    summary = {"ran": True, "records": len(feats), "flagged": int(len(idx)), "new_flags": new, "version": version,
               "message": f"Analysed {len(feats)} kiosk records: {len(idx)} flagged ({new} new)."}
    (d / "anomaly_last_run.json").write_text(json.dumps({**summary, "at": db.now()}, indent=2), encoding="utf-8")
    return summary


def last_run() -> dict | None:
    f = config.path("trained_models") / "anomaly_last_run.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def list_flags(actor: Actor, status: str | None = None) -> pd.DataFrame:
    require_admin(actor)
    sql = ("SELECT f.*, a.session_id, a.user_id, a.status AS attendance_status, a.check_in_time, a.match_distance, "
           "a.liveness_score, a.failed_attempts, u.full_name, u.roll_no, sub.code, s.date, rv.username AS reviewer "
           "FROM anomaly_flags f JOIN attendance a ON a.id=f.attendance_id JOIN users u ON u.id=a.user_id "
           "JOIN sessions s ON s.id=a.session_id JOIN subjects sub ON sub.id=s.subject_id "
           "LEFT JOIN users rv ON rv.id=f.reviewed_by")
    params: list = []
    if status:
        sql += " WHERE f.review_status=?"
        params.append(status)
    with db.tx() as con:
        df = pd.DataFrame(db.rows(con, sql + " ORDER BY f.score DESC", params))
    if not df.empty:
        df["top_features"] = df["top_features"].map(json.loads)
    return df


def review_flag(actor: Actor, flag_id: int, decision: str, notes: str = "") -> None:
    require_admin(actor)
    if decision not in ("confirmed", "dismissed", "pending"):
        raise ValueError("Invalid review decision.")
    with db.tx() as con:
        con.execute("UPDATE anomaly_flags SET review_status=?, reviewed_by=?, reviewed_at=?, notes=? WHERE id=?",
                    (decision, actor.id, db.now(), notes.strip() or None, flag_id))
        db.log_admin_action(con, actor, "review_anomaly", f"flag:{flag_id}", {"decision": decision, "notes": notes})
