"""Synthetic attendance generator for the cold-start problem (Section 2.7).

A fresh deployment has no past semesters to learn from, so whole semesters
are simulated from four student archetypes:

* regular   - consistently high attendance, occasional lateness
* declining - starts well, then attendance decays part-way through
* irregular - medium attendance, absences clustered on one weekday and in
              short "sick" streaks
* chronic   - persistently low attendance

The risk model is trained on samples taken at checkpoints within each
simulated semester (features from the first t sessions, label = whether the
*final* attendance fell below the threshold). This is stated openly in the
project report: real data replaces/augments it as semesters complete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import attendance as att_rules
from core import config
from ml.features import compute_features

ARCHETYPES = ("regular", "declining", "irregular", "chronic")


def _attend_prob(arch: str, rng: np.random.Generator, T: int) -> tuple[np.ndarray, float, int | None]:
    """Per-session attendance probability, late fraction, and a 'bad weekday'."""
    t = np.arange(T)
    bad_day = None
    if arch == "regular":
        p = np.full(T, rng.uniform(0.80, 0.98))
        late = rng.uniform(0.0, 0.10)
    elif arch == "declining":
        start, end = rng.uniform(0.80, 0.97), rng.uniform(0.30, 0.70)
        knee = int(T * rng.uniform(0.15, 0.55))
        p = np.where(t < knee, start, start + (end - start) * (t - knee) / max(1, T - 1 - knee))
        late = rng.uniform(0.03, 0.18)
    elif arch == "irregular":
        p = np.full(T, rng.uniform(0.62, 0.90))
        bad_day = int(rng.integers(0, 5))
        late = rng.uniform(0.05, 0.25)
    else:  # chronic
        p = np.full(T, rng.uniform(0.30, 0.68))
        late = rng.uniform(0.05, 0.30)
    p = np.clip(p + rng.normal(0, 0.03, T), 0.02, 0.99)
    return p, late, bad_day


def simulate_history(arch: str, rng: np.random.Generator, T: int) -> tuple[list[str], list[int]]:
    """One simulated semester of a subject for one student."""
    p, late_frac, bad_day = _attend_prob(arch, rng, T)
    days = sorted(rng.choice(5, size=int(rng.integers(2, 4)), replace=False).tolist())
    weekdays = [days[i % len(days)] for i in range(T)]
    statuses: list[str] = []
    sick_left = 0
    for i in range(T):
        pi = p[i]
        if bad_day is not None and weekdays[i] == bad_day:
            pi *= rng.uniform(0.35, 0.7)
        if arch == "irregular" and sick_left == 0 and rng.random() < 0.04:
            sick_left = int(rng.integers(2, 5))
        if sick_left > 0:
            sick_left -= 1
            statuses.append("absent")
            continue
        if rng.random() < pi:
            statuses.append("late" if rng.random() < late_frac else "present")
        else:
            statuses.append("absent")
    return statuses, weekdays


def generate_training_set(students_per_archetype: int | None = None, semesters: int | None = None,
                          seed: int | None = None, checkpoint_every: int = 3) -> pd.DataFrame:
    n = int(students_per_archetype or config.get("ml.synthetic_students_per_archetype", 120))
    sems = int(semesters or config.get("ml.synthetic_semesters", 3))
    rng = np.random.default_rng(config.get("ml.random_seed", 42) if seed is None else seed)
    thr = float(config.get("attendance.required_percentage", 75))
    w = float(config.get("attendance.late_weight", 1.0))
    min_n = int(config.get("ml.min_sessions_for_model", 5))
    rows = []
    for sem in range(sems):
        for arch in ARCHETYPES:
            for s in range(n):
                T = int(rng.integers(30, 46))
                statuses, weekdays = simulate_history(arch, rng, T)
                final = att_rules.attendance_percentage(statuses.count("present"), statuses.count("late"), T, w)
                group = f"syn-{sem}-{arch}-{s}"
                start = int(rng.integers(min_n, min_n + checkpoint_every))
                for t in range(start, T, checkpoint_every):
                    f = compute_features(statuses[:t], weekdays[:t], T, w)
                    rows.append({**f, "label": int(final < thr), "group": group, "archetype": arch,
                                 "source": "synthetic", "final_pct": final})
    return pd.DataFrame(rows)
