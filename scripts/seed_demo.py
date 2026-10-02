"""Load DEMONSTRATION data so the dashboards, reports and ML modules can be
shown before real semesters exist.

    python scripts/seed_demo.py [--students 30] [--weeks 10] [--password demo1234]

Requires that the first administrator was already created in the app.
Creates subjects, demo student accounts (no face data), closed sessions with
kiosk-like attendance generated from the same archetypes as the synthetic
generator, a few injected anomalies, and one open session for today. Then
trains the risk model, stores predictions and runs anomaly detection.
All demo usernames start with "demo_" and every record is clearly synthetic.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import attendance, auth, config, db  # noqa: E402
from core.access import Actor  # noqa: E402
from ml import anomaly, risk_model  # noqa: E402
from ml.synthetic_data import ARCHETYPES, simulate_history  # noqa: E402

SUBJECTS = [("KCS-701", "Machine Learning", [0, 2, 4], "09:00"),
            ("KCS-702", "Cloud Computing", [1, 3], "10:00"),
            ("KCS-703", "Compiler Design", [0, 3], "11:30"),
            ("KCS-704", "Cryptography & Network Security", [1, 4], "14:00")]
FIRST = ["Aarav", "Vivaan", "Aditya", "Ananya", "Diya", "Ishaan", "Kavya", "Krishna", "Meera", "Nikhil", "Pooja",
         "Priya", "Rahul", "Riya", "Rohan", "Sakshi", "Sanjana", "Shivam", "Shreya", "Siddharth", "Sneha", "Tanvi",
         "Utkarsh", "Vaishnavi", "Yash", "Aman", "Neha", "Harsh", "Divya", "Saurabh", "Anjali", "Kunal"]
LAST = ["Sharma", "Verma", "Gupta", "Mishra", "Pandey", "Tiwari", "Srivastava", "Yadav", "Singh", "Shukla",
        "Tripathi", "Dubey", "Maurya", "Patel", "Kesarwani", "Chaurasia"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=30)
    ap.add_argument("--weeks", type=int, default=10)
    ap.add_argument("--password", default="demo1234")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    db.init_db()
    with db.tx() as con:
        a = db.one(con, "SELECT id, username, full_name FROM users WHERE role='admin' ORDER BY id LIMIT 1")
        if con.execute("SELECT 1 FROM users WHERE username LIKE 'demo\\_%' ESCAPE '\\'").fetchone():
            print("Demo data already present - nothing to do.")
            return 0
    if not a:
        print("Create the first administrator in the app (streamlit run app.py) before seeding.")
        return 1
    admin = Actor(a["id"], a["username"], "admin", a["full_name"])

    # Students
    student_ids, arch_of = [], {}
    for i in range(args.students):
        name = f"{FIRST[i % len(FIRST)]} {LAST[int(rng.integers(len(LAST)))]}"
        uid = auth.create_user_by_admin(admin, f"demo_{i + 1:02d}", name, f"DEMO{2201640100 + i + 1}",
                                        f"demo{i + 1}@example.edu", args.password, "student")
        student_ids.append(uid)
        arch_of[uid] = ARCHETYPES[int(rng.choice(4, p=[0.45, 0.2, 0.2, 0.15]))]
    with db.tx() as con:
        con.execute(f"UPDATE users SET must_change_password=0 WHERE id IN ({','.join('?' * len(student_ids))})",
                    student_ids)

    # Subjects + enrollment (each student takes 3-4 subjects)
    subj_ids = []
    for code, name, _, _ in SUBJECTS:
        existing = [s for s in db.list_subjects(admin) if s["code"] == code]
        sid = existing[0]["id"] if existing else db.create_subject(admin, code, name, 40)
        subj_ids.append(sid)
        enrolled = [u for u in student_ids if rng.random() < 0.9]
        db.set_subject_enrollment(admin, sid, enrolled)

    # Sessions + records, written in one transaction for speed
    today = date.today()
    start_day = today - timedelta(weeks=args.weeks)
    grace, cutoff = int(config.get("attendance.grace_minutes")), int(config.get("attendance.late_cutoff_minutes"))
    n_sessions = n_records = 0
    with db.tx() as con:
        for (code, _, days, start), sid in zip(SUBJECTS, subj_ids):
            dates = [start_day + timedelta(days=d) for d in range((today - start_day).days)
                     if (start_day + timedelta(days=d)).weekday() in days]
            enrolled = [r[0] for r in con.execute("SELECT user_id FROM enrollments WHERE subject_id=?", (sid,))]
            histories = {u: simulate_history(arch_of[u], rng, len(dates))[0] for u in enrolled}
            for k, d in enumerate(dates):
                st = datetime.fromisoformat(f"{d}T{start}")
                cur = con.execute(
                    "INSERT INTO sessions(subject_id, date, start_time, grace_minutes, late_cutoff_minutes, status, "
                    "created_by, opened_by, opened_at, closed_at) VALUES (?,?,?,?,?,'closed',?,?,?,?)",
                    (sid, d.isoformat(), start, grace, cutoff, admin.id, admin.id,
                     (st - timedelta(minutes=10)).isoformat(), (st + timedelta(minutes=cutoff + 5)).isoformat()))
                sess = cur.lastrowid
                n_sessions += 1
                # check-ins arrive in a queue at the kiosk
                arrivals = []
                for u in enrolled:
                    s = histories[u][k]
                    if s == "absent":
                        con.execute("INSERT INTO attendance(session_id, user_id, status, source, created_at) "
                                    "VALUES (?,?,'absent','system',?)", (sess, u, st.isoformat()))
                        continue
                    off = rng.uniform(-6, grace) if s == "present" else rng.uniform(grace + 0.2, cutoff)
                    arrivals.append((st + timedelta(minutes=float(off)), u, s))
                arrivals.sort()
                last_t = None
                for t, u, s in arrivals:
                    if last_t and (t - last_t).total_seconds() < 8:
                        t = last_t + timedelta(seconds=float(rng.uniform(8, 20)))  # one person at a time
                    last_t = t
                    status = attendance.classify_check_in(t, st, grace, cutoff) or "late"
                    con.execute(
                        "INSERT INTO attendance(session_id, user_id, status, check_in_time, match_distance, "
                        "liveness_score, failed_attempts, source, created_at) VALUES (?,?,?,?,?,?,?,'kiosk',?)",
                        (sess, u, status, t.isoformat(timespec="seconds"),
                         float(np.clip(rng.normal(0.30, 0.06), 0.12, 0.5)),
                         float(np.clip(rng.beta(18, 1.5), 0.62, 0.999)),
                         int(rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])), t.isoformat(timespec="seconds")))
                    n_records += 1
        # Injected anomalies: borderline match + weak liveness + retries, and a burst of check-ins
        kiosk_ids = [r[0] for r in con.execute("SELECT id FROM attendance WHERE source='kiosk' ORDER BY RANDOM() LIMIT 6")]
        for j, aid in enumerate(kiosk_ids):
            con.execute("UPDATE attendance SET match_distance=?, liveness_score=?, failed_attempts=? WHERE id=?",
                        (float(rng.uniform(0.55, 0.59)), float(rng.uniform(0.6, 0.7)), int(rng.integers(2, 5)), aid))
        db.log_admin_action(con, admin, "seed_demo_data", "demo",
                            {"students": len(student_ids), "sessions": n_sessions, "kiosk_records": n_records,
                             "injected_anomalies": len(kiosk_ids)})

    # One open session today for the kiosk demo
    sess_today = attendance.create_session(admin, subj_ids[0], today, datetime.now().strftime("%H:%M"))
    attendance.open_session(admin, sess_today)

    print(f"Seeded {len(student_ids)} demo students, {n_sessions} closed sessions, {n_records} kiosk records.")
    print("Training risk model (synthetic + completed real data)...")
    m = risk_model.train(admin)
    print(f"  RF AUC {m['random_forest']['roc_auc']:.3f} | LR AUC {m['logistic_regression']['roc_auc']:.3f}")
    print(f"  Stored {risk_model.refresh_predictions(admin)} risk predictions.")
    print("  " + anomaly.run_detection(admin)["message"])
    print(f"Demo student logins: demo_01 .. demo_{len(student_ids):02d} / password '{args.password}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
