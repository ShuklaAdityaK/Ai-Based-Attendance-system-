# AI-Based Smart Attendance & Analytics System

**B.Tech. (CSE) Final Year Project** — Shukla Aditya Kumar Deepak
Shambhunath Institute of Engineering & Technology, Prayagraj

A Python + Streamlit attendance platform. Students mark attendance at a classroom
**kiosk** that combines **ArcFace deep-learning face recognition** with **two-layer
liveness detection** (MiniFASNet CNN + an active blink / head-turn challenge).
Records are classified against configurable session rules. **Machine learning**
predicts students at risk of falling below the required attendance (Random
Forest, with a Logistic Regression baseline) and flags anomalous check-ins for
review (Isolation Forest). The app provides role-based dashboards, Plotly
analytics, Excel/PDF reports, an immutable audit trail, and local backup/restore.

---

> **New here?** Read the step-by-step [User Guide](docs/USER_GUIDE.md).

## 1. Quick start (Windows / Linux, Python 3.10 or 3.11)

```bash
# 1. Create an environment (uv shown; plain venv + pip works too)
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
#   or:  python -m venv .venv && .venv\Scripts\pip install -r requirements.txt

# 2. Download the pretrained models into ./models (~200 MB)
.venv\Scripts\python scripts/download_models.py

# 3. Run
.venv\Scripts\streamlit run app.py
```

Open http://localhost:8501. On first run the app asks you to **create the
administrator account**; there are no default credentials.

*Optional demo data* (for presentations, before real semesters exist):
`.venv\Scripts\python scripts/seed_demo.py`. It creates 30 demo students
(`demo_01`–`demo_30`, password `demo1234`, no face data), 10 weeks of closed
sessions, a few injected anomalies, and one open session for today. It then trains
the models. Every demo username starts with `demo_`.

> Camera access in the browser needs `localhost` or **HTTPS**. To run the kiosk on
> another machine on the network, put the app behind HTTPS (for example a reverse proxy
> with a certificate). A browser blocks the webcam on plain `http://<ip>`.

## 2. Typical workflow

1. **Student** registers → **admin** approves the account (*Users & approvals*).
2. Student logs in → *Face enrollment*: accepts the biometric consent notice and
   captures 5–10 quality-checked frames (frontal / slight left / slight right).
   The embeddings go through a duplicate-face check and wait for admin approval.
3. Admin creates **subjects**, enrolls students, and creates and **opens a session**.
4. Admin presses **Start kiosk**. The browser is now locked to the camera screen.
   Each student: face detected → passive CNN liveness → random challenge →
   1:N recognition → **Present / Late** recorded.
5. Admin exits the kiosk with their password and **closes the session**. Every
   enrolled student without a record is auto-marked **Absent**.
6. Analytics, risk list, anomaly review, corrections (reason required, audit-logged),
   reports, backups.

## 3. Requirement → implementation map

| Requirement | Where |
|---|---|
| 2.1 Registration, approval, bcrypt, lockout (5 / 15 min, configurable), timeout, first-run admin | `core/auth.py`, `ui/session.py`, `app_pages/setup.py`, `login.py` |
| 2.2 Consent, 5–10 frames, quality checks, embeddings only, duplicate check, approval, re-enrollment | `vision/enrollment.py`, `vision/detector.py`, `app_pages/face_enrollment.py`, `core/db.py` |
| 2.3 YuNet + ArcFace + cosine 1:N, rejection rules, stored distance/liveness/attempts | `vision/detector.py`, `vision/recognizer.py`, `vision/kiosk.py`, `core/attendance.py` |
| 2.4 MiniFASNet (ONNX Runtime) + MediaPipe EAR blink / head-turn, streamlit-webrtc | `vision/liveness.py`, `vision/kiosk.py`, `app_pages/kiosk.py` |
| 2.5 Present / Late / Absent rules, auto-absent on close, late weight, threshold | `core/attendance.py` |
| 2.6 Corrections with mandatory reason + immutable audit | `core/attendance.py`, `core/audit.py`, SQLite triggers in `core/db.py` |
| 2.7 Risk model (RF + LR baseline), features, synthetic cold start, rule fallback, levels + top factors | `ml/features.py`, `ml/synthetic_data.py`, `ml/risk_model.py` |
| 2.8 Isolation Forest anomalies, review queue, ≥ 30 records | `ml/anomaly.py`, `app_pages/admin_ai.py` |
| 2.9 Analytics (overall, subject-wise, month-wise, P/L/A, low, risk, anomaly) with filters | `analytics/data.py`, `analytics/charts.py`, `app_pages/admin_dashboard.py` |
| 2.10 Student dashboard (own data only, classes needed, risk, report) | `app_pages/student_dashboard.py`, `my_report.py` |
| 2.11 Admin dashboard (users, approvals, sessions, kiosk, corrections, AI/ML, model panel, reports, backup) | `app_pages/admin_*.py`, `kiosk.py` |
| 2.12 Excel (openpyxl) + PDF (ReportLab) with filters | `reports/` |
| 2.13 SQLite persistence, backup .zip export / restore | `core/db.py`, `core/backup.py`, `app_pages/admin_system.py` |
| 7 Access control in the data layer | `core/access.py` (every data function takes the calling `Actor`) |
| 9 Security & privacy (DPDP Act 2023) | consent notice, embeddings only, deletion cascade, `.gitignore` |
| 12 Testing & evaluation | `tests/`, `scripts/evaluate_*.py` |

## 4. Project structure

```
app.py                 Streamlit entry: role-based navigation, timeout, kiosk lock
config.yaml            thresholds, timings, paths (admin-editable keys can be overridden in-app)
core/                  access.py, auth.py, db.py, attendance.py, audit.py, backup.py, config.py
vision/                detector.py (YuNet + quality), recognizer.py (ArcFace), liveness.py,
                       enrollment.py, kiosk.py (streamlit-webrtc pipeline)
ml/                    features.py, synthetic_data.py, risk_model.py, anomaly.py
analytics/             data.py (Pandas), charts.py (Plotly)
reports/               excel_report.py, pdf_report.py, common.py
app_pages/             UI pages (student / admin / kiosk)
ui/                    session handling and shared widgets
models/                pretrained ONNX/TFLite models; models/trained = risk/anomaly models
data/                  attendance.db, reports/, backups/   (git-ignored)
scripts/               download_models, seed_demo, evaluate_recognition / liveness / models,
                       capture_liveness_samples
tests/                 pytest suite
docs/                  liveness_trial_sheet.csv
```

The UI pages live in `app_pages/` instead of the `pages/` folder named in the
specification. Streamlit treats a folder named `pages/` as legacy auto-navigation,
which would let any visitor open admin pages directly. `app_pages/` with
`st.navigation` shows each role only its own pages, and the kiosk lock hides
everything else.

## 5. AI / ML / DL modules

| # | Module | Model | Notes |
|---|---|---|---|
| 1 | Face detection | OpenCV **YuNet** (`face_detection_yunet_2023mar.onnx`) | box + 5 landmarks |
| 2 | Face recognition | **ArcFace ResNet-50** (`w600k_r50`, InsightFace *buffalo_l*) via ONNX Runtime | 5-point alignment to 112×112; 512-d L2-normalised embedding; cosine distance; per-user best match |
| 3 | Passive liveness | **MiniFASNetV2 + MiniFASNetV1SE** (Silent-Face-Anti-Spoofing) via ONNX Runtime | crops at scale 2.7 / 4.0; mean "real" probability over several frames |
| 4 | Active liveness | **MediaPipe Face Landmarker** (478-point mesh) | EAR blink counting with an adaptive baseline; head turn from nose position between the cheek contours; must start frontal; time-limited |
| 5 | Classification | rule engine | `classify_check_in` |
| 6 | Risk prediction | **Random Forest** vs **Logistic Regression** | 9 engineered features; student-grouped hold-out; perturbation-based top factors |
| 7 | Anomaly detection | **Isolation Forest** | 6 features (log-scaled kiosk gap); robust-z explanations |
| 8 | Analytics | Pandas + Plotly | |

The specification allows "DeepFace or InsightFace". This project runs InsightFace's
ArcFace model directly with ONNX Runtime, so the embeddings are identical to the
`insightface` package. That package needs a C++ build toolchain on Windows,
which this approach avoids.

## 6. Configuration

All timings, thresholds and weights live in `config.yaml`. The following can be
changed at runtime under **System & backup → Settings**, and every change is
audit-logged: lockout attempts and duration, inactivity timeout, grace, late
cutoff, late weight, required %, planned sessions, recognition threshold,
duplicate threshold, passive-liveness threshold, challenge time limit, anomaly
contamination.

## 7. Testing & evaluation

```bash
.venv\Scripts\python -m pytest -q                                 # 53 tests
.venv\Scripts\python scripts/evaluate_models.py                   # risk model + anomaly injection
.venv\Scripts\python scripts/evaluate_recognition.py --lfw         # FAR/FRR on LFW (public)
.venv\Scripts\python scripts/evaluate_recognition.py --data eval_data/faces   # your volunteers
.venv\Scripts\python scripts/capture_liveness_samples.py           # record real / print / replay clips
.venv\Scripts\python scripts/evaluate_liveness.py                  # attack rejection / false reject
```

The unit tests cover the rule boundaries (grace and cutoff, auto-absent), duplicate
prevention, audit immutability, lockout, password policy, **access control** (a
student calling about 30 admin and other-user functions is blocked), ML sanity,
anomaly detection on injected cases, the reports, a backup round-trip, and the liveness challenge and
kiosk state machine (spoof, unknown face, multiple faces, wrong subject, duplicate mark), with the neural
networks replaced by fakes. Results are
written to `eval_results/`. See `docs/EVALUATION.md` for the measured numbers.

## 8. Security & privacy

* bcrypt (cost 12), lockout, inactivity timeout, forced change of admin-issued passwords, no default credentials.
* Role checks in the data layer (`core/access.py`); a student's queries are pinned to their own user id.
* Only **embeddings** are stored. Frames are discarded after enrollment (`enrollment.keep_raw_images: false`).
  Rejected or superseded embeddings are deleted.
* Consent is timestamped. Deleting a user cascades to embeddings, enrollments, attendance, predictions and flags.
* `attendance_audit` and `admin_audit` are append-only (SQLite triggers abort UPDATE/DELETE).
* `data/`, `eval_data/`, `logs/`, model files and backups are git-ignored, so biometric data is never committed.
* Follows the principles of India's **Digital Personal Data Protection Act, 2023**:
  consent, purpose limitation, minimal retention, and deletion on request.

## 9. Known limitations (state these in the report)

* Liveness targets **printed-photo and phone/screen replay** attacks. It is **not**
  designed to stop 3D masks or deepfake/virtual-camera injection.
* The risk model is trained on **simulated semesters** (cold start). Completed real
  semesters are added automatically on retraining. Below 5 sessions a rule-based
  estimate is shown instead.
* The MiniFASNet scores depend on the camera and lighting. Tune
  `liveness.passive_threshold` with `evaluate_liveness.py` on the actual kiosk camera.
* The kiosk serves one student at a time (frames with more than one face are rejected).

## 10. Third-party models and licences

The pretrained models are not stored in this repository. `scripts/download_models.py`
downloads them from their original sources and checks each file against a pinned
SHA-256 hash.

| Model | Source | Licence |
|---|---|---|
| YuNet face detector | OpenCV Zoo | MIT |
| ArcFace R50 (`w600k_r50`, buffalo_l pack) | InsightFace | **Non-commercial research use only** |
| MiniFASNetV2 / V1SE | Minivision Silent-Face-Anti-Spoofing (ONNX export by yakhyo/face-anti-spoofing) | Apache 2.0 |
| Face Landmarker | Google MediaPipe | Apache 2.0 |
| LFW dataset (evaluation only) | University of Massachusetts, Amherst | research use |

This is an academic project, which the InsightFace licence permits. Any commercial
deployment would need a commercially licensed face-recognition model in place of
ArcFace `buffalo_l`. Licence terms can change, so check each upstream page before
reuse.

## 11. Future scope

GPS/geofencing for self check-in, QR attendance, email/SMS alerts, cloud sync,
mobile/PWA app, scheduled backups, encryption at rest, a Faculty role, leave/excused
status, a holiday calendar.
