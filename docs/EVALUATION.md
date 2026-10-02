# Evaluation results (Section 12)

Measured on 2 Oct 2026, CPU only (Windows 11, Python 3.11). The raw outputs are in `eval_results/`.

## 1. Face recognition: ArcFace R50, cosine distance

**Dataset:** LFW (public; Labeled Faces in the Wild), the 62 people with at least 20
images, 20 images each = 1,240 faces. Every face was detected by YuNet.
**Protocol:** all 11,780 genuine pairs plus 11,780 random impostor pairs.
Command: `scripts/evaluate_recognition.py --lfw`.

| Threshold (cosine distance) | FAR | FRR | Accuracy |
|---|---|---|---|
| 0.50 | 0.00% | 5.31% | 97.34% |
| 0.55 | 0.00% | 2.41% | 98.80% |
| **0.60 (configured default)** | **0.00%** | **1.08%** | **99.46%** |
| 0.65 | 0.00% | 0.54% | 99.73% |
| 0.70 | 0.00% | 0.31% | 99.85% |
| 0.775 (lowest FRR with FAR ≤ 1%) | 0.03% | 0.16% | 99.90% |
| 0.805 (equal error rate) | 0.14% | 0.16% | 99.85% |

* Genuine distance: mean 0.34, 95th percentile 0.50. Impostor distance: mean 0.99, 1st percentile 0.85.
* **1:N identification** (gallery = 5 images per person, 930 probes, run at threshold 0.85):
  100% correct, 0% misidentified, 0% rejected.
* The NFR target (FAR ≤ 1%) is met at the default threshold with a wide margin. The default 0.60 is
  deliberately conservative, because webcam images at a kiosk are harder than LFW press photos.
  **Re-tune on enrolled volunteers** at the kiosk with
  `scripts/evaluate_recognition.py --data eval_data/faces` (one folder per person).

## 2. Liveness

* **Passive (MiniFASNet):** run `scripts/capture_liveness_samples.py` to record ≥ 20 real, ≥ 20 printed-photo
  and ≥ 20 phone-replay attempts on the kiosk camera. Then run `scripts/evaluate_liveness.py`, which reports
  the attack rejection rate (per attack type) and the false-reject rate for real users across thresholds.
* **Active challenge:** evaluate it live at the kiosk and log each attempt in `docs/liveness_trial_sheet.csv`.
* *Results: to be filled in after the kiosk trials.*

## 3. Risk model

Trained on simulated semesters (4 archetypes × 120 students × 3 semesters, 15,566 checkpoint samples).
The split is a held-out **25% of students** (`GroupShuffleSplit`, so no student appears in both train and test).
Command: `scripts/evaluate_models.py`.

| Model | Precision | Recall | F1 | ROC-AUC | Accuracy |
|---|---|---|---|---|---|
| **Random Forest** | 0.928 | 0.843 | 0.884 | 0.958 | 0.884 |
| Logistic Regression (baseline) | 0.919 | 0.861 | 0.889 | 0.957 | 0.887 |

The Random Forest and the baseline perform about the same on this data. The at-risk signal (attendance to
date, recent trend, maximum achievable %) is close to linear in the simulated semesters. The forest is kept as
the deployed model because it captures non-linear effects (absence streaks, weekday patterns) and gives
per-student factor attributions. Most important features: attendance to date (0.43), last-10-session
attendance (0.18), maximum achievable % (0.14).

## 4. Anomaly detector: Isolation Forest, contamination 0.05

1,800 realistic kiosk records with 45 injected anomalies:

| Injected kind | Detection rate |
|---|---|
| Borderline match + retries (possible look-alike / proxy) | 100% |
| Weak liveness that still passed | 100% |
| Unusual arrival time for this student | 100% |
| Burst of check-ins seconds apart | 80% |
| **Overall** | **93.3%** |
| False-flag rate on normal records | 2.8% |

For bursts, the first record of each injected burst has a normal gap, so it is rarely flagged; the later
records are caught.

## 5. Unit tests: 37 passed

`pytest -q`: attendance rules at the grace and cutoff boundaries, auto-absent, duplicate-mark prevention
(application and `UNIQUE` constraint), corrections and audit immutability, bcrypt, lockout,
first-run admin, approval, **access control** (a student calling admin and other-user functions is blocked
in the data layer), ML sanity, anomaly injection, reports, and a backup round-trip.

## 6. Performance (NFR ≤ 5 s)

Kiosk pipeline on CPU: detection + 5-frame passive liveness + 2 ArcFace embeddings + matching
≈ **1.4 s** for the first student (including model warm-up), excluding challenge time. A single
ArcFace embedding takes ~0.15–0.25 s and a MiniFASNet pair ~10 ms.
