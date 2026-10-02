import pandas as pd
import streamlit as st

from analytics import charts
from analytics import data as an
from core import attendance, config, db
from ml import anomaly, risk_model
from ui import session as ui
from ui import widgets as w

actor = ui.require_admin()
st.title("Attendance analytics", anchor=False)

unread = db.list_notifications(actor, unread_only=True)
pending_faces = db.list_face_enrollments(actor, "pending")
pending_users = [u for u in db.list_users(actor, "student") if not u["is_active"]]
open_sessions = attendance.list_sessions(actor, status="open")
with st.container(horizontal=True):
    if pending_users:
        st.badge(f"{len(pending_users)} account(s) awaiting approval", icon=":material/person_add:", color="orange")
    if pending_faces:
        st.badge(f"{len(pending_faces)} face enrollment(s) awaiting approval", icon=":material/face:", color="orange")
    if any(n["kind"] == "duplicate_face" for n in unread):
        st.badge("Duplicate-face alert", icon=":material/warning:", color="red")
    if open_sessions:
        st.badge(f"{len(open_sessions)} session(s) open", icon=":material/radio_button_checked:", color="blue")

thr = float(config.get("attendance.required_percentage", 75))
f = w.filter_bar(actor, "dash", with_student=True)
df = an.records_frame(actor, f["user_id"], f["subject_id"], f["date_from"], f["date_to"])
k = an.overall_kpis(df)
risk = risk_model.stored_predictions(actor, f["user_id"])
if not risk.empty and f["subject_id"]:
    risk = risk[risk["subject_id"] == f["subject_id"]]
flags = anomaly.list_flags(actor)
low = an.low_attendance(df)

with st.container(horizontal=True):
    st.metric("Overall attendance", f"{k['pct']:.1f}%", border=True)
    st.metric("Sessions", k["sessions"], border=True)
    st.metric("Present", k["present"], border=True)
    st.metric("Late", k["late"], border=True)
    st.metric("Absent", k["absent"], border=True)
    st.metric(f"Below {thr:.0f}%", len(low), border=True, help="Student-subject pairs below the threshold")
    st.metric("High risk", int((risk["level"] == "High").sum()) if not risk.empty else 0, border=True)
    st.metric("Anomalies pending", int((flags["review_status"] == "pending").sum()) if not flags.empty else 0,
              border=True)

if df.empty:
    w.empty_state("No closed-session records for these filters yet. Create and close sessions, or run "
                  "`python scripts/seed_demo.py` to load demonstration data.")
    st.stop()

subj = an.subject_summary(df)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(charts.subject_bar(subj))
with c2:
    st.plotly_chart(charts.status_stack_by_subject(subj))

tab_trend, tab_month = st.tabs(["Subject-wise trend", "Month-wise trend"])
with tab_trend:
    st.plotly_chart(charts.trend_lines(an.session_trend(df), "date_dt", "attendance_pct",
                                       "Attendance % per session"))
with tab_month:
    st.plotly_chart(charts.monthly_bars(an.monthly_trend(df)))

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader(f"Low attendance (below {thr:.0f}%)", anchor=False)
    w.df_or_empty(low[["roll_no", "full_name", "subject_code", "conducted", "attendance_pct", "classes_needed"]]
                  if not low.empty else low, "No student is below the threshold.",
                  column_config={"attendance_pct": st.column_config.ProgressColumn(
                      "Attendance %", min_value=0, max_value=100, format="%.1f%%"),
                      "classes_needed": "Classes needed", "roll_no": "Roll no", "full_name": "Name",
                      "subject_code": "Subject", "conducted": "Conducted"})
with c2:
    st.subheader("Risk indicators", anchor=False)
    if risk.empty:
        w.empty_state("No risk predictions yet. Generate them in AI/ML insights.")
    else:
        st.plotly_chart(charts.risk_level_bar(risk))

st.subheader("Anomaly indicators", anchor=False)
kiosk = df[df["source"].eq("kiosk") & df["match_distance"].notna()].copy()
if kiosk.empty:
    w.empty_state("No kiosk records in this range.")
else:
    flagged_ids = set(flags["attendance_id"]) if not flags.empty else set()
    kiosk["flagged"] = kiosk["id"].isin(flagged_ids)
    st.plotly_chart(charts.anomaly_scatter(kiosk))
    last = anomaly.last_run()
    st.caption(f"Last anomaly scan: {last['at']} - {last['message']}" if last else
               "Anomaly detection has not run yet (AI/ML insights).")

with st.expander("Per-student summary table"):
    st.dataframe(an.student_summary(df), hide_index=True)
