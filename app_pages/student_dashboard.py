import pandas as pd
import streamlit as st

from analytics import charts
from analytics import data as an
from core import attendance, config
from ml import risk_model
from ui import session as ui
from ui import widgets as w

actor = ui.require_actor()
st.title(f"Hello, {actor.full_name.split()[0]}", anchor=False)
thr = float(config.get("attendance.required_percentage", 75))
st.caption(f"Required attendance: {thr:.0f}% · Late counts as {config.get('attendance.late_weight')} of a class. "
           "Figures use closed sessions only.")

f = w.filter_bar(actor, "me")
# user_id is always the logged-in user; the data layer enforces it as well.
df = an.records_frame(actor, actor.id, f["subject_id"], f["date_from"], f["date_to"])
k = an.overall_kpis(df)

with st.container(horizontal=True):
    st.metric("Overall attendance", f"{k['pct']:.1f}%", border=True,
              delta=f"{k['pct'] - thr:+.1f} pts vs required" if k["records"] else None)
    st.metric("Sessions", k["records"], border=True)
    st.metric("Present", k["present"], border=True)
    st.metric("Late", k["late"], border=True)
    st.metric("Absent", k["absent"], border=True)

if df.empty:
    w.empty_state("No attendance records yet for the selected filters. Records appear after sessions are closed.")

st.subheader("By subject", anchor=False)
risk = risk_model.predict_frame(actor, actor.id)
summary = an.student_subject_summary(df)
if risk.empty:
    w.empty_state("You are not enrolled in any subject yet.")
else:
    if f["subject_id"]:
        risk = risk[risk["subject_id"] == f["subject_id"]]
    cols = st.columns(min(3, max(1, len(risk))))
    for i, r in enumerate(risk.itertuples()):
        s = summary[summary["subject_id"] == r.subject_id] if not summary.empty else pd.DataFrame()
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"**{r.code}** · {r.subject_name}")
            if s.empty:
                st.caption("No closed sessions in the selected range.")
            else:
                s = s.iloc[0]
                st.metric("Attendance", f"{s.attendance_pct:.1f}%", label_visibility="collapsed")
                st.caption(f"{s.present} present · {s.late} late · {s.absent} absent of {s.conducted}")
                if s.classes_needed > 0:
                    st.markdown(f":red[Attend the next **{int(s.classes_needed)}** classes in a row to reach "
                                f"{thr:.0f}%.]")
                elif s.classes_needed == 0:
                    st.markdown(":green[On track - at or above the requirement.]")
            w.level_badge(r.level, r.probability, r.method)
            st.caption(f"Sessions remaining (planned): {int(r.sessions_remaining)} · "
                       f"max achievable {r.max_achievable_pct:.0f}%")
            with st.expander("Why this risk level?"):
                for fct in r.top_factors:
                    st.markdown(f"- **{fct['label']}**: {fct['value']} (typical {fct['typical']}) - {fct['direction']}")

if not df.empty:
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.trend_lines(an.cumulative_trend(df), "date_dt", "cum_pct",
                                           "Your running attendance % by subject"))
    with right:
        st.plotly_chart(charts.status_bar(k, "Your Present / Late / Absent"))

    st.subheader("Session history", anchor=False)
    hist = df.sort_values(["date", "start_time"], ascending=False)[
        ["date", "start_time", "subject_code", "subject_name", "status", "check_in_time", "source"]].copy()
    hist["status"] = hist["status"].map(w.status_label)
    st.dataframe(hist, hide_index=True, column_config={
        "date": "Date", "start_time": "Start", "subject_code": "Code", "subject_name": "Subject",
        "status": "Status", "check_in_time": "Checked in at", "source": "Source"})

open_now = attendance.list_sessions(actor, status="open")
if open_now:
    st.info("Open now: " + ", ".join(f"{s['code']} ({s['start_time']})" for s in open_now)
            + " - check in at the classroom kiosk.", icon=":material/photo_camera:")
