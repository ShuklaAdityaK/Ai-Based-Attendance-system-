import pandas as pd
import streamlit as st

from core import attendance, audit
from ui import session as ui
from ui import widgets as w

actor = ui.require_admin()
st.title("Records & corrections", anchor=False)

t_rec, t_fix, t_audit = st.tabs(["All records", "Manual correction", "Correction audit log"])

with t_rec:
    f = w.filter_bar(actor, "rec", with_student=True)
    recs = pd.DataFrame(attendance.get_records(actor, f["user_id"], f["subject_id"], f["date_from"], f["date_to"]))
    if recs.empty:
        w.empty_state("No records for these filters.")
    else:
        recs["status"] = recs["status"].map(w.status_label)
        st.caption(f"{len(recs)} records (includes open sessions).")
        st.dataframe(recs[["date", "start_time", "subject_code", "roll_no", "full_name", "status", "check_in_time",
                           "source", "match_distance", "liveness_score", "failed_attempts", "session_status"]],
                     hide_index=True, column_config={
                         "match_distance": st.column_config.NumberColumn("Match dist.", format="%.3f"),
                         "liveness_score": st.column_config.NumberColumn("Liveness", format="%.2f"),
                         "failed_attempts": "Failed tries", "session_status": "Session"})

with t_fix:
    st.caption("Use for recognition failures or disputed records. A reason is mandatory and every change is "
               "written to the immutable audit log. Absent can also be set manually.")
    sessions = [s for s in attendance.list_sessions(actor) if s["status"] != "scheduled"]
    if not sessions:
        w.empty_state("No open or closed sessions.")
    else:
        labels = {s["id"]: f"{s['date']} {s['start_time']} · {s['code']} ({s['status']})" for s in sessions}
        sid = st.selectbox("Session", list(labels), format_func=labels.get)
        roster = pd.DataFrame(attendance.session_roster(actor, sid))
        if roster.empty:
            w.empty_state("No students enrolled in this subject.")
        else:
            show = roster.copy()
            show["status"] = show["status"].map(w.status_label)
            st.dataframe(show[["roll_no", "full_name", "status", "check_in_time", "source"]], hide_index=True,
                         height=min(420, 38 + 35 * len(show)))
            stu_labels = {r.user_id: f"{r.roll_no or '-'} - {r.full_name} (now: {w.status_label(r.status)})"
                          for r in roster.itertuples()}
            with st.form("correction", clear_on_submit=True):
                uid = st.selectbox("Student", list(stu_labels), format_func=stu_labels.get)
                new_status = st.segmented_control("New status", ["present", "late", "absent"],
                                                  format_func=w.status_label, default="present")
                reason = st.text_area("Reason (required)", placeholder="e.g. Recognition failed due to lighting; "
                                                                     "verified present by the faculty.")
                if st.form_submit_button("Apply correction", type="primary", icon=":material/edit:"):
                    ui.guarded(attendance.correct_attendance, actor, sid, uid, new_status or "", reason,
                               success="Correction applied and audit-logged.")

with t_audit:
    df = audit.attendance_audit_frame(actor)
    w.df_or_empty(df, "No manual corrections yet.", column_config={
        "changed_at": "When", "changed_by_name": "Admin", "full_name": "Student", "roll_no": "Roll no",
        "code": "Subject", "date": "Session date", "old_status": "Old", "new_status": "New", "reason": "Reason"})
