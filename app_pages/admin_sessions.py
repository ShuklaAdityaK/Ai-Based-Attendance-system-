from datetime import date, datetime

import pandas as pd
import streamlit as st

from core import attendance, config, db
from ui import session as ui
from ui import widgets as w

actor = ui.require_admin()
st.title("Subjects & sessions", anchor=False)

subjects = db.list_subjects(actor)
sub_label = {s["id"]: f"{s['code']} - {s['name']}" for s in subjects}
t_sess, t_subj, t_enr = st.tabs(["Sessions", "Subjects", "Enrollments"])

with t_sess:
    if not subjects:
        w.empty_state("Create a subject first (Subjects tab).")
    else:
        with st.expander("Create a session", expanded=False, icon=":material/add:"):
            with st.form("new_session", clear_on_submit=False):
                c1, c2, c3 = st.columns(3)
                sid = c1.selectbox("Subject", list(sub_label), format_func=sub_label.get)
                d = c2.date_input("Date", value=date.today())
                t = c3.time_input("Start time", value=datetime.now().replace(second=0, microsecond=0).time(), step=300)
                c1, c2 = st.columns(2)
                grace = c1.number_input("Grace minutes (Present)", 0, 120, int(config.get("attendance.grace_minutes")))
                cutoff = c2.number_input("Late cutoff minutes", 1, 240, int(config.get("attendance.late_cutoff_minutes")))
                open_now = st.checkbox("Open immediately", value=True)
                if st.form_submit_button("Create session", type="primary", icon=":material/event_available:"):
                    def _create():
                        new_id = attendance.create_session(actor, sid, d, t.strftime("%H:%M"), int(grace), int(cutoff))
                        if open_now:
                            attendance.open_session(actor, new_id)
                    ui.guarded(_create, success="Session created" + (" and opened." if open_now else "."))

        active = attendance.list_sessions(actor, status="open") + attendance.list_sessions(actor, status="scheduled")
        st.subheader("Open and scheduled", anchor=False)
        if not active:
            w.empty_state("No open or scheduled sessions.")
        for s in active:
            with st.container(border=True, horizontal=True, vertical_alignment="center"):
                badge = ("Open", "green", ":material/radio_button_checked:") if s["status"] == "open" else \
                        ("Scheduled", "gray", ":material/schedule:")
                st.badge(badge[0], color=badge[1], icon=badge[2])
                st.markdown(f"**{s['code']}** · {s['subject_name']}  \n{s['date']} at {s['start_time']} · grace "
                            f"{s['grace_minutes']} min · cutoff {s['late_cutoff_minutes']} min · "
                            f"{s['checked_in']}/{s['enrolled']} checked in")
                if s["status"] == "scheduled":
                    if st.button("Open", key=f"op_{s['id']}", icon=":material/play_arrow:"):
                        ui.guarded(attendance.open_session, actor, s["id"], success="Session opened.")
                    if st.button("Delete", key=f"del_{s['id']}", icon=":material/delete:"):
                        ui.guarded(attendance.delete_session, actor, s["id"], success="Session deleted.")
                else:
                    if st.button("Start kiosk", key=f"k_{s['id']}", type="primary", icon=":material/photo_camera:"):
                        st.session_state.kiosk_session_id = s["id"]
                        st.rerun()
                    if st.button("Close session", key=f"cl_{s['id']}", icon=":material/stop_circle:",
                                 help="Marks every enrolled student without a record as Absent"):
                        n = ui.guarded(attendance.close_session, actor, s["id"], rerun=False)
                        if n is not None:
                            ui.flash(f"Session closed. {n} student(s) auto-marked Absent.")
                        st.rerun()

        st.subheader("Recently closed", anchor=False)
        closed = attendance.list_sessions(actor, status="closed")[:50]
        if closed:
            df = pd.DataFrame(closed)
            df["attended"] = df["checked_in"].astype(str) + " / " + df["enrolled"].astype(str)
            st.dataframe(df[["id", "code", "date", "start_time", "attended", "closed_at"]], hide_index=True,
                         column_config={"id": "Session", "code": "Subject", "start_time": "Start",
                                        "attended": "Present+Late / enrolled", "closed_at": "Closed at"})
        else:
            w.empty_state("No closed sessions yet.")

with t_subj:
    with st.form("new_subject", clear_on_submit=True):
        c1, c2, c3 = st.columns([1, 3, 1])
        code = c1.text_input("Code", placeholder="KCS-701")
        name = c2.text_input("Name", placeholder="Machine Learning")
        planned = c3.number_input("Planned sessions", 1, 200, int(config.get("attendance.planned_sessions_per_subject")))
        if st.form_submit_button("Add subject", icon=":material/add:"):
            ui.guarded(db.create_subject, actor, code, name, int(planned), success=f"Subject {code.upper()} added.")
    if subjects:
        st.caption("Edit names or planned session counts directly in the table, then save.")
        edited = st.data_editor(pd.DataFrame(subjects), hide_index=True, disabled=["id", "code"], key="subj_editor",
                                column_config={"planned_sessions": st.column_config.NumberColumn(
                                    "Planned sessions", min_value=1, max_value=200)})
        with st.container(horizontal=True):
            if st.button("Save changes", icon=":material/save:"):
                def _save():
                    for r in edited.itertuples():
                        orig = next(s for s in subjects if s["id"] == r.id)
                        if orig["name"] != r.name or orig["planned_sessions"] != r.planned_sessions:
                            db.update_subject(actor, r.id, r.name, int(r.planned_sessions))
                ui.guarded(_save, success="Subjects updated.")
            del_id = st.selectbox("Delete subject", [None] + list(sub_label),
                                  format_func=lambda i: "Select..." if i is None else sub_label[i],
                                  label_visibility="collapsed", width=260)
            if st.button("Delete", icon=":material/delete:", disabled=del_id is None):
                ui.guarded(db.delete_subject, actor, del_id, success="Subject deleted.")

with t_enr:
    if not subjects:
        w.empty_state("Create a subject first.")
    else:
        sid = st.selectbox("Subject", list(sub_label), format_func=sub_label.get, key="enr_subject")
        students = db.students_with_roles(actor)
        current = [e["user_id"] for e in db.list_enrollments(actor, sid)]
        labels = {s["id"]: f"{s['roll_no'] or '-'} - {s['full_name']}" for s in students}
        chosen = st.multiselect("Enrolled students", list(labels), default=[u for u in current if u in labels],
                                format_func=labels.get, key=f"enr_ms_{sid}")
        with st.container(horizontal=True):
            if st.button("Save enrollment", type="primary", icon=":material/save:"):
                ui.guarded(db.set_subject_enrollment, actor, sid, chosen,
                           success=f"Enrollment saved ({len(chosen)} students).")
            if st.button("Enroll all active students", icon=":material/group_add:"):
                ui.guarded(db.set_subject_enrollment, actor, sid, list(labels),
                           success=f"Enrolled all {len(labels)} students.")
