import json

import pandas as pd
import streamlit as st

from core import auth, db
from ui import session as ui
from ui import widgets as w

actor = ui.require_admin()
st.title("Users & approvals", anchor=False)

users = db.list_users(actor)
pending_users = [u for u in users if not u["is_active"] and u["role"] == "student"]
face_queue = db.list_face_enrollments(actor)
face_pending = [f for f in face_queue if f["status"] in ("pending", "blocked")]
notes = db.list_notifications(actor)
unread = sum(1 for n in notes if not n["is_read"])

t_users, t_faces, t_notes, t_new = st.tabs([
    f"Users ({len(pending_users)} pending)", f"Face approvals ({len(face_pending)})",
    f"Notifications ({unread} new)", "Create user"])

with t_users:
    if pending_users:
        st.subheader("Awaiting approval", anchor=False)
        for u in pending_users:
            with st.container(border=True, horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{u['full_name']}** · {u['roll_no']} · {u['username']} · {u['email'] or '-'}  \n"
                            f":gray[Registered {u['created_at']}]")
                if st.button("Approve", key=f"ap_{u['id']}", type="primary", icon=":material/check:"):
                    ui.guarded(db.set_user_active, actor, u["id"], True, success=f"Approved {u['full_name']}.")
                if st.button("Delete", key=f"rj_{u['id']}", icon=":material/delete:"):
                    ui.guarded(db.delete_user, actor, u["id"], success=f"Deleted registration of {u['full_name']}.")

    st.subheader("All users", anchor=False)
    latest_face = {}
    for f in sorted(face_queue, key=lambda r: r["id"]):
        latest_face[f["user_id"]] = f["status"]
    table = pd.DataFrame([{**u, "face": latest_face.get(u["id"], "none"),
                           "active": bool(u["is_active"]),
                           "locked": bool(u["locked_until"] and u["locked_until"] > db.now())} for u in users])
    st.dataframe(table[["id", "username", "full_name", "roll_no", "email", "role", "active", "locked", "face",
                        "created_at"]], hide_index=True)

    st.subheader("Manage a user", anchor=False)
    opts = {u["id"]: f"{u['full_name']} ({u['username']}, {u['role']})" for u in users}
    uid = st.selectbox("User", list(opts), format_func=opts.get, key="manage_user")
    target = next(u for u in users if u["id"] == uid)
    with st.container(horizontal=True):
        if target["is_active"]:
            if st.button("Deactivate", icon=":material/person_off:", disabled=uid == actor.id):
                ui.guarded(db.set_user_active, actor, uid, False, success="User deactivated.")
        else:
            if st.button("Activate", icon=":material/person_check:"):
                ui.guarded(db.set_user_active, actor, uid, True, success="User activated.")
        if st.button("Reset password", icon=":material/lock_reset:"):
            temp = ui.guarded(auth.reset_password, actor, uid, rerun=False)
            if temp:
                st.session_state.temp_pw = (target["username"], temp)
        confirm = st.checkbox("Confirm permanent deletion (removes face data and attendance)", key=f"cd_{uid}",
                              disabled=uid == actor.id)
        if st.button("Delete user", icon=":material/delete_forever:", type="primary",
                     disabled=not confirm or uid == actor.id):
            ui.guarded(db.delete_user, actor, uid, success=f"Deleted {target['username']} and their face data.")
    if st.session_state.get("temp_pw"):
        name, temp = st.session_state.temp_pw
        st.warning(f"Temporary password for **{name}**: `{temp}` - share it securely. "
                   "The user must change it at next login. It will not be shown again.")
        if st.button("Done", key="tmp_done"):
            del st.session_state["temp_pw"]
            st.rerun()

with t_faces:
    if not face_pending:
        w.empty_state("No face enrollments awaiting review.")
    for f in face_pending:
        q = json.loads(f["quality_summary"] or "{}")
        with st.container(border=True):
            head = f"**{f['full_name']}** ({f['roll_no'] or f['username']}) · {f['frame_count']} frames · {f['created_at']}"
            st.markdown(head)
            if f["status"] == "blocked":
                st.error(f"Blocked by the duplicate-face check: nearest other user at cosine distance "
                         f"{q.get('nearest_other_user_distance')}. Approve only if you have verified this is a "
                         f"different person (e.g. identical twins).", icon=":material/warning:")
            st.caption(f"Consent given {f['consent_at']} · mean sharpness {q.get('mean_blur')} · mean brightness "
                       f"{q.get('mean_brightness')} · yaw range {q.get('yaw_range')} · nearest other user "
                       f"{q.get('nearest_other_user_distance')}")
            note = st.text_input("Review note (optional)", key=f"fn_{f['id']}")
            with st.container(horizontal=True):
                if st.button("Approve", key=f"fa_{f['id']}", type="primary", icon=":material/check:"):
                    ui.guarded(db.review_face_enrollment, actor, f["id"], True, note, success="Face enrollment approved.")
                if st.button("Reject", key=f"fr_{f['id']}", icon=":material/close:"):
                    ui.guarded(db.review_face_enrollment, actor, f["id"], False, note,
                               success="Face enrollment rejected; its embeddings were deleted.")
    with st.expander("Enrollment history"):
        st.dataframe(pd.DataFrame(face_queue)[["id", "full_name", "status", "frame_count", "consent_at", "created_at",
                                               "reviewed_at", "review_note"]] if face_queue else pd.DataFrame(),
                     hide_index=True)

with t_notes:
    if not notes:
        w.empty_state("No notifications.")
    else:
        if unread and st.button("Mark all as read", icon=":material/done_all:"):
            db.mark_notifications_read(actor)
            st.rerun()
        icon = {"duplicate_face": ":material/warning:", "face_pending": ":material/face:",
                "user_pending": ":material/person_add:"}
        for n in notes[:100]:
            text = f"{n['created_at']} - {n['message']}"
            if n["kind"] == "duplicate_face":
                st.error(text, icon=icon[n["kind"]])
            elif not n["is_read"]:
                st.info(text, icon=icon.get(n["kind"], ":material/info:"))
            else:
                st.caption(text)

with t_new:
    st.caption("Accounts created here are active immediately; the user must change the password at first login.")
    with st.form("new_user", clear_on_submit=True):
        role = st.segmented_control("Role", ["student", "admin"], default="student", format_func=str.title)
        full_name = st.text_input("Full name")
        roll = st.text_input("Roll number (students)")
        email = st.text_input("Email")
        uname = st.text_input("Username")
        pw = st.text_input("Initial password", type="password")
        if st.form_submit_button("Create", type="primary", icon=":material/person_add:"):
            ui.guarded(auth.create_user_by_admin, actor, uname, full_name, roll or None, email, pw, role or "student",
                       success=f"Created {uname}.")
