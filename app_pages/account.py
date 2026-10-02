import streamlit as st

from core import auth, db
from ui import session as ui

actor = ui.require_actor()
st.title("Account", anchor=False)

u = db.get_user(actor, actor.id)
with st.container(border=True):
    st.markdown(f"**{u['full_name']}**")
    st.caption(f"Username: {u['username']} · Role: {u['role'].title()}"
               + (f" · Roll no: {u['roll_no']}" if u["roll_no"] else "")
               + (f" · {u['email']}" if u["email"] else ""))
    enr = db.face_enrollment_status(actor, actor.id)
    st.caption("Face enrollment: " + (enr["status"].title() if enr else "not enrolled"))

st.subheader("Change password", anchor=False)
with st.form("change_pw"):
    old = st.text_input("Current password", type="password")
    new1 = st.text_input("New password", type="password")
    new2 = st.text_input("Confirm new password", type="password")
    ok = st.form_submit_button("Update password", type="primary")
if ok:
    if new1 != new2:
        st.error("Passwords do not match.")
    else:
        try:
            auth.change_password(actor, old, new1)
            st.success("Password updated.")
        except auth.AuthError as e:
            st.error(str(e))

st.subheader("Your data and privacy", anchor=False)
st.markdown(
    "Biometric data is handled under the *Digital Personal Data Protection Act, 2023*: only face "
    "embeddings are stored, solely for attendance. To withdraw consent or request deletion of your data, "
    "contact the administrator. Deleting your account removes your embeddings and enrollment records.")
