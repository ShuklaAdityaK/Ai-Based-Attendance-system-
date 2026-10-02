import streamlit as st

from core import auth
from ui import session as ui

actor = ui.require_actor()
st.title("Change your password", anchor=False)
st.warning("Your password was set or reset by an administrator. Choose a new password to continue.")

with st.form("force_change"):
    old = st.text_input("Temporary password", type="password")
    new1 = st.text_input("New password", type="password")
    new2 = st.text_input("Confirm new password", type="password")
    ok = st.form_submit_button("Change password", type="primary")

if ok:
    if new1 != new2:
        st.error("Passwords do not match.")
    else:
        try:
            auth.change_password(actor, old, new1)
        except auth.AuthError as e:
            st.error(str(e))
        else:
            st.session_state.must_change_password = False
            ui.flash("Password changed.")
            st.rerun()

if st.button("Log out", icon=":material/logout:"):
    ui.logout()
    st.rerun()
