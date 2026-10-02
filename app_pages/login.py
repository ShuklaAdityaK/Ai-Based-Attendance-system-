import streamlit as st

from core import auth, config
from ui import session as ui

_, mid, _ = st.columns([1, 2, 1])
with mid:
    st.title("Smart Attendance & Analytics", anchor=False)
    st.caption(f"{config.get('institution.name')} · {config.get('institution.department')}")

    tab_login, tab_register = st.tabs(["Sign in", "Register as student"])

    with tab_login:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in", type="primary", icon=":material/login:", width="stretch")
        if ok:
            res = auth.login(username, password)
            if res.ok:
                ui.login(res.actor, res.must_change_password)
                st.rerun()
            else:
                st.error(res.message)

    with tab_register:
        st.caption("Your account stays inactive until an administrator approves it.")
        with st.form("register", clear_on_submit=False):
            full_name = st.text_input("Full name")
            roll_no = st.text_input("Roll number")
            email = st.text_input("Email")
            r_user = st.text_input("Choose a username")
            pw1 = st.text_input("Password", type="password",
                                help=f"At least {config.get('auth.min_password_length')} characters, letters and digits")
            pw2 = st.text_input("Confirm password", type="password")
            reg = st.form_submit_button("Register", icon=":material/person_add:", width="stretch")
        if reg:
            if pw1 != pw2:
                st.error("Passwords do not match.")
            else:
                try:
                    auth.register_student(r_user, full_name, roll_no, email, pw1)
                except auth.AuthError as e:
                    st.error(str(e))
                else:
                    st.success("Registration submitted. You can sign in once an administrator approves your account.")
