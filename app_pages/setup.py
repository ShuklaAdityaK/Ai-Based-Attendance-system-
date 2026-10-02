import streamlit as st

from core import auth, config, db

st.title("Welcome - first-run setup", anchor=False)
st.caption(config.get("institution.name"))
st.markdown(
    "No administrator account exists yet. Create one to start using the system. "
    "There are **no default credentials**: the password you choose here is the only way in.")

if db.admin_exists():
    st.success("Setup is already complete.")
    st.stop()

with st.form("first_admin", border=True):
    full_name = st.text_input("Full name")
    username = st.text_input("Username", help="3-32 characters: letters, digits, '_', '.', '-'")
    email = st.text_input("Email (optional)")
    pw1 = st.text_input("Password", type="password",
                        help=f"At least {config.get('auth.min_password_length')} characters with letters and digits")
    pw2 = st.text_input("Confirm password", type="password")
    submitted = st.form_submit_button("Create administrator", type="primary", icon=":material/shield_person:")

if submitted:
    if pw1 != pw2:
        st.error("Passwords do not match.")
    else:
        try:
            auth.create_first_admin(username, full_name, email, pw1)
        except auth.AuthError as e:
            st.error(str(e))
        else:
            st.session_state.flash = ("success", "Administrator created. Please sign in.")
            st.rerun()
