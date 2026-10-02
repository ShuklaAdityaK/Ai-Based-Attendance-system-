"""AI-Based Smart Attendance & Analytics System - Streamlit entry point.

Run:  streamlit run app.py
"""
import streamlit as st

from core import config, db
from ui import session as ui

st.set_page_config(page_title="Smart Attendance", page_icon=":material/how_to_reg:", layout="wide")


@st.cache_resource
def _bootstrap() -> bool:
    db.init_db()
    for key in ("reports", "backups", "trained_models"):
        config.path(key).mkdir(parents=True, exist_ok=True)
    config.log.info("Application started")
    return True


_bootstrap()
ui.init_state()
ui.enforce_timeout_and_refresh()

actor = ui.actor()
P = st.Page

if not db.admin_exists():
    nav = st.navigation([P("app_pages/setup.py", title="First-run setup", icon=":material/rocket_launch:")],
                       position="hidden")
elif actor is None:
    nav = st.navigation([P("app_pages/login.py", title="Sign in", icon=":material/login:")], position="hidden")
elif st.session_state.must_change_password:
    nav = st.navigation([P("app_pages/change_password.py", title="Change password", icon=":material/key:")],
                        position="hidden")
elif actor.is_admin and st.session_state.kiosk_session_id is not None:
    # Kiosk lock: only the kiosk page is reachable until an admin unlocks it.
    nav = st.navigation([P("app_pages/kiosk.py", title="Attendance kiosk", icon=":material/photo_camera:")],
                        position="hidden")
elif actor.is_admin:
    nav = st.navigation({
        "Overview": [
            P("app_pages/admin_dashboard.py", title="Analytics", icon=":material/monitoring:", default=True),
            P("app_pages/admin_ai.py", title="AI/ML insights", icon=":material/psychology:"),
        ],
        "Attendance": [
            P("app_pages/admin_sessions.py", title="Subjects & sessions", icon=":material/event:"),
            P("app_pages/kiosk.py", title="Attendance kiosk", icon=":material/photo_camera:"),
            P("app_pages/admin_attendance.py", title="Records & corrections", icon=":material/fact_check:"),
            P("app_pages/admin_reports.py", title="Reports", icon=":material/description:"),
        ],
        "Administration": [
            P("app_pages/admin_users.py", title="Users & approvals", icon=":material/group:"),
            P("app_pages/admin_system.py", title="System & backup", icon=":material/settings:"),
        ],
        "My account": [
            P("app_pages/student_dashboard.py", title="My attendance", icon=":material/person:"),
            P("app_pages/face_enrollment.py", title="My face enrollment", icon=":material/face:"),
            P("app_pages/account.py", title="Account", icon=":material/manage_accounts:"),
        ],
    })
else:
    nav = st.navigation([
        P("app_pages/student_dashboard.py", title="My attendance", icon=":material/dashboard:", default=True),
        P("app_pages/face_enrollment.py", title="Face enrollment", icon=":material/face:"),
        P("app_pages/my_report.py", title="My report", icon=":material/description:"),
        P("app_pages/account.py", title="Account", icon=":material/manage_accounts:"),
    ])

if actor is not None and st.session_state.kiosk_session_id is None and not st.session_state.must_change_password:
    with st.sidebar:
        st.caption("Signed in as")
        st.markdown(f"**{actor.full_name}**  \n{actor.username} · {actor.role.title()}")
        if st.button("Log out", icon=":material/logout:", width="stretch"):
            ui.logout("You have been logged out.")
            st.rerun()
        st.caption(config.get("institution.name"))

ui.show_flash()
nav.run()
