import streamlit as st

from core import attendance, auth, config
from ui import session as ui
from ui import widgets as w

actor = ui.require_admin()
kiosk_sid = st.session_state.get("kiosk_session_id")

if kiosk_sid is None:
    # ---- launcher (normal admin navigation) ---------------------------------
    st.title("Attendance kiosk", anchor=False)
    st.markdown("Kiosk mode locks this browser to the camera screen for one open session. Students walk up, "
                "pass the liveness check and are recognised automatically. Exiting needs your password.")
    open_s = attendance.list_sessions(actor, status="open")
    if not open_s:
        w.empty_state("No session is open. Open one in Subjects & sessions first.")
        st.stop()
    labels = {s["id"]: f"{s['code']} - {s['subject_name']} · {s['date']} {s['start_time']}" for s in open_s}
    sid = st.selectbox("Open session", list(labels), format_func=labels.get)
    st.caption("Camera access requires `localhost` or HTTPS.")
    if st.button("Start kiosk mode", type="primary", icon=":material/lock:"):
        st.session_state.kiosk_session_id = sid
        st.rerun()
    st.stop()

# ---- kiosk mode ---------------------------------------------------------------
from streamlit_webrtc import WebRtcMode, webrtc_streamer  # noqa: E402


@st.cache_resource(show_spinner="Loading AI models (face recognition, anti-spoofing)...")
def _warm_models() -> bool:
    from vision.liveness import get_passive
    from vision.recognizer import get_embedder
    get_embedder()
    get_passive()
    return True


session = attendance.get_session(actor, kiosk_sid)
if session is None or session["status"] != "open":
    st.error("This session is no longer open.")
    if st.button("Leave kiosk mode"):
        st.session_state.kiosk_session_id = None
        st.rerun()
    st.stop()

_warm_models()
st.title(f"{session['code']} · {session['subject_name']}", anchor=False)
st.caption(f"{config.get('institution.name')} · Session {session['date']} at {session['start_time']} · "
           f"Present until +{session['grace_minutes']} min, Late until +{session['late_cutoff_minutes']} min")

left, right = st.columns([3, 2])
with left:
    from vision.kiosk import KioskProcessor

    sid_for_factory = int(kiosk_sid)
    ctx = webrtc_streamer(
        key=f"kiosk-{sid_for_factory}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: KioskProcessor(sid_for_factory, actor),
        media_stream_constraints={"video": {"width": int(config.get("kiosk.frame_width", 640)),
                                            "height": int(config.get("kiosk.frame_height", 480))},
                                  "audio": False},
        rtc_configuration={"iceServers": []},   # local / LAN kiosk: no external STUN needed
        async_processing=True,
    )
    st.caption("1. Look at the camera  2. Hold still for the liveness check  3. Follow the on-screen challenge "
               "(blink twice or turn your head)  4. Wait for your name.")


@st.fragment(run_every=1)
def live_panel():
    proc = ctx.video_processor if ctx and ctx.state.playing else None
    if proc is None:
        st.info("Press **START** to switch the camera on.", icon=":material/videocam:")
        return
    events = proc.recent_events()
    roster = attendance.session_roster(actor, sid_for_factory)
    done = sum(1 for r in roster if r["status"] in ("present", "late"))
    st.metric("Checked in", f"{done} / {len(roster)}", border=True)
    st.caption(f"{len(proc.gallery)} approved faces loaded")
    for ev in events[:12]:
        icon = ":material/check_circle:" if ev.ok else ":material/cancel:"
        detail = ""
        if ev.distance is not None:
            detail = f" · dist {ev.distance:.2f} · live {ev.liveness:.2f}"
            if ev.proc_ms is not None:
                detail += f" · {ev.proc_ms:.0f} ms"
        text = f"**{ev.at}** {ev.title}  \n{ev.message}{detail}"
        if ev.ok and ev.status in ("present", "late"):
            st.success(text, icon=icon)
        elif ev.ok:
            st.info(text, icon=icon)
        else:
            st.error(text, icon=icon)


with right:
    debug = st.toggle("Show diagnostics overlay", key="kiosk_debug",
                      help="Draws live liveness / blink / head-turn / match values on the video for tuning")
    if ctx and ctx.video_processor:
        ctx.video_processor.debug = debug
    live_panel()
    with st.container(horizontal=True):
        if st.button("Reload enrolled faces", icon=":material/refresh:",
                     help="Use after approving new face enrollments"):
            if ctx and ctx.video_processor:
                n = ctx.video_processor.reload_gallery()
                st.toast(f"Reloaded {n} approved faces.")
    with st.popover("Exit kiosk mode", icon=":material/lock_open:"):
        with st.form("unlock"):
            pw = st.text_input("Administrator password", type="password")
            close_too = st.checkbox("Also close the session now (auto-marks Absent)")
            if st.form_submit_button("Unlock", type="primary"):
                if auth.verify_admin_password(actor, pw):
                    st.session_state.kiosk_session_id = None
                    if close_too:
                        n = attendance.close_session(actor, sid_for_factory)
                        ui.flash(f"Kiosk closed. Session closed; {n} student(s) auto-marked Absent.")
                    st.rerun()
                else:
                    st.error("Wrong password.")
