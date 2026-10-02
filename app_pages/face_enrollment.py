from datetime import datetime

import streamlit as st

from core import config, db
from core.access import PermissionDenied
from ui import session as ui

actor = ui.require_actor()
st.title("Face enrollment", anchor=False)

MIN_F, MAX_F = int(config.get("enrollment.min_frames", 5)), int(config.get("enrollment.max_frames", 10))
latest = db.face_enrollment_status(actor, actor.id)
STATUS_TEXT = {
    "pending": ("orange", ":material/hourglass_top:", "Pending administrator approval"),
    "approved": ("green", ":material/verified:", "Approved - you can mark attendance at the kiosk"),
    "rejected": ("red", ":material/cancel:", "Rejected - please enroll again"),
    "blocked": ("red", ":material/block:", "Blocked by the duplicate-face check - contact the administrator"),
    "superseded": ("gray", ":material/history:", "Superseded"),
}
if latest:
    color, icon, text = STATUS_TEXT[latest["status"]]
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(text, icon=icon, color=color)
        st.caption(f"Submitted {latest['created_at']} · {latest['frame_count']} frames"
                   + (f" · Note: {latest['review_note']}" if latest.get("review_note") else ""))
else:
    st.badge("Not enrolled yet", icon=":material/face:", color="gray")

if actor.is_admin:
    st.info("Administrator faces are not used by the kiosk (only students are marked). To test the kiosk with "
            "your own face, enroll it on a student account instead.", icon=":material/info:")

ok, reason = db.can_start_enrollment(actor, actor.id)
state = st.session_state.get("enr")

if not ok:
    st.info(reason)
    st.stop()

if state is None:
    if latest and latest["status"] == "approved":
        st.info("You are already enrolled. A re-enrollment must be approved by the administrator; until then your "
                "current enrollment stays active.")
    with st.container(border=True):
        from vision.enrollment import CONSENT_NOTICE
        st.markdown(CONSENT_NOTICE)
        consent = st.checkbox("I have read the notice and consent to the processing of my biometric data.")
        if st.button("Start capture", type="primary", icon=":material/photo_camera:", disabled=not consent):
            st.session_state.enr = {"consent_at": datetime.now(), "embeddings": [], "metrics": [], "frames": [],
                                    "attempt": 0, "last_msg": None}
            st.rerun()
    st.stop()


@st.cache_resource(show_spinner="Loading face models...")
def _warm_models():
    from vision.recognizer import get_embedder
    get_embedder()
    return True


_warm_models()
from vision.detector import FaceDetector  # noqa: E402
from vision.enrollment import POSE_PROMPTS, decode_image, process_frame, submit_enrollment  # noqa: E402

if "enr_detector" not in st.session_state:
    st.session_state.enr_detector = FaceDetector()

n = len(state["embeddings"])
st.progress(min(1.0, n / MAX_F), text=f"Accepted frames: {n} (need {MIN_F}-{MAX_F})")
st.caption(f"Consent recorded at {state['consent_at']:%d %b %Y %H:%M:%S}")

if state["last_msg"]:
    kind, msg = state["last_msg"]
    getattr(st, kind)(msg)

left, right = st.columns([3, 2])
with left:
    if n < MAX_F:
        prompt = POSE_PROMPTS[n % len(POSE_PROMPTS)]
        st.subheader(f"Frame {n + 1}: {prompt}", anchor=False)
        shot = st.camera_input("Take a photo", key=f"enr_cam_{state['attempt']}", label_visibility="collapsed")
        if shot is not None:
            img = decode_image(shot.getvalue())
            res = process_frame(img, st.session_state.enr_detector)
            state["attempt"] += 1
            if res.report.ok:
                state["embeddings"].append(res.embedding)
                state["metrics"].append(res.report.metrics)
                if config.get("enrollment.keep_raw_images", False):
                    state["frames"].append(img)
                state["last_msg"] = ("success", f"Frame accepted (sharpness {res.report.metrics['blur']:.0f}, "
                                                f"brightness {res.report.metrics['brightness']:.0f}).")
            else:
                state["last_msg"] = ("warning", "Frame rejected - " + " ".join(res.report.reasons) + " Please retake.")
            del img
            st.rerun()
    else:
        st.success("Maximum number of frames captured.")
with right:
    with st.container(border=True):
        st.markdown("**Tips**")
        st.markdown("- Only you in the frame, face fully visible\n- Even, front lighting; no strong backlight\n"
                    "- Hold still while the photo is taken\n- Remove sunglasses / masks")
    if state["metrics"]:
        st.dataframe([{"#": i + 1, "yaw°": m["yaw"], "sharpness": m["blur"], "brightness": m["brightness"]}
                      for i, m in enumerate(state["metrics"])], hide_index=True, height=220)

with st.container(horizontal=True):
    submit = st.button("Submit for approval", type="primary", icon=":material/send:", disabled=n < MIN_F)
    if st.button("Cancel", icon=":material/close:"):
        del st.session_state["enr"]
        st.rerun()

if submit:
    with st.spinner("Checking for duplicates and saving..."):
        try:
            out = submit_enrollment(actor, actor.id, state["embeddings"], state["consent_at"], state["metrics"],
                                    state["frames"] or None)
        except (ValueError, PermissionDenied) as e:
            st.error(str(e))
            st.stop()
    del st.session_state["enr"]
    if out["status"] == "blocked":
        ui.flash("Enrollment blocked: this face matches another registered user. The administrator has been "
                 "alerted.", "error")
    else:
        ui.flash("Enrollment submitted. Embeddings stored, photos discarded. Awaiting administrator approval.")
    st.rerun()
