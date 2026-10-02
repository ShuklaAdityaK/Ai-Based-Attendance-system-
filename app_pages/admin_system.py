import tempfile
from pathlib import Path

import streamlit as st

from core import audit, backup, config, db
from ui import session as ui
from ui import widgets as w

actor = ui.require_admin()
st.title("System & backup", anchor=False)

t_set, t_bak, t_log = st.tabs(["Settings", "Backup & restore", "Admin audit log"])

with t_set:
    st.caption("These values override config.yaml and take effect immediately. Every change is audit-logged.")
    current = db.get_settings(actor)
    groups: dict[str, list[str]] = {}
    for key in config.EDITABLE_SETTINGS:
        groups.setdefault(key.split(".")[0], []).append(key)
    with st.form("settings"):
        values = {}
        for group, keys in groups.items():
            st.markdown(f"**{group.title()}**")
            cols = st.columns(3)
            for i, key in enumerate(keys):
                typ, lo, hi, label = config.EDITABLE_SETTINGS[key]
                val = config.get(key)
                with cols[i % 3]:
                    if typ is int:
                        values[key] = st.number_input(label, int(lo), int(hi), int(val), key=f"set_{key}",
                                                      help=f"{key}" + (" (overridden)" if key in current else ""))
                    else:
                        values[key] = st.number_input(label, float(lo), float(hi), float(val), step=0.01,
                                                      format="%.2f", key=f"set_{key}",
                                                      help=f"{key}" + (" (overridden)" if key in current else ""))
        if st.form_submit_button("Save settings", type="primary", icon=":material/save:"):
            def _save():
                if values["attendance.late_cutoff_minutes"] < values["attendance.grace_minutes"]:
                    raise ValueError("Late cutoff must be at least the grace period.")
                changed = 0
                for key, v in values.items():
                    if v != config.get(key):
                        db.set_setting(actor, key, v)
                        changed += 1
                return changed
            ui.guarded(_save, success="Settings saved.")
    st.caption("New sessions use the current grace/cutoff values; existing sessions keep theirs.")

with t_bak:
    st.markdown("A backup is one `.zip` with the database, the trained models and the configuration. "
                "It contains biometric embeddings - store it securely.")
    if st.button("Create backup now", type="primary", icon=":material/backup:"):
        p = ui.guarded(backup.create_backup, actor, rerun=False)
        if p:
            ui.flash(f"Backup created: {p.name}")
        st.rerun()
    items = backup.list_backups(actor)
    if not items:
        w.empty_state("No backups yet.")
    for b in items[:20]:
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{b['file']}**  \n{b['modified']} · {b['size_kb']} KB")
            st.download_button("Download", Path(b["path"]).read_bytes(), b["file"], "application/zip",
                               key=f"dl_{b['file']}", icon=":material/download:", on_click="ignore")

    st.subheader("Restore", anchor=False)
    st.warning("Restoring replaces ALL current data with the backup. A safety backup of the current state is "
               "taken first. You will be logged out afterwards.", icon=":material/warning:")
    src = st.radio("Source", ["Existing backup", "Upload a .zip"], horizontal=True)
    chosen = None
    if src == "Existing backup" and items:
        chosen = st.selectbox("Backup file", [b["path"] for b in items], format_func=lambda p: Path(p).name)
    elif src == "Upload a .zip":
        up = st.file_uploader("Backup file", type=["zip"])
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / f"sa_restore_{up.file_id}.zip"
            tmp.write_bytes(up.getvalue())
            chosen = str(tmp)
    confirm = st.checkbox("I understand that current data will be replaced.")
    if st.button("Restore backup", icon=":material/restore:", disabled=not (chosen and confirm)):
        try:
            safety = backup.restore_backup(actor, Path(chosen))
        except (ValueError, OSError) as e:
            st.error(f"Restore failed: {e}")
        else:
            ui.logout(f"Backup restored. A safety copy of the previous state was saved as {safety.name}. "
                      "Please sign in again.")
            st.rerun()

with t_log:
    df = audit.admin_audit_frame(actor)
    w.df_or_empty(df, "No admin actions recorded yet.",
                  column_config={"at": "When", "actor_name": "Admin", "action": "Action", "target": "Target",
                                 "details": st.column_config.TextColumn("Details", width="large")})
