import streamlit as st

from reports import excel_report, pdf_report
from reports.common import ReportFilters, report_filename
from ui import session as ui
from ui import widgets as w

actor = ui.require_actor()
st.title("My attendance report", anchor=False)
st.caption("Reports contain only your own records.")

f = w.filter_bar(actor, "myrep")
filters = ReportFilters(f["date_from"], f["date_to"], f["subject_id"], actor.id)

with st.container(horizontal=True):
    if st.button("Generate PDF", icon=":material/picture_as_pdf:", type="primary"):
        with st.spinner("Building PDF..."):
            st.session_state.my_pdf = pdf_report.generate(actor, filters)
    if st.button("Generate Excel", icon=":material/table_view:"):
        with st.spinner("Building Excel..."):
            st.session_state.my_xlsx = excel_report.generate(actor, filters)

with st.container(horizontal=True):
    if st.session_state.get("my_pdf"):
        st.download_button("Download PDF", st.session_state.my_pdf, report_filename(f"attendance_{actor.username}", "pdf"),
                           "application/pdf", icon=":material/download:", on_click="ignore")
    if st.session_state.get("my_xlsx"):
        st.download_button("Download Excel", st.session_state.my_xlsx,
                           report_filename(f"attendance_{actor.username}", "xlsx"),
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           icon=":material/download:", on_click="ignore")
