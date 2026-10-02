import streamlit as st

from reports import excel_report, pdf_report
from reports.common import ReportFilters, report_filename, save_copy
from ui import session as ui
from ui import widgets as w

actor = ui.require_admin()
st.title("Reports", anchor=False)
st.caption("Excel: session records, per-student and per-subject summaries. PDF: header, summary statistics, "
           "charts, low-attendance list and risk list. A copy of each report is saved under data/reports.")

f = w.filter_bar(actor, "rep", with_student=True)
filters = ReportFilters(f["date_from"], f["date_to"], f["subject_id"], f["user_id"])

with st.container(horizontal=True):
    if st.button("Generate PDF", type="primary", icon=":material/picture_as_pdf:"):
        with st.spinner("Building PDF..."):
            data = pdf_report.generate(actor, filters)
            name = report_filename("attendance_report", "pdf")
            save_copy(data, name)
            st.session_state.rep_pdf = (data, name)
    if st.button("Generate Excel", icon=":material/table_view:"):
        with st.spinner("Building Excel..."):
            data = excel_report.generate(actor, filters)
            name = report_filename("attendance_report", "xlsx")
            save_copy(data, name)
            st.session_state.rep_xlsx = (data, name)

with st.container(horizontal=True):
    if st.session_state.get("rep_pdf"):
        data, name = st.session_state.rep_pdf
        st.download_button(f"Download {name}", data, name, "application/pdf", icon=":material/download:",
                           on_click="ignore")
    if st.session_state.get("rep_xlsx"):
        data, name = st.session_state.rep_xlsx
        st.download_button(f"Download {name}", data, name,
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           icon=":material/download:", on_click="ignore")
