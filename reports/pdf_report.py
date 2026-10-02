"""PDF report (ReportLab): header, summary statistics, charts, low-attendance
list and risk list. Charts are drawn with ReportLab's own vector graphics so
no browser/image backend is required."""
from __future__ import annotations

from io import BytesIO

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from analytics.charts import STATUS_COLORS, subject_color
from core import config
from core.access import Actor
from reports.common import ReportFilters, build, date_range_label

BLUE = colors.HexColor("#1c5cab")
GRID = colors.HexColor("#c3c2b7")
INK2 = colors.HexColor("#52514e")


def _styles():
    ss = getSampleStyleSheet()
    return {
        "inst": ParagraphStyle("inst", parent=ss["Title"], fontSize=15, leading=19, textColor=BLUE, spaceAfter=0),
        "dept": ParagraphStyle("dept", parent=ss["Normal"], alignment=TA_CENTER, textColor=INK2, fontSize=10),
        "h1": ParagraphStyle("h1", parent=ss["Heading2"], fontSize=13, textColor=BLUE, spaceBefore=10, spaceAfter=4),
        "body": ss["Normal"],
        "small": ParagraphStyle("small", parent=ss["Normal"], fontSize=8, textColor=INK2),
    }


def _table(data: list[list], col_widths=None, highlight_rows: list[int] | None = None) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
             ("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fa")])]
    for r in highlight_rows or []:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f8d7d7")))
    t.setStyle(TableStyle(style))
    return t


def _subject_chart(subjects) -> Drawing:
    d = Drawing(170 * mm, 62 * mm)
    ch = VerticalBarChart()
    ch.x, ch.y, ch.width, ch.height = 12 * mm, 10 * mm, 150 * mm, 44 * mm
    ch.data = [list(subjects["attendance_pct"])]
    ch.categoryAxis.categoryNames = list(subjects["subject_code"])
    ch.categoryAxis.labels.fontSize = 7
    ch.categoryAxis.labels.fontName = "Helvetica"
    ch.valueAxis.labels.fontName = "Helvetica"
    ch.valueAxis.valueMin, ch.valueAxis.valueMax, ch.valueAxis.valueStep = 0, 100, 25
    ch.valueAxis.labels.fontSize = 7
    ch.barSpacing = 2
    for i, sid in enumerate(subjects["subject_id"]):
        ch.bars[(0, i)].fillColor = colors.HexColor(subject_color(sid))
    ch.bars.strokeColor = None
    ch.barLabelFormat = "%.1f%%"
    ch.barLabels.fontSize = 7
    ch.barLabels.fontName = "Helvetica"
    ch.barLabels.nudge = 6
    d.add(ch)
    thr = float(config.get("attendance.required_percentage", 75))
    y = ch.y + ch.height * thr / 100
    d.add(Line(ch.x, y, ch.x + ch.width, y, strokeColor=colors.HexColor("#898781"), strokeDashArray=[3, 2]))
    d.add(String(0, 58 * mm, f"Attendance % by subject (dashed line = required {thr:.0f}%)", fontSize=9,
                 fontName="Helvetica", fillColor=INK2))
    return d


def _status_pie(k: dict) -> Drawing:
    d = Drawing(170 * mm, 60 * mm)
    vals = [k["present"], k["late"], k["absent"]]
    if sum(vals) == 0:
        return d
    p = Pie()
    p.x, p.y, p.width, p.height = 30 * mm, 4 * mm, 40 * mm, 40 * mm
    p.data = vals
    total = sum(vals)
    p.labels = [f"{n} {v} ({100 * v / total:.0f}%)" for n, v in zip(["Present", "Late", "Absent"], vals)]
    p.sideLabels = True
    p.slices.fontSize = 8
    p.slices.fontName = "Helvetica"
    p.slices.strokeColor = colors.white
    p.slices.strokeWidth = 1.5
    for i, s in enumerate(["present", "late", "absent"]):
        p.slices[i].fillColor = colors.HexColor(STATUS_COLORS[s])
    d.add(p)
    d.add(String(0, 56 * mm, "Present / Late / Absent", fontSize=9, fontName="Helvetica", fillColor=INK2))
    return d


def generate(actor: Actor, filters: ReportFilters) -> bytes:
    d = build(actor, filters)
    st = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="Attendance Report", author=d.generated_by)
    thr = float(config.get("attendance.required_percentage", 75))
    el = [Paragraph(config.get("institution.name"), st["inst"]),
          Paragraph(config.get("institution.department", ""), st["dept"]),
          Paragraph("AI-Based Smart Attendance &amp; Analytics System - Attendance Report", st["dept"]),
          Spacer(1, 6),
          _table([["Subject", d.subject_label, "Period", date_range_label(d.filters)],
                  ["Student", d.student_label, "Generated", f"{d.generated_at} by {d.generated_by}"]],
                 [22 * mm, 68 * mm, 22 * mm, 68 * mm])]

    k = d.kpis
    el += [Paragraph("Summary statistics", st["h1"]),
           _table([["Overall attendance", "Sessions", "Records", "Present", "Late", "Absent", "Below threshold"],
                   [f"{k['pct']:.1f}%", k["sessions"], k["records"], k["present"], k["late"], k["absent"],
                    f"{len(d.low)} (of {len(d.student_subjects)})"]]),
           Paragraph(f"Attendance % = (Present + Late x {config.get('attendance.late_weight')}) / sessions conducted "
                     f"(closed sessions only). Required: {thr:.0f}%.", st["small"])]
    if k["records"]:
        el += [Paragraph("Charts", st["h1"]), _subject_chart(d.subjects), _status_pie(k)]

    if not d.subjects.empty:
        rows = [["Code", "Subject", "Sessions", "Present", "Late", "Absent", "Attendance %", "Below thr."]]
        rows += [[r.subject_code, r.subject_name[:34], r.sessions, r.present, r.late, r.absent,
                  f"{r.attendance_pct:.1f}", r.students_below] for r in d.subjects.itertuples()]
        el += [Paragraph("Subject summary", st["h1"]), _table(rows)]

    el.append(Paragraph(f"Low-attendance list (below {thr:.0f}%)", st["h1"]))
    if d.low.empty:
        el.append(Paragraph("No student is below the threshold for the selected filters.", st["body"]))
    else:
        rows = [["Roll no", "Name", "Subject", "Conducted", "Attendance %", "Classes needed"]]
        rows += [[r.roll_no or "-", r.full_name, r.subject_code, r.conducted, f"{r.attendance_pct:.1f}",
                  r.classes_needed if r.classes_needed >= 0 else "n/a"] for r in d.low.itertuples()]
        el.append(_table(rows, highlight_rows=list(range(1, len(rows)))))

    el.append(Paragraph("Risk list (ML prediction)", st["h1"]))
    risk = d.risk[d.risk["level"].isin(["High", "Medium"])] if not d.risk.empty else d.risk
    if risk.empty:
        el.append(Paragraph("No Medium/High risk predictions (or predictions not yet generated).", st["body"]))
    else:
        rows = [["Roll no", "Name", "Subject", "Risk", "Level", "Key factors"]]
        for r in risk.itertuples():
            factors = "; ".join(f"{x['label']}: {x['value']}" for x in r.top_factors[:2])
            rows.append([r.roll_no or "-", r.full_name, r.code, f"{r.probability:.2f}", r.level,
                         Paragraph(factors, st["small"])])
        el.append(_table(rows, [20 * mm, 38 * mm, 18 * mm, 13 * mm, 15 * mm, 76 * mm]))
        el.append(Paragraph("Risk = predicted probability that final attendance ends below the threshold "
                            "(Random Forest; rule-based estimate for students with fewer than "
                            f"{config.get('ml.min_sessions_for_model')} sessions).", st["small"]))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(INK2)
        canvas.drawString(15 * mm, 8 * mm, "Smart Attendance & Analytics System - confidential student data")
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
