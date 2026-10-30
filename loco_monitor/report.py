"""PDF report generation."""
from __future__ import annotations

import os
from fpdf import FPDF
from .config import BASE_DIR, DETECTION_DIR


def strip_emoji(t):
    return t.encode('latin-1', 'ignore').decode('latin-1') if t else t

def _fmt_report_seconds(seconds):
    if seconds is None:
        return 'N/A'
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:02d}'

class CrewReportPDF(FPDF):
    def __init__(self, video_info):
        super().__init__()
        self._vi = video_info
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(30, 64, 175)
        self.cell(0, 10, 'Crew Behavior Monitoring Report', align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 9)
        self.set_text_color(120, 120, 120)
        vi = self._vi
        self.cell(
            0,
            5,
            strip_emoji(
                f"Generated: {vi.get('analysis_time', 'N/A')}  |  File: {vi.get('filename', 'N/A')}  "
                f"|  Duration: {vi.get('duration', 0)}s  |  Frames: {vi.get('frames_analyzed', 0)}"
            ),
            align='C',
            new_x='LMARGIN',
            new_y='NEXT'
        )
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Qwen2.5-VL + YOLOv8 + MediaPipe  |  Page {self.page_no()}/{{nb}}', align='C')

def generate_pdf_report(parameters, overall_summary, recommendations, video_info):
    pdf = CrewReportPDF(video_info)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    def status_color(key, st):
        s = st.upper()
        if s == 'LOW' or (key in ('nap', 'mobile') and s == 'OBSERVED'):
            return (220, 38, 38)
        if s in ('NOT_OBSERVED', 'NONE') or (key == 'alertness' and s == 'HIGH'):
            return (34, 197, 94)
        if s == 'OBSERVED':
            return (59, 130, 246)
        return (245, 158, 11)

    for n, (key, p) in enumerate(parameters.items(), 1):
        if pdf.get_y() > 220:
            pdf.add_page()
        pdf.set_fill_color(24, 30, 50)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, strip_emoji(f'  #{n}  {p.get("label", key)}'), fill=True, new_x='LMARGIN', new_y='NEXT')
        st = p.get('status', 'N/A')
        r, g, b = status_color(key, st)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(r, g, b)
        pdf.cell(22, 7, '  Status: ', new_x='END')
        pdf.cell(0, 7, st, new_x='LMARGIN', new_y='NEXT')

        if p.get('description'):
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 5, strip_emoji(p['description']), new_x='LMARGIN', new_y='NEXT')

        details = [f"Confidence: {p.get('confidence', 'N/A')}"]
        if p.get('evidence_timestamp_s') is not None:
            details.append(f"Evidence time: {_fmt_report_seconds(p.get('evidence_timestamp_s'))}")
        if p.get('event_duration_s'):
            details.append(f"Event duration: {_fmt_report_seconds(p.get('event_duration_s'))}")
        if p.get('evidence_source'):
            details.append(f"Evidence source: {str(p.get('evidence_source')).replace('_', ' ')}")
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(0, 5, '  |  '.join(details), new_x='LMARGIN', new_y='NEXT')

        img_url = p.get('image_url')
        if img_url:
            img_path = str(BASE_DIR / img_url.lstrip("/"))
            if os.path.exists(img_path):
                try:
                    if pdf.get_y() > 210:
                        pdf.add_page()
                    pdf.image(img_path, x=15, w=80, h=55)
                    pdf.ln(2)
                except Exception:
                    pass
        pdf.ln(4)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    if overall_summary:
        if pdf.get_y() > 230:
            pdf.add_page()
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, '  Overall Safety Assessment', fill=True, new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(50, 50, 50)
        pdf.ln(2)
        pdf.multi_cell(0, 5, strip_emoji(overall_summary), new_x='LMARGIN', new_y='NEXT')
        pdf.ln(4)

    if recommendations:
        if pdf.get_y() > 230:
            pdf.add_page()
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, '  Recommendations', fill=True, new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(50, 50, 50)
        pdf.ln(2)
        pdf.multi_cell(0, 5, strip_emoji(recommendations), new_x='LMARGIN', new_y='NEXT')

    pdf_path = f'{DETECTION_DIR}/crew_monitoring_report.pdf'
    pdf.output(pdf_path)
    return pdf_path
