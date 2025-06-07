import streamlit as st
import pandas as pd
import bisect
import datetime
import math
import tempfile
from io import StringIO
from typing import List, Dict
import os
import base64

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER

# -------------------------------
# Register Source Sans Pro Fonts & Family
# -------------------------------
FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
pdfmetrics.registerFont(TTFont("SourceSansPro", os.path.join(FONT_DIR, "SourceSansPro-Regular.ttf")))
pdfmetrics.registerFont(TTFont("SourceSansPro-Bold", os.path.join(FONT_DIR, "SourceSansPro-Bold.ttf")))
registerFontFamily(
    "SourceSansPro",
    normal="SourceSansPro",
    bold="SourceSansPro-Bold",
    italic="SourceSansPro",
    boldItalic="SourceSansPro-Bold"
)

st.title("MileIQ Billables Processor, Editor, and PDF Export")

def round_to_quarter_hour(hours: float) -> float:
    if pd.isnull(hours):
        return 0.0
    return math.ceil(float(hours) * 4) / 4

# … [all your helper functions unchanged: load_and_clean_mileiq_csv, parse_timestamps, extract_sites, dynamic_clamp, build_segments, allocate_hours, pivot_billables, get_monday, find_available_weeks, week_columns, reformat_for_pdf] …

def export_weekly_pdf_reportlab(table_df, week_days, total_hours) -> bytes:
    # — blank zero cells —
    clean_df = table_df.copy()
    for col in clean_df.columns:
        clean_df[col] = clean_df[col].apply(
            lambda x: "" if isinstance(x, (int, float)) and x == 0 else x
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=landscape(letter),
                            leftMargin=0.5*inch, rightMargin=0.5*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)

    header = ParagraphStyle("H", fontName="SourceSansPro-Bold", fontSize=18,
                            alignment=TA_CENTER, spaceAfter=28, textColor=colors.HexColor("#31333f"))
    label = ParagraphStyle("L", fontName="SourceSansPro", fontSize=10,
                            spaceAfter=10, textColor=colors.HexColor("#31333f"))

    elems = [
        Paragraph("HCB TIMESHEET", header),
        Paragraph(f"Employee: <b>Chad Barlow</b>", label),
        Paragraph(f"Week of: <b>{min(week_days).strftime('%B %-d, %Y')}</b>", label),
        Paragraph(
            f'Total Hours: <b><font backcolor="#fffac1" color="#373737">'
            f'{int(total_hours) if total_hours == int(total_hours) else total_hours}'
            f'</font></b>',
            label
        ),
        Spacer(1, 0.18*inch),
    ]

    # Use clean_df (zeros now blank)
    data = [list(clean_df.columns)] + clean_df.values.tolist()
    page_w = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
    widths = [2.8*inch] + [(page_w - 2.8*inch)/(len(data[0])-1)]*(len(data[0])-1)
    tbl = Table(data, colWidths=widths, repeatRows=1)

    style = TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f0f2f6")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#81828a")),
        ("FONTNAME",(0,0),(-1,0),"SourceSansPro"),
        ("FONTSIZE",(0,0),(-1,0),10),
        ("ALIGN",(0,0),(-1,0),"LEFT"),
        ("BOTTOMPADDING",(0,0),(-1,0),8),
        ("TOPPADDING",(0,0),(-1,0),8),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#e4e5e8")),
        ("ALIGN",(1,1),(-1,-1),"RIGHT"),
        ("ALIGN",(0,1),(0,-1),"LEFT"),
    ])
    for r in range(1,len(data)):
        if r % 2 == 0:
            style.add("BACKGROUND",(0,r),(-1,r),colors.HexColor("#f0f2f6"))
    tbl.setStyle(style)

    elems.append(tbl)
    doc.build(elems)

    with open(tmp.name, "rb") as f:
        return f.read()


# ---- MAIN LOOP ----

if uploaded_files:
    # … your file‐loading & pivot logic unchanged …

    selected = st.multiselect(
        "Select week(s) to export",
        options=weeks,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        default=default
    )
    if not selected:
        st.info("Pick at least one week.")
        st.stop()

    def extract_short(n):
        if pd.isnull(n): return ""
        s = str(n).strip()
        if "," in s: return s.split(",")[0].strip()
        return s.split()[0]

    for wk in sorted(selected):
        st.markdown("---")

        days = [wk + datetime.timedelta(days=i) for i in range(6)]
        sub = pivot.reindex(columns=days, fill_value=0)
        cols = ['M','Tu','W','Th','F','S'][:len(days)]
        df_wk = sub.reset_index()
        df_wk['Client'] = df_wk['client'].apply(extract_short)
        df_wk = df_wk[['Client'] + days]
        df_wk.columns = ['Client'] + cols
        for c in cols:
            df_wk[c] = df_wk[c].apply(round_to_quarter_hour)
        df_wk = df_wk[(df_wk[cols] != 0).any(axis=1)]

        # 1) Blank zeros for display
        display_df = df_wk.copy()
        for c in cols:
            display_df[c] = display_df[c].apply(lambda x: "" if x == 0 else x)

        edited = st.data_editor(
            display_df,
            key=f"pivot_edit_{wk}",
            use_container_width=True
        )

        # 2) Convert back to numeric for totaling
        numeric_df = edited.copy()
        for c in cols:
            numeric_df[c] = pd.to_numeric(numeric_df[c], errors='coerce').fillna(0)
        total = numeric_df[cols].sum().sum()
        total = int(total) if total == int(total) else total

        # 3) Apply edits back to full pivot
        full = pivot.copy()
        map_full2short = {fn: extract_short(fn) for fn in full.index}
        map_short2full = {short:fn for fn, short in map_full2short.items()}
        for _, row in edited.iterrows():
            short = row['Client']
            fn = map_short2full.get(short)
            if not fn:
                continue
            for cs, day in zip(cols, days):
                val = row[cs] or 0
                full.loc[fn, day] = round_to_quarter_hour(float(val))

        # 4) Build PDF
        table_df, week_days, _ = reformat_for_pdf(full[days])
        pdf_bytes = export_weekly_pdf_reportlab(
            table_df,
            week_days,
            total
        )

        # 5) Inline viewer
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            'width="100%" height="500px" style="border:none;"></iframe>',
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.download_button(
            label=f"Download PDF (Week of {wk:%Y-%m-%d})",
            data=pdf_bytes,
            file_name=f"Billables_Week_of_{wk:%Y-%m-%d}.pdf",
            mime="application/pdf",
            key=f"download_btn_{wk}"
        )
