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

st.title("Heartwood Custom Builders")

def round_to_quarter_hour(hours: float) -> float:
    if pd.isnull(hours):
        return 0.0
    return math.ceil(float(hours) * 4) / 4

uploaded_files = st.file_uploader(
    "Upload MileIQ CSVs",
    type=["csv"],
    accept_multiple_files=True
)

def deduplicate_files(files):
    seen = set()
    unique = []
    for f in files:
        file_id = (f.name, f.size)
        if file_id not in seen:
            unique.append(f)
            seen.add(file_id)
    return unique

def load_and_clean_mileiq_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.read().decode("utf-8")
    lines = content.splitlines()
    header_row = None
    for idx, line in enumerate(lines):
        if 'START_DATE*' in line:
            header_row = idx
            break
    if header_row is None:
        st.error(f"Header 'START_DATE*' not found in file {uploaded_file.name}.")
        st.stop()
    df = pd.read_csv(StringIO(content), skiprows=header_row)
    mask = df['START_DATE*'].astype(str).str.match(r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}')
    df = df.loc[mask].copy()
    drop_patterns = ['CATEGORY', 'RATE', 'MILES']
    cols_to_drop = [col for col in df.columns if any(pat in col for pat in drop_patterns)]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    fmt = '%m/%d/%Y %H:%M'
    df['start'] = pd.to_datetime(df['START_DATE*'], format=fmt)
    df['end'] = pd.to_datetime(df['END_DATE*'], format=fmt)
    return df

def extract_sites(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    orig = df['START*'].astype(str).str.rsplit('|', n=1, expand=True)
    dest = df['STOP*'].astype(str).str.rsplit('|', n=1, expand=True)
    df['orig_name'], df['orig_type'] = orig[0].str.strip(), orig[1].str.strip()
    df['dest_name'], df['dest_type'] = dest[0].str.strip(), dest[1].str.strip()
    df['origin'] = df.apply(
        lambda x: x['orig_name'] if x['orig_type'] == 'Homeowner' else x['orig_type'], axis=1)
    df['destin'] = df.apply(
        lambda x: x['dest_name'] if x['dest_type'] == 'Homeowner' else x['dest_type'], axis=1)
    return df.sort_values('start').reset_index(drop=True)

def dynamic_clamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = df['start'].dt.normalize()
    is_depart = (df['orig_type'] == 'Homeowner') | (df['origin'] == 'Office')
    is_arrive = (df['dest_type'] == 'Homeowner') | (df['destin'] == 'Office')
    day_starts = df.loc[is_depart].groupby('date')['start'].min()
    day_ends = df.loc[is_arrive].groupby('date')['end'].max()
    df = df.join(day_starts.rename('day_start'), on='date')
    df = df.join(day_ends.rename('day_end'), on='date')
    df['clamped_start'] = df[['start', 'day_start']].max(axis=1)
    df['clamped_end'] = df[['end', 'day_end']].min(axis=1)
    df = df[df['clamped_end'] > df['clamped_start']].copy()
    df['duration_hr'] = (df['clamped_end'] - df['clamped_start']).dt.total_seconds() / 3600
    return df

def build_segments(df: pd.DataFrame) -> List[Dict]:
    segments = []
    records = df.to_dict('records')
    for prev, curr in zip(records, records[1:]):
        segments.append({
            'start': prev['clamped_start'], 'end': prev['clamped_end'], 'duration': prev['duration_hr'],
            'orig_type': prev['orig_type'], 'dest_type': prev['dest_type'],
            'orig_name': prev['origin'], 'dest_name': prev['destin']
        })
        if (prev['clamped_end'].date() == curr['clamped_start'].date()
                and curr['clamped_start'] > prev['clamped_end']):
            gap_hr = (curr['clamped_start'] - prev['clamped_end']).total_seconds() / 3600
            segments.append({
                'start': prev['clamped_end'], 'end': curr['clamped_start'], 'duration': gap_hr,
                'orig_type': prev['dest_type'], 'dest_type': curr['orig_type'],
                'orig_name': prev['destin'], 'dest_name': curr['origin']
            })
    if records:
        last = records[-1]
        segments.append({
            'start': last['clamped_start'], 'end': last['clamped_end'], 'duration': last['duration_hr'],
            'orig_type': last['orig_type'], 'dest_type': last['dest_type'],
            'orig_name': last['origin'], 'dest_name': last['destin']
        })
    return segments

def allocate_hours(segments: List[Dict]) -> pd.DataFrame:
    home_depart = sorted([s for s in segments if s['orig_type'] == 'Homeowner'], key=lambda x: x['start'])
    home_arrive = sorted([s for s in segments if s['dest_type'] == 'Homeowner'], key=lambda x: x['end'])
    start_times = [s['start'] for s in home_depart]
    end_times = [s['end'] for s in home_arrive]
    allocs = []
    for seg in segments:
        if ('sittler' in str(seg['orig_name']).lower() or 'sittler' in str(seg['dest_name']).lower()
                or 'sittler' in str(seg['orig_type']).lower() or 'sittler' in str(seg['dest_type']).lower()):
            owner = 'Other'
        elif seg['orig_type'] == 'Homeowner':
            owner = seg['orig_name']
        elif seg['dest_type'] == 'Homeowner':
            owner = seg['dest_name']
        else:
            i = bisect.bisect_right(end_times, seg['start']) - 1
            h1 = home_arrive[i]['dest_name'] if i >= 0 else None
            j = bisect.bisect_left(start_times, seg['end'])
            h2 = home_depart[j]['orig_name'] if j < len(start_times) else None
            owner = h1 or h2 or 'Other'
        date = seg['start'].date()
        allocs.append((date, owner, seg['duration']))
    return pd.DataFrame(allocs, columns=['date', 'client', 'hours'])

def pivot_billables(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df['client'] = df['client'].apply(
        lambda x: 'Other' if isinstance(x, str) and 'sittler' in x.lower() else x
    )
    pivot = df.pivot_table(
        index='client', columns='date', values='hours',
        aggfunc='sum', fill_value=0
    )
    pivot.columns = [pd.to_datetime(c) for c in pivot.columns]
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # --- replace .append() with pd.concat() ---
    if 'Other' in pivot.index:
        other = pivot.loc[['Other']]
        pivot = pivot.drop('Other', axis=0, errors='ignore')
        pivot = pd.concat([pivot, other], axis=0)

    return pivot


def get_monday(date):
    return date - datetime.timedelta(days=date.weekday())

def find_available_weeks(date_cols):
    return sorted(set(get_monday(c) for c in date_cols))

def week_columns(dates):
    labels = ['M','Tu','W','Th','F','S']
    days = sorted(dates)
    monday = min(days) - datetime.timedelta(days=min(days).weekday())
    week = [monday + datetime.timedelta(days=i) for i in range(6)]
    return week, labels, monday

def reformat_for_pdf(pivot: pd.DataFrame):
    week_order, day_labels, monday = week_columns(pivot.columns)
    df = pd.DataFrame({d: pivot.get(d, pd.Series(0)) for d in week_order})
    df['Subtotals'] = df.sum(axis=1)
    df.index = [str(idx).title() for idx in pivot.index]
    table = df.reset_index()
    table.columns = ['Client'] + day_labels + ['Subtotals']
    # drop zero rows except Other
    data = table.copy()
    mask = (data[day_labels] != 0).any(axis=1)
    if 'Other' in data.loc[~mask, 'Client'].values:
        keep = data.loc[~mask]
        data = data.loc[mask].append(keep)
    return data, week_order, monday

def export_weekly_pdf_reportlab(table_df, week_days, total_hours) -> bytes:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=landscape(letter),
                            leftMargin=0.5*inch, rightMargin=0.5*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    header = ParagraphStyle("H", fontName="SourceSansPro-Bold", fontSize=18,
                            alignment=TA_CENTER, spaceAfter=28, textColor=colors.HexColor("#31333f"))
    label = ParagraphStyle("L", fontName="SourceSansPro", fontSize=10,
                            spaceAfter=10, textColor=colors.HexColor("#31333f"))
    elems = []
    elems.append(Paragraph("HCB TIMESHEET", header))
    elems.append(Paragraph(f"Employee: <b>Chad Barlow</b>", label))
    elems.append(Paragraph(f"Week of: <b>{min(week_days).strftime('%B %-d, %Y')}</b>", label))
    elems.append(Paragraph(
        f'Total Hours: <b><font backcolor="#fffac1" color="#373737">{int(total_hours) if total_hours == int(total_hours) else total_hours}</font></b>',
        label))
    elems.append(Spacer(1, 0.18*inch))

    data = [list(table_df.columns)] + table_df.values.tolist()
    page_w = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
    widths = [2.8*inch] + [(page_w - 2.8*inch)/(len(data[0])-1)]*(len(data[0])-1)
    tbl = Table(data, colWidths=widths, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),   colors.HexColor("#f0f2f6")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),   colors.HexColor("#31333f")),
        ("FONTNAME",      (0, 0), (-1, 0),   "SourceSansPro"),
        ("FONTSIZE",      (0, 0), (-1, 0),   10),
        ("ALIGN",         (0, 0), (-2, 0),   "LEFT"),
        ("ALIGN",         (-1, 0), (-1, 0),  "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, 0),   8),
        ("TOPPADDING",    (0, 0), (-1, 0),   8),

        ("FONTNAME",      (0, 1), (-1, -1),  "SourceSansPro"),
        ("FONTSIZE",      (0, 1), (-1, -1),  10),
        ("TEXTCOLOR",     (0, 1), (-1, -1),  colors.HexColor("#31333f")),
        ("TOPPADDING",    (0, 1), (-1, -1),  8),
        ("BOTTOMPADDING", (0, 1), (-1, -1),  8),

        ("GRID",          (0, 0), (-1, -1),  0.3, colors.HexColor("#e4e5e8")),
        ("ALIGN",         (1, 1), (-1, -1),  "RIGHT"),
        ("ALIGN",         (0, 1), (0, -1),   "LEFT"),
    ])
    for r in range(1,len(data)):
        if r%2==0:
            style.add("BACKGROUND",(0,r),(-1,r),colors.HexColor("#f0f2f6"))
    tbl.setStyle(style)
    elems.append(tbl)
    doc.build(elems)
    with open(tmp.name,"rb") as f:
        return f.read()


# ---- MAIN ----

if uploaded_files:
    unique = deduplicate_files(uploaded_files)
    if len(unique) < len(uploaded_files):
        st.warning("Duplicate files ignored.")
    all_allocs = []
    for f in unique:
        df0 = load_and_clean_mileiq_csv(f)
        df1 = parse_timestamps(df0)
        df2 = extract_sites(df1)
        df3 = dynamic_clamp(df2)
        segs = build_segments(df3)
        all_allocs.append(allocate_hours(segs))
    allocs = pd.concat(all_allocs, ignore_index=True)
    pivot = pivot_billables(allocs)
    weeks = find_available_weeks(pivot.columns)
    today = datetime.date.today()
    default = get_monday(today) if get_monday(today) in weeks else (weeks[-1:] or [])

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
        # regenerate pivot for this week
        days = [wk + datetime.timedelta(days=i) for i in range(6)]
        sub = pivot.reindex(columns=days, fill_value=0)
        cols = ['M','Tu','W','Th','F','S'][:len(days)]
        df_wk = sub.reset_index()
        df_wk['Client'] = df_wk['client'].apply(extract_short)
        df_wk = df_wk[['Client'] + days]
        df_wk.columns = ['Client'] + cols
        for c in cols:
            df_wk[c] = df_wk[c].apply(round_to_quarter_hour)
        df_wk = df_wk[ (df_wk[cols] != 0).any(axis=1) ]

        edited = st.data_editor(
            df_wk, key=f"pivot_edit_{wk}", use_container_width=True
        )

        # total
        num = edited[[*cols]].sum().sum()
        num = int(num) if num==int(num) else num
        # st.markdown(f"**Total Hours:** <span style='background:#fffac1;padding:2px'>{num}</span>",
                    # unsafe_allow_html=True)

        # regenerate full-index pivot and apply edits
        full = pivot.copy()
        # build reverse lookup
        map_full2short = {full_name: extract_short(full_name) for full_name in full.index}
        map_short2full = {short:full for full,short in map_full2short.items()}
        for i,row in edited.iterrows():
            short = row['Client']
            full_name = map_short2full.get(short)
            if not full_name: continue
            for col_short,day in zip(cols, days):
                val = row[col_short] or 0
                full.loc[full_name, day] = round_to_quarter_hour(float(val))
        
        # 1) Unpack only the table and week_days (discard week_start)
        table_df, week_days, _ = reformat_for_pdf(full[days])
        
        # 2) Call with exactly the three positional args your function expects
        pdf_bytes = export_weekly_pdf_reportlab(
            table_df,
            week_days,
            num
        )


        # # ← Back to editor
        # if st.button("← Back to editor", key=f"back_{wk}"):
        #     for k in list(st.session_state):
        #         if k.startswith("pivot_edit_"):
        #             del st.session_state[k]
        #     st.experimental_rerun()

        # Inline viewer
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        # st.markdown("**View PDF inline:**")
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
