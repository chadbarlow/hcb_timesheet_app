import streamlit as st
import pandas as pd
import bisect
import datetime
import math
import tempfile
from io import StringIO
from typing import List, Dict
import os

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

uploaded_files = st.file_uploader(
    "Upload one or more MileIQ CSVs (duplicate files will be ignored)",
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
    new_cols = []
    for col in pivot.columns:
        try:
            new_cols.append(pd.to_datetime(col))
        except Exception:
            new_cols.append(col)
    pivot.columns = new_cols
    date_cols = [c for c in pivot.columns if isinstance(c, pd.Timestamp)]
    other_cols = [c for c in pivot.columns if not isinstance(c, pd.Timestamp)]
    pivot = pivot[[*sorted(date_cols), *other_cols]]
    if 'Other' in pivot.index:
        others = pivot.loc[['Other']]
        pivot = pivot.drop('Other', errors='ignore')
        pivot = pd.concat([pivot, others])
    return pivot

def get_monday(date):
    return date - datetime.timedelta(days=date.weekday())

def find_available_weeks(date_cols):
    return sorted(set(get_monday(col) for col in date_cols))

def week_columns(dates):
    col_map = {0: 'M', 1: 'Tu', 2: 'W', 3: 'Th', 4: 'F', 5: 'S'}
    days = sorted(dates)
    if not days:
        return [], [], None
    monday = min(days) - datetime.timedelta(days=min(days).weekday())
    col_order = [monday + datetime.timedelta(days=i) for i in range(6)]
    return col_order, [col_map[i] for i in range(6)], monday

def reformat_for_pdf(pivot: pd.DataFrame):
    date_cols = [col for col in pivot.columns if isinstance(col, pd.Timestamp)]
    week_order, day_labels, monday = week_columns(date_cols)
    table = pd.DataFrame(index=pivot.index)
    for d in week_order:
        table[d] = pivot[d].apply(round_to_quarter_hour) if d in pivot.columns else 0.0
    table['Subtotals'] = table[week_order].sum(axis=1)
    table.index = [str(idx).title() for idx in table.index]
    col_labels = ['Client'] + day_labels + ['Subtotals']
    table = table.reset_index()
    table.columns = col_labels
    data_cols = day_labels
    def has_nonzero(row):
        for x in row:
            try:
                if float(x) != 0:
                    return True
            except:
                if x not in ("", "0"):
                    return True
        return False
    table = table[table[data_cols].apply(has_nonzero, axis=1)].reset_index(drop=True)
    if 'Other' in table['Client'].values:
        other_row = table[table['Client'] == 'Other']
        table = table[table['Client'] != 'Other']
        table = pd.concat([table, other_row], ignore_index=True)
    def format_for_pdf_cell(x):
        if isinstance(x, (int, float)):
            if x == 0:
                return ""
            return str(int(x)) if x == int(x) else str(x)
        return x
    return table.applymap(format_for_pdf_cell), week_order, monday

def export_weekly_pdf_reportlab(table_df, week_days, total_hours) -> bytes:
    tmp_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(
        tmp_fp.name,
        pagesize=landscape(letter),
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    )
    header_style = ParagraphStyle(
        name="Header",
        fontName="SourceSansPro-Bold",
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=28,
        textColor=colors.HexColor("#31333f"),
    )
    label_style = ParagraphStyle(
        name="Label",
        fontName="SourceSansPro",
        fontSize=10,
        alignment=0,
        spaceAfter=10,
        textColor=colors.HexColor("#31333f"),
    )
    elements = []
    elements.append(Paragraph("HCB TIMESHEET", header_style))
    week_of_str = f"Week of: <b>{min(week_days).strftime('%B %-d, %Y')}</b>"
    total_hours_str = (
        f'Total Hours: <b><font backcolor="#fffac1" color="#373737">{int(total_hours) if total_hours == int(total_hours) else total_hours}</font></b>'
    )
    employee_name_str = f"Employee: <b>Chad Barlow</b>"
    elements.append(Paragraph(employee_name_str, label_style))
    elements.append(Paragraph(week_of_str, label_style))
    elements.append(Paragraph(total_hours_str, label_style))
    elements.append(Spacer(1, 0.18 * inch))
    data = [list(table_df.columns)] + [list(row) for row in table_df.itertuples(index=False)]
    page_width = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
    num_cols = len(table_df.columns)
    client_col_w = 2.8 * inch
    other_col_w = (page_width - client_col_w) / (num_cols - 1)
    col_widths = [client_col_w] + [other_col_w] * (num_cols - 1)
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = TableStyle()
    style.add("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f6"))
    style.add("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#81828a"))
    style.add("FONTNAME", (0, 0), (-1, 0), "SourceSansPro")
    style.add("FONTSIZE", (0, 0), (-1, 0), 10)
    style.add("ALIGN", (0, 0), (-1, 0), "LEFT")
    style.add("BOTTOMPADDING", (0, 0), (-1, 0), 8)
    style.add("TOPPADDING", (0, 0), (-1, 0), 8)
    style.add("FONTNAME", (0, 1), (-1, -1), "SourceSansPro")
    style.add("FONTSIZE", (0, 1), (-1, -1), 10)
    style.add("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#31333f"))
    style.add("ALIGN", (1, 1), (-1, -1), "RIGHT")
    style.add("ALIGN", (0, 1), (0, -1), "LEFT")
    style.add("TOPPADDING", (0, 1), (-1, -1), 6)
    style.add("BOTTOMPADDING", (0, 1), (-1, -1), 6)
    for row_idx in range(1, len(data)):
        if row_idx % 2 == 0:
            style.add("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#f0f2f6"))
    subtotals_col_idx = len(data[0]) - 1
    style.add("ALIGN", (subtotals_col_idx, 0), (subtotals_col_idx, -1), "RIGHT")
    style.add("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e4e5e8"))
    tbl.setStyle(style)
    elements.append(tbl)
    doc.build(elements)
    with open(tmp_fp.name, "rb") as f:
        pdf_bytes = f.read()
    return pdf_bytes

# -------------- MAIN LOGIC ---------------

if uploaded_files:
    unique_files = deduplicate_files(uploaded_files)
    if len(unique_files) < len(uploaded_files):
        st.warning("Duplicate files detected by name and size. Only unique files will be processed.")
    files_to_process = unique_files
    all_allocs = []
    for file in files_to_process:
        df_raw = load_and_clean_mileiq_csv(file)
        df_times = parse_timestamps(df_raw)
        df_sites = extract_sites(df_times)
        df_clamp = dynamic_clamp(df_sites)
        segs = build_segments(df_clamp)
        alloc = allocate_hours(segs)
        all_allocs.append(alloc)
    allocs_combined = pd.concat(all_allocs, ignore_index=True)
    pivot = pivot_billables(allocs_combined)
    date_cols = [col for col in pivot.columns if isinstance(col, pd.Timestamp)]
    available_weeks = find_available_weeks(date_cols)
    today = datetime.date.today()
    current_monday = get_monday(today)
    default_weeks = (
        [current_monday] if current_monday in available_weeks
        else [max(available_weeks)] if available_weeks else []
    )
    selected_weeks = st.multiselect(
        "Select week(s) to export (Monday start)",
        options=available_weeks,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        default=default_weeks
    )
    if not selected_weeks:
        st.info("Select at least one week to proceed.")
        st.stop()
    for idx, week_start in enumerate(sorted(selected_weeks)):
        if idx > 0:
            st.markdown("---")
        week_days = [week_start + datetime.timedelta(days=i) for i in range(6)]
        week_cols = [col for col in week_days if col in pivot.columns]
        all_day_short_names = ['M', 'Tu', 'W', 'Th', 'F', 'S']
        num_days = len(week_cols)
        used_short_names = all_day_short_names[:num_days]
        # Build DataFrame with day short names as headers
        pivot_week = pivot[week_cols].copy().reset_index()
        pivot_week.columns = ['Client'] + used_short_names

        # Replace full names with last names (or pre-comma, or first word for "Smith John", never after comma)
        def extract_last_name(name):
            if pd.isnull(name):
                return ""
            s = str(name).strip()
            if s.lower() == "other":
                return "Other"
            # If name contains a comma, use the part before the comma and strip
            if "," in s:
                last = s.split(",")[0].strip()
                return last
            # Otherwise, use the first word (for "Smith John")
            parts = s.split()
            return parts[0] if parts else s

        pivot_week['Client'] = pivot_week['Client'].apply(extract_last_name)

        # Round ALL hours to quarter-hour (for display and for editing)
        for col in used_short_names:
            pivot_week[col] = pivot_week[col].apply(round_to_quarter_hour)

        def row_has_hours(row):
            for val in row:
                try:
                    if float(val) != 0:
                        return True
                except:
                    if val not in ("", "0"):
                        return True
            return False

        pivot_week_nozero = pivot_week[pivot_week[used_short_names].apply(row_has_hours, axis=1)].reset_index(drop=True)

        st.subheader(f"Week of {week_start.strftime('%B %-d')}")

        column_config = {col: {"editable": True} for col in used_short_names}
        edited_table = st.data_editor(
            pivot_week_nozero,
            num_rows="dynamic",
            use_container_width=True,
            key=f"pivot_edit_{week_start}",
            column_config=column_config
        )
        # --- Dynamically calculate total_hours from the edited table ---
        edited_table_clean = edited_table[(edited_table['Client'] != "")].copy()
        for col in used_short_names:
            edited_table_clean[col] = pd.to_numeric(edited_table_clean[col], errors='coerce').fillna(0).apply(round_to_quarter_hour)
        total_hours = edited_table_clean[used_short_names].sum().sum()
        yellow_bg = "#fffac1"
        st.markdown(
            f'<b>Total Hours:</b> <span style="background-color: {yellow_bg}; padding:2px 6px;">{int(total_hours) if total_hours == int(total_hours) else total_hours}</span>',
            unsafe_allow_html=True
        )

        # --- Update pivot for PDF/Formatted Table using the edited table values ---
        edited_pivot = pivot.copy()
        for col_date, col_short in zip(week_cols, used_short_names):
            if col_short in edited_table_clean.columns:
                for row_idx, client in enumerate(edited_table_clean['Client']):
                    if client in edited_pivot.index:
                        val_str = edited_table_clean.iloc[row_idx][col_short]
                        val = float(val_str) if val_str != "" else 0.0
                        # Round before saving!
                        val = round_to_quarter_hour(val)
                        edited_pivot.loc[client, col_date] = val

        pdf_input = edited_pivot[week_cols]
        pdf_table, week_days_out, week_start_out = reformat_for_pdf(pdf_input)
        if not week_days_out or week_start_out is None:
            st.warning(f"No valid date columns found for export in week of {week_start.strftime('%Y-%m-%d')}.")
            continue

        with st.expander(f"Show Formatted Table"):
            st.subheader(f"Formatted Table (Week of {week_start.strftime('%Y-%m-%d')}, Monday–Saturday)")
            st.dataframe(pdf_table, use_container_width=True)

        pdf_bytes = export_weekly_pdf_reportlab(pdf_table, week_days_out, total_hours)
        st.download_button(
            label=f"Download Billables PDF (Week of {week_start:%Y-%m-%d})",
            data=pdf_bytes,
            file_name=f"Billables_Week_of_{week_start:%Y-%m-%d}.pdf",
            mime="application/pdf",
            key=f"download_btn_{week_start}"
        )
