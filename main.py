import streamlit as st
import pandas as pd
import bisect
import datetime
import math
import tempfile
from io import StringIO
from typing import List, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER

# -------------------------------
# App Title
# -------------------------------
st.title("MileIQ Billables Processor, Editor, and PDF Export")

# -------------------------------
# Utility: Round hours up to nearest quarter hour (0.25 increments)
# -------------------------------
def round_to_quarter_hour(hours: float) -> float:
    return math.ceil(hours * 4) / 4

# -------------------------------
# File uploader: accept multiple MileIQ CSVs
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more MileIQ CSVs",
    type=["csv"],
    accept_multiple_files=True
)

# -------------------------------
# CSV Loading and Cleaning
# -------------------------------
def load_and_clean_mileiq_csv(uploaded_file) -> pd.DataFrame:
    """
    Loads and cleans a MileIQ CSV file:
    - Finds header row with 'START_DATE*'
    - Skips to that row and reads as CSV
    - Drops rows with invalid 'START_DATE*'
    - Removes CATEGORY, RATE, MILES columns
    """
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
    # Filter to valid timestamp rows
    mask = df['START_DATE*'].astype(str).str.match(r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}')
    df = df.loc[mask].copy()
    # Drop unwanted columns
    drop_patterns = ['CATEGORY', 'RATE', 'MILES']
    cols_to_drop = [col for col in df.columns if any(pat in col for pat in drop_patterns)]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

# -------------------------------
# Timestamp Parsing
# -------------------------------
def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    fmt = '%m/%d/%Y %H:%M'
    df['start'] = pd.to_datetime(df['START_DATE*'], format=fmt)
    df['end'] = pd.to_datetime(df['END_DATE*'], format=fmt)
    return df

# -------------------------------
# Site Extraction
# -------------------------------
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

# -------------------------------
# Clamp Segments to Workday Window
# -------------------------------
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

# -------------------------------
# Build Segments (including gaps)
# -------------------------------
def build_segments(df: pd.DataFrame) -> List[Dict]:
    segments = []
    records = df.to_dict('records')
    for prev, curr in zip(records, records[1:]):
        segments.append({
            'start': prev['clamped_start'], 'end': prev['clamped_end'], 'duration': prev['duration_hr'],
            'orig_type': prev['orig_type'], 'dest_type': prev['dest_type'],
            'orig_name': prev['origin'], 'dest_name': prev['destin']
        })
        # Add gap if same-day and gap exists
        if (prev['clamped_end'].date() == curr['clamped_start'].date()
                and curr['clamped_start'] > prev['clamped_end']):
            gap_hr = (curr['clamped_start'] - prev['clamped_end']).total_seconds() / 3600
            segments.append({
                'start': prev['clamped_end'], 'end': curr['clamped_start'], 'duration': gap_hr,
                'orig_type': prev['dest_type'], 'dest_type': curr['orig_type'],
                'orig_name': prev['destin'], 'dest_name': curr['origin']
            })
    # Always append last segment
    if records:
        last = records[-1]
        segments.append({
            'start': last['clamped_start'], 'end': last['clamped_end'], 'duration': last['duration_hr'],
            'orig_type': last['orig_type'], 'dest_type': last['dest_type'],
            'orig_name': last['origin'], 'dest_name': last['destin']
        })
    return segments

# -------------------------------
# Allocate Segment Hours to Clients
# -------------------------------
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

# -------------------------------
# Pivot Allocations to Table
# -------------------------------
def pivot_billables(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df['client'] = df['client'].apply(
        lambda x: 'Other' if isinstance(x, str) and 'sittler' in x.lower() else x
    )
    pivot = df.pivot_table(
        index='client', columns='date', values='hours',
        aggfunc='sum', fill_value=0
    )
    # Ensure all columns are datetime
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

# -------------------------------
# Add DAILY TOTAL Row
# -------------------------------
def add_daily_totals_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    totals_row = {}
    for col in df.columns:
        totals_row[col] = df[col].sum() if pd.api.types.is_numeric_dtype(df[col]) else ""
    first_col = df.columns[0]
    totals_row[first_col] = "DAILY TOTAL"
    blank_row = {col: "" if col == first_col else 0.0 for col in df.columns}
    blank_df = pd.DataFrame([blank_row])
    totals_df = pd.DataFrame([totals_row])
    return pd.concat([df, blank_df, totals_df], ignore_index=True)

# -------------------------------
# Week Calculation Helpers
# -------------------------------
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

# -------------------------------
# PDF Table Formatting
# -------------------------------
def reformat_for_pdf(pivot: pd.DataFrame):
    date_cols = [col for col in pivot.columns if isinstance(col, pd.Timestamp)]
    week_order, day_labels, monday = week_columns(date_cols)
    table = pd.DataFrame(index=pivot.index)
    for d in week_order:
        table[d] = pivot[d] if d in pivot.columns else 0.0
    existing_days = [d for d in week_order if d in pivot.columns]
    table['TOTAL'] = pivot[existing_days].sum(axis=1) if existing_days else 0
    table.index = [str(idx).upper() for idx in table.index]
    col_labels = ['CLIENT'] + day_labels + ['TOTAL']
    table = table.reset_index()
    table.columns = col_labels
    if 'OTHER' in table['CLIENT'].values:
        other_row = table[table['CLIENT'] == 'OTHER']
        table = table[table['CLIENT'] != 'OTHER']
        table = pd.concat([table, other_row], ignore_index=True)
    table = add_daily_totals_row(table)
    def format_for_pdf_cell(x):
        if isinstance(x, (int, float)):
            if x == 0:
                return ""
            return str(int(x)) if x == int(x) else str(x)
        return x
    return table.applymap(format_for_pdf_cell), week_order, monday

# -------------------------------
# PDF Export with ReportLab
# -------------------------------
def export_weekly_pdf_reportlab(table_df, week_days, total_hours) -> bytes:
    tmp_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(
        tmp_fp.name,
        pagesize=landscape(letter),
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    )
    elements = [Spacer(1, 0.2 * inch)]
    # Data for table
    data = [list(table_df.columns)] + [list(row) for row in table_df.itertuples(index=False)]
    page_width = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
    num_cols = len(table_df.columns)
    client_col_w = 2.8 * inch
    other_col_w = (page_width - client_col_w) / (num_cols - 1)
    col_widths = [client_col_w] + [other_col_w] * (num_cols - 1)
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = TableStyle()
    # Header
    style.add("BACKGROUND", (0,0), (-1,0), colors.HexColor("#b6bbbf"))
    style.add("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#373737"))
    style.add("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")
    style.add("FONTSIZE", (0,0), (-1,0), 11)
    style.add("ALIGN", (0,0), (-1,0), "LEFT")
    style.add("BOTTOMPADDING", (0,0), (-1,0), 8)
    style.add("TOPPADDING", (0,0), (-1,0), 8)
    # Body
    style.add("FONTNAME", (0,1), (-1,-1), "Helvetica")
    style.add("FONTSIZE", (0,1), (-1,-1), 10)
    style.add("TEXTCOLOR", (0,1), (-1,-1), colors.HexColor("#232323"))
    style.add("ALIGN", (1,1), (-1,-1), "RIGHT")
    style.add("ALIGN", (0,1), (0,-1), "LEFT")
    style.add("TOPPADDING", (0,1), (-1,-1), 6)
    style.add("BOTTOMPADDING", (0,1), (-1,-1), 6)
    # Subtle grid
    style.add("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#ececec"))
    # Totals row
    totals_row_idx = len(data) - 1
    style.add("FONTNAME", (0, totals_row_idx), (-1, totals_row_idx), "Helvetica-Bold")
    style.add("BACKGROUND", (0, totals_row_idx), (-1, totals_row_idx), colors.HexColor("#f8f9fa"))
    style.add("TEXTCOLOR", (0, totals_row_idx), (-1, totals_row_idx), colors.HexColor("#232323"))
    style.add("ALIGN", (0, totals_row_idx), (-1, totals_row_idx), "RIGHT")
    style.add("SPAN", (0, totals_row_idx), (0, totals_row_idx))
    style.add("ALIGN", (0, totals_row_idx), (0, totals_row_idx), "LEFT")

    tbl.setStyle(style)
    elements.append(tbl)
    # Footer: WEEK OF / TOTAL HOURS
    footer_data = [[
        Paragraph(f"<b>WEEK OF:</b> {min(week_days):%Y-%m-%d}", ParagraphStyle(name="FooterLeft", fontName="Helvetica", fontSize=10)),
        Paragraph(f"<b>TOTAL HOURS:</b> {int(total_hours) if total_hours == int(total_hours) else total_hours}", ParagraphStyle(name="FooterRight", alignment=TA_CENTER, fontName="Helvetica", fontSize=10))
    ]]
    footer_tbl = Table(footer_data, colWidths=[page_width / 2, page_width / 2], hAlign="LEFT")
    footer_tbl.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    elements.append(footer_tbl)
    doc.build(elements)
    with open(tmp_fp.name, "rb") as f:
        pdf_bytes = f.read()
    return pdf_bytes

# -------------------------------
# MAIN APP LOGIC
# -------------------------------
if uploaded_files:
    all_allocs = []
    for file in uploaded_files:
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
        if not week_cols:
            st.warning(f"No data for the week of {week_start.strftime('%Y-%m-%d')}.")
            continue
        pivot_week = pivot[week_cols].copy().reset_index()
        pivot_week.columns = ['CLIENT'] + [c.strftime("%Y-%m-%d") for c in week_cols]
        week_cols_str = [c.strftime("%Y-%m-%d") for c in week_cols]
        for col in week_cols_str:
            pivot_week[col] = pivot_week[col].apply(lambda x: round_to_quarter_hour(x))
        pivot_week['TOTAL'] = pivot_week[week_cols_str].sum(axis=1)
        pivot_week_with_totals = add_daily_totals_row(pivot_week)
        def format_editor_cell(x):
            if isinstance(x, (int, float)):
                if x == 0:
                    return ""
                return str(int(x)) if x == int(x) else str(x)
            return x
        for col in week_cols_str + ['TOTAL']:
            pivot_week_with_totals[col] = pivot_week_with_totals[col].apply(format_editor_cell)
        column_config = {col: {"editable": True} for col in week_cols_str}
        st.subheader(f"Editable Billables Table for Week of {week_start.strftime('%Y-%m-%d')}")
        edited_table = st.data_editor(
            pivot_week_with_totals,
            num_rows="dynamic",
            use_container_width=True,
            key=f"pivot_edit_{week_start}",
            column_config=column_config
        )
        edited_table_clean = edited_table[
            (edited_table['CLIENT'] != "") &
            (edited_table['CLIENT'] != "DAILY TOTAL")
        ].copy()
        edited_pivot = pivot.copy()
        for col_date, col_str in zip(week_cols, week_cols_str):
            if col_str in edited_table_clean.columns:
                for row_idx, client in enumerate(edited_table_clean['CLIENT']):
                    if client in edited_pivot.index:
                        val_str = edited_table_clean.iloc[row_idx][col_str]
                        edited_pivot.loc[client, col_date] = float(val_str) if val_str != "" else 0.0
        pdf_table, week_days_out, week_start_out = reformat_for_pdf(
            edited_pivot[week_cols]
        )
        if not week_days_out or week_start_out is None:
            st.warning(f"No valid date columns found for export in week of {week_start.strftime('%Y-%m-%d')}.")
            continue
        st.subheader(f"Formatted Table (Week of {week_start.strftime('%Y-%m-%d')}, Monday–Saturday)")
        st.dataframe(pdf_table, use_container_width=True)
        total_hours = edited_pivot[week_cols].sum().sum()
        week_md = week_start.strftime("%m-%d")
        total_str = str(int(total_hours)) if total_hours == int(total_hours) else str(total_hours)
        st.markdown(f"**Total Hours for Week of {week_md}: {total_str}**")
        pdf_bytes = export_weekly_pdf_reportlab(pdf_table, week_days_out, total_hours)
        st.download_button(
            label=f"Download Billables PDF (Week of {week_start:%Y-%m-%d})",
            data=pdf_bytes,
            file_name=f"Billables_Week_of_{week_start:%Y-%m-%d}.pdf",
            mime="application/pdf",
            key=f"download_btn_{week_start}"
        )
