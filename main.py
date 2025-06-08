# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import datetime, math, base64, bisect, os, tempfile
from io import StringIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# --- CONSTANTS & HELPERS ---
PDF_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
DATE_FORMAT = '%m/%d/%Y %H:%M'
round_qtr_hr = lambda h: math.ceil(float(h) * 4) / 4 if pd.notnull(h) and h != '' else 0.0
get_monday = lambda d: d - datetime.timedelta(days=d.weekday())
get_short_name = lambda n: str(n).strip().split(",")[0].split(" ")[0] if pd.notnull(n) else ""

# --- FONT SETUP ---
def setup_fonts():
    """Registers custom fonts for PDF generation."""
    try:
        pdfmetrics.registerFont(TTFont("SourceSansPro", os.path.join(PDF_FONT_DIR, "SourceSansPro-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("SourceSansPro-Bold", os.path.join(PDF_FONT_DIR, "SourceSansPro-Bold.ttf")))
        registerFontFamily("SourceSansPro", normal="SourceSansPro", bold="SourceSansPro-Bold")
    except Exception:
        st.warning("Source Sans Pro font not found. Using default PDF font.")

# --- DATA PROCESSING PIPELINE ---
def process_csv_to_pivot(files):
    """Processes a list of uploaded CSV files into a single pivot table of hours per client."""
    unique_files = {f.name: f for f in files}.values() # Simple deduplication
    if not unique_files: return pd.DataFrame()

    df_list = [pd.read_csv(f, header=_find_header_row(f)).pipe(_clean_and_parse) for f in unique_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    if combined_df.empty: return pd.DataFrame()

    processed_df = (combined_df
                    .pipe(_extract_and_sort_sites)
                    .pipe(_clamp_to_workday)
                    .pipe(_build_and_allocate_segments))
    
    pivot = processed_df.pivot_table(index='client', columns='date', values='hours', aggfunc='sum', fill_value=0)
    if 'Other' in pivot.index: # Move 'Other' to the end
        pivot = pd.concat([pivot.drop('Other'), pivot.loc[['Other']]])
    return pivot

def _find_header_row(file):
    """Finds the header row in a file-like object."""
    try:
        content = file.read().decode("utf-8")
        file.seek(0)
        return next(i for i, line in enumerate(content.splitlines()) if 'START_DATE*' in line)
    except StopIteration:
        st.error(f"Header row with 'START_DATE*' not found in {file.name}.")
        st.stop()

def _clean_and_parse(df):
    """Cleans raw DataFrame and parses dates."""
    df = df.loc[df['START_DATE*'].astype(str).str.match(r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}')].copy()
    df = df.drop(columns=[c for c in df.columns if any(p in c for p in ['CATEGORY', 'RATE', 'MILES'])], errors='ignore')
    df['start'] = pd.to_datetime(df['START_DATE*'], format=DATE_FORMAT)
    df['end'] = pd.to_datetime(df['END_DATE*'], format=DATE_FORMAT)
    return df

def _extract_and_sort_sites(df):
    """Extracts origin/destination sites and sorts by start time."""
    for col, prefix in [('START*', 'o'), ('STOP*', 'd')]:
        split_data = df[col].astype(str).str.rsplit('|', n=1, expand=True)
        df[[f'{prefix}n', f'{prefix}t']] = split_data
    df['origin'] = np.where(df['ot'] == 'Homeowner', df['on'], df['ot']).str.strip()
    df['destin'] = np.where(df['dt'] == 'Homeowner', df['dn'], df['dt']).str.strip()
    return df.sort_values('start').reset_index(drop=True)

def _clamp_to_workday(df):
    """Filters trips to be within the defined workday (first/last home/office trip)."""
    df['date'] = df['start'].dt.normalize()
    daily_bounds = df.groupby('date').agg(day_start=('start', 'min'), day_end=('end', 'max'))
    df = df.join(daily_bounds, on='date')
    df['cs'] = df[['start', 'day_start']].max(axis=1) # Clamped Start
    df['ce'] = df[['end', 'day_end']].min(axis=1)   # Clamped End
    df = df[df['ce'] > df['cs']].copy()
    df['dur_hr'] = (df['ce'] - df['cs']).dt.total_seconds() / 3600
    return df

def _build_and_allocate_segments(df):
    """Builds time segments (drives and gaps) and allocates them to clients."""
    if df.empty: return pd.DataFrame(columns=['date', 'client', 'hours'])
    
    recs = df.to_dict('records')
    segs = []
    # Create segments for drives and the gaps between them
    for p, c in zip(recs, recs[1:]):
        segs.append({'s': p['cs'], 'e': p['ce'], 'on': p['origin'], 'dn': p['destin'], 'ot': p['ot'], 'dt': p['dt']})
        if p['ce'].date() == c['cs'].date() and c['cs'] > p['ce']:
             segs.append({'s': p['ce'], 'e': c['cs'], 'on': p['destin'], 'dn': c['origin'], 'ot': p['dt'], 'dt': c['ot']})
    if recs: segs.append({'s': recs[-1]['cs'], 'e': recs[-1]['ce'], 'on': recs[-1]['origin'], 'dn': recs[-1]['destin'], 'ot': recs[-1]['ot'], 'dt': recs[-1]['dt']})

    # Allocate segments to clients
    home_deps = sorted([s for s in segs if s['ot']=='Homeowner'], key=lambda x:x['s'])
    home_arrs = sorted([s for s in segs if s['dt']=='Homeowner'], key=lambda x:x['e'])
    s_ts, e_ts = [s['s'] for s in home_deps], [s['e'] for s in home_arrs]
    
    allocs = []
    for s in segs:
        if any('sittler' in str(n).lower() for n in [s.get('on'), s.get('dn')]): owner = 'Other'
        elif s['ot'] == 'Homeowner': owner = s['on']
        elif s['dt'] == 'Homeowner': owner = s['dn']
        else: # For gaps, find the last client visited or the next client to be visited
            prev_idx = bisect.bisect_right(e_ts, s['s']) - 1
            next_idx = bisect.bisect_left(s_ts, s['e'])
            h1 = home_arrs[prev_idx]['dn'] if prev_idx >= 0 else None
            h2 = home_deps[next_idx]['on'] if next_idx < len(s_ts) else None
            owner = h1 or h2 or 'Other'
        allocs.append((s['s'].date(), owner, (s['e'] - s['s']).total_seconds() / 3600))
        
    return pd.DataFrame(allocs, columns=['date', 'client', 'hours'])

# --- PDF GENERATION ---
def reformat_pdf_data(pivot, week_monday):
    """Reformats pivot data for a specific week into a table-ready DataFrame."""
    week_days = [week_monday + datetime.timedelta(days=i) for i in range(6)]
    day_labels = ['M', 'Tu', 'W', 'Th', 'F', 'S']
    
    # Ensure all days of the week are columns
    df = pivot.reindex(columns=week_days, fill_value=0)
    
    # Filter to clients with hours this week
    df = df.loc[(df != 0).any(axis=1)]
    if df.empty: return pd.DataFrame(), week_days
    
    df['Subtotals'] = df.sum(axis=1)
    df.index = [str(i).title() for i in df.index]
    
    table_data = df.reset_index()
    table_data.columns = ['Client'] + day_labels + ['Subtotals']
    return table_data, week_days

def create_pdf_bytes(table_df, week_days, total_hours):
    """Generates a styled timesheet PDF and returns its bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        doc = SimpleDocTemplate(tmp.name, pagesize=landscape(letter), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Styles
        h_style = ParagraphStyle("H", fontName="SourceSansPro-Bold", fontSize=18, alignment=TA_CENTER, spaceAfter=28)
        l_style = ParagraphStyle("L", fontName="SourceSansPro", fontSize=10, spaceAfter=10)

        # Content
        total_hrs_str = f'{total_hours:g}' # Format to int if whole, else float
        elements = [
            Paragraph("HCB TIMESHEET", h_style),
            Paragraph("Employee: <b>Chad Barlow</b>", l_style),
            Paragraph(f"Week of: <b>{min(week_days).strftime('%B %-d, %Y')}</b>", l_style),
            Paragraph(f'Total Hours: <b><font backcolor="#fffac1">{total_hrs_str}</font></b>', l_style),
            Spacer(1, 0.18 * inch)
        ]

        # Table Data & Style
        data = [list(table_df.columns)] + table_df.applymap(lambda x: f'{x:g}').values.tolist()
        total_width = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
        widths = [2.8*inch] + [(total_width - 2.8*inch) / (len(data[0]) - 1)] * (len(data[0]) - 1)
        
        tbl = Table(data, colWidths=widths, repeatRows=1)
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f0f2f6")),
            ('FONTNAME', (0, 0), (-1, 0), 'SourceSansPro-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#e4e5e8")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [None, colors.HexColor("#f0f2f6")]),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ])
        tbl.setStyle(style)
        elements.append(tbl)
        
        doc.build(elements)
        tmp.seek(0)
        pdf_bytes = tmp.read()
    os.remove(tmp.name)
    return pdf_bytes

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("MileIQ Billables Processor")
setup_fonts()

uploaded_files = st.file_uploader("Upload MileIQ CSVs", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    pivot_table = process_csv_to_pivot(uploaded_files)
    
    if pivot_table.empty:
        st.warning("No valid data found in uploaded files.")
        st.stop()

    available_weeks = sorted(list(set(get_monday(c.date()) for c in pivot_table.columns)))
    today_week_monday = get_monday(datetime.date.today())
    default_selection = [w for w in [today_week_monday] if w in available_weeks] or (available_weeks[-1:])
    
    selected_weeks = st.multiselect("Select week(s) to export", available_weeks, 
                                    format_func=lambda d: d.strftime("%Y-%m-%d"), 
                                    default=default_selection)
    
    if not selected_weeks: st.info("Select at least one week to proceed."); st.stop()

    # Create name mappings for display
    full_to_short = {name: get_short_name(name) for name in pivot_table.index}
    short_to_full = {v: k for k, v in full_to_short.items()}

    for week_monday in sorted(selected_weeks):
        st.markdown(f"### Week of {week_monday:%B %d, %Y}")

        # Prepare DataFrame for the editor
        week_days = [week_monday + datetime.timedelta(days=i) for i in range(6)]
        day_labels = ['M','Tu','W','Th','F','S']
        
        editor_df = pivot_table.reindex(columns=week_days, fill_value=0)
        editor_df = editor_df.loc[(editor_df != 0).any(axis=1)].copy() # Filter rows with no hours
        editor_df.index = editor_df.index.map(full_to_short)
        editor_df.columns = day_labels
        
        # Display the editor
        edited_df = st.data_editor(editor_df.reset_index().rename(columns={'index':'Client'}),
                                   key=f"edit_{week_monday}",
                                   use_container_width=True,
                                   hide_index=True)
        
        total_hours = sum(edited_df[day_labels].sum())
        st.markdown(f"**Total Hours:** <span style='background:#fffac1;padding:2px 4px;border-radius:3px;'>{total_hours:g}</span>", unsafe_allow_html=True)
        
        # Generate PDF from edited data
        pdf_pivot = edited_df.set_index('Client')
        pdf_pivot.index = pdf_pivot.index.map(short_to_full)
        pdf_pivot.columns = week_days
        
        pdf_table_data, pdf_week_days = reformat_pdf_data(pdf_pivot, week_monday)
        
        if not pdf_table_data.empty:
            pdf_bytes = create_pdf_bytes(pdf_table_data, pdf_week_days, total_hours)
            
            st.download_button(
                label=f"Download PDF (Week of {week_monday:%Y-%m-%d})",
                data=pdf_bytes,
                file_name=f"HCB_Timesheet_{week_monday:%Y-%m-%d}.pdf",
                mime="application/pdf"
            )

            # Display PDF preview
            b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            st.markdown(f'<embed src="data:application/pdf;base64,{b64_pdf}" width="100%" height="500" type="application/pdf">', unsafe_allow_html=True)
