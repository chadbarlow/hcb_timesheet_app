import streamlit as st, pandas as pd, numpy as np, bisect, datetime, math, tempfile, os, base64
from io import StringIO
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER
from streamlit.components.v1 import html



# --- FONT SETUP ---
try:
    font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    pdfmetrics.registerFont(TTFont("SourceSansPro", os.path.join(font_dir, "SourceSansPro-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("SourceSansPro-Bold", os.path.join(font_dir, "SourceSansPro-Bold.ttf")))
    registerFontFamily("SourceSansPro", normal="SourceSansPro", bold="SourceSansPro-Bold", italic="SourceSansPro", boldItalic="SourceSansPro-Bold")
except Exception: # Handle cases where fonts aren't found, like in some cloud environments
    st.warning("Source Sans Pro font not found. Using default PDF font.")

# --- HELPERS ---
r_q_h = lambda h: math.ceil(float(h) * 4) / 4 if pd.notnull(h) and h != '' else 0.0
get_mon = lambda d: d - datetime.timedelta(days=d.weekday())
ext_short = lambda n: "" if pd.isnull(n) else (s.split(",")[0].strip() if "," in (s:=str(n).strip()) else s.split(" ")[0])
find_weeks = lambda cols: sorted(list(set(get_mon(c) for c in cols)))

# --- DATA PROCESSING ---
def dedup_files(files):
    seen = set()
    return [f for f in files if (f.name, f.size) not in seen and not seen.add((f.name, f.size))]
def load_clean(f):
    c = f.read().decode("utf-8")
    try: h_row = next(i for i, l in enumerate(c.splitlines()) if 'START_DATE*' in l)
    except StopIteration: st.error(f"Header not found in {f.name}."); st.stop()
    df = pd.read_csv(StringIO(c), skiprows=h_row)
    df = df.loc[df['START_DATE*'].astype(str).str.match(r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}')].copy()
    return df.drop(columns=[c for c in df.columns if any(p in c for p in ['CATEGORY', 'RATE', 'MILES'])], errors='ignore')
def parse_ts(df):
    return df.assign(start=pd.to_datetime(df['START_DATE*'], format='%m/%d/%Y %H:%M'), end=pd.to_datetime(df['END_DATE*'], format='%m/%d/%Y %H:%M'))
def extract_sites(df):
    o = df['START*'].astype(str).str.rsplit('|', n=1, expand=True)
    d = df['STOP*'].astype(str).str.rsplit('|', n=1, expand=True)
    df = df.assign(on=o[0].str.strip(), ot=o[1].str.strip(), dn=d[0].str.strip(), dt=d[1].str.strip())
    df = df.assign(origin=np.where(df['ot'] == 'Homeowner', df['on'], df['ot']), destin=np.where(df['dt'] == 'Homeowner', df['dn'], df['dt']))
    return df.sort_values('start').reset_index(drop=True)
def clamp(df):
    df = df.assign(date=df['start'].dt.normalize())
    starts = df.loc[df['ot'].eq('Homeowner') | df['origin'].eq('Office')].groupby('date')['start'].min()
    ends = df.loc[df['dt'].eq('Homeowner') | df['destin'].eq('Office')].groupby('date')['end'].max()
    df = df.join(starts.rename('day_start'), on='date').join(ends.rename('day_end'), on='date')
    df = df.assign(cs=df[['start', 'day_start']].max(axis=1), ce=df[['end', 'day_end']].min(axis=1))
    df = df[df['ce'] > df['cs']].copy()
    return df.assign(dur_hr=(df['ce'] - df['cs']).dt.total_seconds() / 3600)
def build_segs(df):
    segs, recs = [], df.to_dict('records')
    if not recs: return []
    for p, c in zip(recs, recs[1:]):
        segs.append({'s': p['cs'], 'e': p['ce'], 'dur': p['dur_hr'], 'ot': p['ot'], 'dt': p['dt'], 'on': p['origin'], 'dn': p['destin']})
        if p['ce'].date() == c['cs'].date() and c['cs'] > p['ce']:
            segs.append({'s': p['ce'], 'e': c['cs'], 'dur': (c['cs'] - p['ce']).total_seconds() / 3600, 'ot': p['dt'], 'dt': c['ot'], 'on': p['destin'], 'dn': c['origin']})
    l = recs[-1]
    segs.append({'s': l['cs'], 'e': l['ce'], 'dur': l['dur_hr'], 'ot': l['ot'], 'dt': l['dt'], 'on': l['origin'], 'dn': l['destin']})
    return segs
def alloc_hrs(segs):
    dep, arr = sorted([s for s in segs if s['ot']=='Homeowner'], key=lambda x:x['s']), sorted([s for s in segs if s['dt']=='Homeowner'], key=lambda x:x['e'])
    s_ts, e_ts = [s['s'] for s in dep], [s['e'] for s in arr]
    allocs = []
    for s in segs:
        if any('sittler' in str(n).lower() for n in [s['on'],s['dn'],s['ot'],s['dt']]): owner = 'Other'
        elif s['ot'] == 'Homeowner': owner = s['on']
        elif s['dt'] == 'Homeowner': owner = s['dn']
        else:
            h1 = arr[bisect.bisect_right(e_ts, s['s']) - 1]['dn'] if bisect.bisect_right(e_ts, s['s']) > 0 else None
            j = bisect.bisect_left(s_ts, s['e'])
            h2 = dep[j]['on'] if j < len(s_ts) else None
            owner = h1 or h2 or 'Other'
        allocs.append((s['s'].date(), owner, s['dur']))
    return pd.DataFrame(allocs, columns=['date', 'client', 'hours'])
def pivot_bills(df):
    df['date'] = pd.to_datetime(df['date'])
    df['client'] = df['client'].apply(lambda x: 'Other' if isinstance(x, str) and 'sittler' in x.lower() else x)
    p = df.pivot_table(index='client', columns='date', values='hours', aggfunc='sum', fill_value=0)
    p = p.reindex(sorted(p.columns), axis=1)
    if 'Other' in p.index: p = pd.concat([p.drop('Other', errors='ignore'), p.loc[['Other']]])
    return p

# --- PDF GENERATION ---
def reformat_pdf(p):
    # 1) turn any NaN into 0 on the pivot DataFrame
    p = p.fillna(0)

    # 2) if it’s empty, bail out
    if p.empty:
        return pd.DataFrame(), [], datetime.date.today()

    # 3) determine the Monday of the first column’s week
    mon = min(p.columns) - datetime.timedelta(days=min(p.columns).weekday())

    # 4) build the six weekdays (Mon–Sat) for that week
    w_days = [mon + datetime.timedelta(days=i) for i in range(6)]
    d_labels = ['M', 'Tu', 'W', 'Th', 'F', 'S']

    # 5) assemble a DataFrame with those days plus a Subtotals column
    df = pd.DataFrame({d: p.get(d, 0) for d in w_days})
    df['Subtotals'] = df.sum(axis=1)
    df.index = [str(i).title() for i in p.index]

    # 6) reset index & rename columns for the PDF table
    tbl = df.reset_index()
    tbl.columns = ['Client'] + d_labels + ['Subtotals']

    # 7) drop any rows with all zeros (but keep “Other” if it’s the only one)
    mask = (tbl[d_labels] != 0).any(axis=1)
    if 'Other' in tbl.loc[~mask, 'Client'].values:
        tbl = pd.concat([tbl[mask], tbl.loc[(~mask) & (tbl['Client'] == 'Other')]])
    else:
        tbl = tbl[mask]

    # 8) return the final table, the day list, and the week’s Monday
    return tbl.reset_index(drop=True), w_days, mon

def export_pdf(tbl_df, w_days, total_hrs):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        # Set up the document
        doc = SimpleDocTemplate(
            tmp.name,
            pagesize=landscape(letter),
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )

        # Header and label styles
        h_style = ParagraphStyle(
            "H",
            fontName="SourceSansPro-Bold",
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=28,
            textColor=colors.HexColor("#31333f")
        )
        l_style = ParagraphStyle(
            "L",
            fontName="SourceSansPro",
            fontSize=10,
            spaceAfter=10,
            textColor=colors.HexColor("#31333f")
        )

        # Total hours formatting
        th = int(total_hrs) if total_hrs == int(total_hrs) else total_hrs

        # Build the PDF elements list
        elems = [
            Paragraph("HCB TIMESHEET", h_style),
            Paragraph("Employee: <b>Chad Barlow</b>", l_style),
            Paragraph(f"Week of: <b>{min(w_days).strftime('%B %-d, %Y')}</b>", l_style),
            Paragraph(
                f'Total Hours: <b><font backcolor="#fffac1" color="#373737">{th}</font></b>',
                l_style
            ),
            Spacer(1, 0.18 * inch)
        ]

        # Prepare table data and column widths
        data = [list(tbl_df.columns)] + tbl_df.apply(
            lambda row: row.map(
                lambda x: int(x) if isinstance(x, float) and x == int(x) else x
            ),
            axis=1
        ).values.tolist()

        total_width = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
        first_col = 2.8 * inch
        other_width = (total_width - first_col) / (len(data[0]) - 1)
        widths = [first_col] + [other_width] * (len(data[0]) - 1)

        # Create the table
        tbl = Table(data, colWidths=widths, repeatRows=1)

        # Base table style
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#31333f")),
            ("FONTNAME", (0, 0), (-1, 0), "SourceSansPro-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("ALIGN", (0, 0), (-2, 0), "LEFT"),
            ("ALIGN", (-1, 0), (-1, 0), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("FONTNAME", (0, 1), (-1, -1), "SourceSansPro"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#31333f")),
            ("TOPPADDING", (0, 1), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e4e5e8")),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ]

        # Zebra‐stripe even rows by alternating backgrounds
        style.append((
            "ROWBACKGROUNDS",
            (0, 1),    # from first data row
            (-1, -1),  # through the last row
            [None, colors.HexColor("#f0f2f6")]
        ))

        # Apply style and build
        tbl.setStyle(TableStyle(style))
        elems.append(tbl)

        doc.build(elems)

        # Read and return PDF bytes
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()

    os.remove(tmp.name)
    return pdf_bytes


# --- UI ---
st.set_page_config(layout="wide")
st.title("MileIQ Billables Processor, Editor, and PDF Export")
files = st.file_uploader("Upload MileIQ CSVs (duplicates ignored)", type=["csv"], accept_multiple_files=True)

if files:
    unique_files = dedup_files(files)
    if len(unique_files) < len(files): st.warning("Duplicate files ignored.")
    
    all_allocs = [alloc_hrs(build_segs(clamp(extract_sites(parse_ts(load_clean(f)))))) for f in unique_files]
    p = pivot_bills(pd.concat(all_allocs, ignore_index=True))

    weeks = find_weeks(p.columns)
    if not weeks: st.warning("No valid data found in uploaded files."); st.stop()
    
    sel_weeks = st.multiselect("Select week(s) to export", weeks, format_func=lambda d: d.strftime("%Y-%m-%d"), default=[w for w in [get_mon(datetime.date.today())] if w in weeks] or (weeks[-1:]))
    if not sel_weeks: st.info("Select at least one week to proceed."); st.stop()
    
    full2short = {fn: ext_short(fn) for fn in p.index}
    short2full = {s: f for f, s in full2short.items()}

    for wk in sorted(sel_weeks):
        # Header for this week
        st.markdown(f"### Week of {wk:%B %d, %Y}")
    
        # Build list of dates and column labels
        days = [wk + datetime.timedelta(days=i) for i in range(6)]
        cols = ['M','Tu','W','Th','F','S']
    
        # --- Build the editable DataFrame ---
        df_wk = pd.DataFrame({
            'Client': [full2short.get(i) for i in p.index]
        })
        n_clients = len(p.index)
    
        for col_label, date in zip(cols, days):
            if date in p.columns:
                series = p[date]
            else:
                # if the pivot has no column for this date, use a zero Series
                series = pd.Series([0] * n_clients, index=p.index)
    
            # reset_index so it lines up, then apply rounding
            df_wk[col_label] = series.reset_index(drop=True).apply(r_q_h)
    
        # drop any client rows with all zeros
        df_wk = df_wk.loc[(df_wk[cols] != 0).any(axis=1)].reset_index(drop=True)
    
        # Let the user edit
        edited = st.data_editor(
            df_wk,
            key=f"edit_{wk}",
            use_container_width=True,
            hide_index=True
        )
    
        # Compute total hours
        total_h = edited[cols].sum().sum()
        th_str = int(total_h) if total_h == int(total_h) else round(total_h, 2)
        st.markdown(
            f"**Total Hours:** "
            f"<span style='background:#fffac1;color:black;"
            f"padding:2px 4px;border-radius:3px;'>{th_str}</span>",
            unsafe_allow_html=True
        )
    
        # Apply edits back into a copy of the pivot
        p_copy = p.copy()
        for _, row in edited.iterrows():
            fn = short2full.get(row['Client'])
            if fn:
                for col_label, date in zip(cols, days):
                    p_copy.loc[fn, date] = r_q_h(row[col_label] or 0)
    
        # Reformat for PDF
        tbl_df, w_days, _ = reformat_pdf(
            p_copy.reindex(columns=days, fill_value=0)
        )
        if tbl_df.empty and total_h > 0:
            # fallback in case everyone’s hours were on weekend
            full_week = [wk + datetime.timedelta(days=i) for i in range(7)]
            tbl_df, w_days, _ = reformat_pdf(
                p_copy.reindex(columns=full_week, fill_value=0)
            )
    
        # Generate and display PDF
        pdf_bytes = export_pdf(tbl_df, w_days, total_h)
        b64 = base64.b64encode(pdf_bytes).decode()
    
        # … above this you already have pdf_bytes and b64 …
        st.download_button(
            label=f"Download PDF (Week of {wk:%Y-%m-%d})",
            data=pdf_bytes,
            file_name=f"HCB_Timesheet_{wk:%Y-%m-%d}.pdf",
            mime="application/pdf",
            key=f"dl_{wk}"
        )
        
        # Use <embed> instead of <iframe> so Chrome desktop will allow it
        # … after your download_button …
        
        # Use <embed> instead of <iframe> so Chrome desktop will allow it
        embed_html = f'''
        <embed
          src="data:application/pdf;base64,{b64}"
          type="application/pdf"
          width="100%"
          height="600px"
        />
        '''
        
        # Render it via the html component
        html(embed_html, height=620)
        
        st.markdown("---")

