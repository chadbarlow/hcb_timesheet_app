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
pdfmetrics.registerFont(
    TTFont("SourceSansPro", os.path.join(FONT_DIR, "SourceSansPro-Regular.ttf"))
)
pdfmetrics.registerFont(
    TTFont("SourceSansPro-Bold", os.path.join(FONT_DIR, "SourceSansPro-Bold.ttf"))
)
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

def load_and_clean_mileiq_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.read().decode("utf-8")
    lines = content.splitlines()
    header_row = next((i for i, line in enumerate(lines) if "START_DATE*" in line), None)
    if header_row is None:
        st.error(f"Header 'START_DATE*' not found in {uploaded_file.name}.")
        st.stop()
    df = pd.read_csv(StringIO(content), skiprows=header_row)
    mask = df["START_DATE*"].astype(str).str.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}")
    df = df.loc[mask].copy()
    df = df.drop(
        columns=[c for c in df.columns if any(p in c for p in ["CATEGORY", "RATE", "MILES"])],
        errors="ignore"
    )
    return df

def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    fmt = "%m/%d/%Y %H:%M"
    df["start"] = pd.to_datetime(df["START_DATE*"], format=fmt)
    df["end"]   = pd.to_datetime(df["END_DATE*"], format=fmt)
    return df

def extract_sites(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    orig = df["START*"].astype(str).str.rsplit("|", n=1, expand=True)
    dest = df["STOP*"].astype(str).str.rsplit("|", n=1, expand=True)
    df["orig_name"], df["orig_type"] = orig[0].str.strip(), orig[1].str.strip()
    df["dest_name"], df["dest_type"] = dest[0].str.strip(), dest[1].str.strip()
    df["origin"] = df.apply(
        lambda x: x["orig_name"] if x["orig_type"] == "Homeowner" else x["orig_type"], axis=1
    )
    df["destin"] = df.apply(
        lambda x: x["dest_name"] if x["dest_type"] == "Homeowner" else x["dest_type"], axis=1
    )
    return df.sort_values("start").reset_index(drop=True)

def dynamic_clamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["start"].dt.normalize()
    depart = (df["orig_type"]=="Homeowner") | (df["origin"]=="Office")
    arrive = (df["dest_type"]=="Homeowner") | (df["destin"]=="Office")
    starts = df.loc[depart].groupby("date")["start"].min()
    ends   = df.loc[arrive].groupby("date")["end"].max()
    df = df.join(starts.rename("day_start"), on="date")
    df = df.join(ends.rename("day_end"), on="date")
    df["clamped_start"] = df[["start","day_start"]].max(axis=1)
    df["clamped_end"]   = df[["end","day_end"]].min(axis=1)
    df = df[df["clamped_end"] > df["clamped_start"]].copy()
    df["duration_hr"] = (df["clamped_end"] - df["clamped_start"]).dt.total_seconds()/3600
    return df

def build_segments(df: pd.DataFrame) -> List[Dict]:
    segs = []
    recs = df.to_dict("records")
    for a, b in zip(recs, recs[1:]):
        segs.append({
            "start": a["clamped_start"], "end": a["clamped_end"], "duration": a["duration_hr"],
            "orig_type": a["orig_type"], "dest_type": a["dest_type"],
            "orig_name": a["origin"], "dest_name": a["destin"]
        })
        if a["clamped_end"].date()==b["clamped_start"].date() and b["clamped_start"]>a["clamped_end"]:
            gap = (b["clamped_start"]-a["clamped_end"]).total_seconds()/3600
            segs.append({
                "start": a["clamped_end"], "end": b["clamped_start"], "duration": gap,
                "orig_type": a["dest_type"], "dest_type": b["orig_type"],
                "orig_name": a["destin"], "dest_name": b["origin"]
            })
    if recs:
        last = recs[-1]
        segs.append({
            "start": last["clamped_start"], "end": last["clamped_end"], "duration": last["duration_hr"],
            "orig_type": last["orig_type"], "dest_type": last["dest_type"],
            "orig_name": last["origin"], "dest_name": last["destin"]
        })
    return segs

def allocate_hours(segs: List[Dict]) -> pd.DataFrame:
    dep = sorted([s for s in segs if s["orig_type"]=="Homeowner"], key=lambda s: s["start"])
    arr = sorted([s for s in segs if s["dest_type"]=="Homeowner"], key=lambda s: s["end"])
    starts = [s["start"] for s in dep]
    ends   = [s["end"]   for s in arr]
    alloc = []
    for s in segs:
        if any("sittler" in str(s[k]).lower() for k in ["orig_name","dest_name","orig_type","dest_type"]):
            owner = "Other"
        elif s["orig_type"]=="Homeowner":
            owner = s["orig_name"]
        elif s["dest_type"]=="Homeowner":
            owner = s["dest_name"]
        else:
            i = bisect.bisect_right(ends, s["start"])-1
            j = bisect.bisect_left(starts, s["end"])
            owner = (arr[i]["dest_name"] if i>=0 else None) or (dep[j]["orig_name"] if j<len(starts) else None) or "Other"
        alloc.append((s["start"].date(), owner, s["duration"]))
    return pd.DataFrame(alloc, columns=["date","client","hours"])

def pivot_billables(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["client"] = df["client"].apply(
        lambda x: "Other" if isinstance(x,str) and "sittler" in x.lower() else x
    )
    p = df.pivot_table(index="client", columns="date", values="hours", aggfunc="sum", fill_value=0)
    p.columns = sorted(pd.to_datetime(c) for c in p.columns)
    if "Other" in p.index:
        other = p.loc[["Other"]]
        p = p.drop("Other", axis=0)
        p = pd.concat([p, other], axis=0)
    return p

def get_monday(d: datetime.date) -> datetime.date:
    return d - datetime.timedelta(days=d.weekday())

def find_available_weeks(cols):
    return sorted({get_monday(c) for c in cols})

def week_columns(cols):
    labels = ["M","Tu","W","Th","F","S"]
    days = sorted(cols)
    mon  = days[0] - datetime.timedelta(days=days[0].weekday())
    week = [mon + datetime.timedelta(days=i) for i in range(6)]
    return week, labels, mon

def reformat_for_pdf(p: pd.DataFrame):
    week_order, day_labels, monday = week_columns(p.columns)
    df = pd.DataFrame({d: p.get(d, pd.Series(0)) for d in week_order})
    df["Subtotals"] = df.sum(axis=1)
    df.index = [str(i).title() for i in p.index]
    table = df.reset_index()
    table.columns = ["Client"] + day_labels + ["Subtotals"]
    nz = table[day_labels].astype(float).any(axis=1)
    zeros = table[~nz]
    table = table[nz]
    if "Other" in zeros["Client"].values:
        other_row = zeros[zeros["Client"]=="Other"]
        table = pd.concat([table, other_row], ignore_index=True)
    return table, week_order, monday

def export_weekly_pdf_reportlab(table_df, week_days, total_hours) -> bytes:
    clean = table_df.copy()
    for c in clean.columns:
        clean[c] = clean[c].apply(lambda x: "" if isinstance(x,(int,float)) and x==0 else x)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(
        tmp.name, pagesize=landscape(letter),
        leftMargin=0.5*inch, rightMargin=0.5*inch,
        topMargin=0.5*inch, bottomMargin=0.5*inch
    )
    header = ParagraphStyle(
        "H", fontName="SourceSansPro-Bold", fontSize=18,
        alignment=TA_CENTER, spaceAfter=28,
        textColor=colors.HexColor("#31333f")
    )
    label = ParagraphStyle(
        "L", fontName="SourceSansPro", fontSize=10,
        spaceAfter=10,
        textColor=colors.HexColor("#31333f")
    )
    elems = [
        Paragraph("HCB TIMESHEET", header),
        Paragraph(f"Employee: <b>Chad Barlow</b>", label),
        Paragraph(f"Week of: <b>{min(week_days):%B %-d, %Y}</b>", label),
        Paragraph(
            f'Total Hours: <b><font backcolor="#fffac1">'
            f'{int(total_hours) if total_hours==int(total_hours) else total_hours}'
            f'</font></b>', label
        ),
        Spacer(1,0.18*inch)
    ]
    data = [list(clean.columns)] + clean.values.tolist()
    page_w = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
    widths = [2.8*inch] + [(page_w-2.8*inch)/(len(data[0])-1)]*(len(data[0])-1)
    tbl = Table(data, colWidths=widths, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f2f6")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#31333f")),
        ("TEXTCOLOR", (0,1), (-1,-1), colors.HexColor("#31333f")),
        ("FONTNAME", (0,0), (-1,0), "SourceSansPro"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("ALIGN", (0,0), (-2,0), "LEFT"),         # header all left except last col
        ("ALIGN", (-1,0), (-1,0), "RIGHT"),       # last header cell right-aligned
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
        ("TOPPADDING", (0,0), (-1,0), 8),
    
        # Add padding for body rows to match header
        ("TOPPADDING", (0,1), (-1,-1), 8),
        ("BOTTOMPADDING", (0,1), (-1,-1), 8),
    
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#e4e5e8")),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("ALIGN", (0,1), (0,-1), "LEFT"),
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

uploaded_files = st.file_uploader(
    "Upload MileIQ CSVs", type="csv", accept_multiple_files=True
)

if uploaded_files:
    seen, unique = set(), []
    for f in uploaded_files:
        key = (f.name, f.size)
        if key not in seen:
            unique.append(f)
            seen.add(key)
    if len(unique)<len(uploaded_files):
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
    default = get_monday(today) if get_monday(today) in weeks else ([weeks[-1]] if weeks else [])

    selected = st.multiselect(
        "Select week(s)", options=weeks,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        default=default
    )
    if not selected:
        st.info("Pick at least one week."); st.stop()

    def extract_short(n):
        if pd.isnull(n): return ""
        s = str(n).strip()
        return s.split(",")[0] if "," in s else s.split()[0]

    for wk in sorted(selected):
        st.markdown("---")
        days = [wk + datetime.timedelta(days=i) for i in range(6)]
        sub = pivot.reindex(columns=days, fill_value=0)
        cols = ["M","Tu","W","Th","F","S"][:len(days)]

        df_wk = sub.reset_index()
        df_wk["Client"] = df_wk["client"].apply(extract_short)
        df_wk = df_wk[["Client"] + days]
        df_wk.columns = ["Client"] + cols
        for c in cols:
            df_wk[c] = df_wk[c].apply(round_to_quarter_hour)
        df_wk = df_wk[(df_wk[cols]!=0).any(axis=1)]

        # Blank zeros for display
        display_df = df_wk.copy()
        for c in cols:
            display_df[c] = display_df[c].apply(lambda x: "" if x==0 else x)

        edited = st.data_editor(display_df, key=f"edit_{wk}", use_container_width=True)

        # Numeric total
        numeric = edited.copy()
        for c in cols:
            numeric[c] = pd.to_numeric(numeric[c], errors="coerce").fillna(0)
        total = numeric[cols].sum().sum()
        total = int(total) if total==int(total) else total

        # Apply edits back
        full = pivot.copy()
        f2s = {fn: extract_short(fn) for fn in full.index}
        s2f = {s:fn for fn,s in f2s.items()}
        for _, row in edited.iterrows():
            short = row["Client"]
            fn = s2f.get(short)
            if not fn: continue
            for cs, day in zip(cols, days):
                val = row[cs] or 0
                full.loc[fn, day] = round_to_quarter_hour(float(val))

        table_df, week_days, _ = reformat_for_pdf(full[days])
        pdf_bytes = export_weekly_pdf_reportlab(table_df, week_days, total)

        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="500px" style="border:none;"></iframe>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.download_button(
            label=f"Download PDF (Week of {wk:%Y-%m-%d})",
            data=pdf_bytes,
            file_name=f"Billables_{wk:%Y-%m-%d}.pdf",
            mime="application/pdf",
            key=f"dl_{wk}"
        )
