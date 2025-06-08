import streamlit as st, pandas as pd, numpy as np, bisect, datetime, math, tempfile, os, base64
from io import StringIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER

# Font setup condensed
try:
    font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.pdfmetrics import registerFont, registerFontFamily
    registerFont(TTFont("SourceSansPro", os.path.join(font_dir, "SourceSansPro-Regular.ttf")))
    registerFont(TTFont("SourceSansPro-Bold", os.path.join(font_dir, "SourceSansPro-Bold.ttf")))
    registerFontFamily("SourceSansPro", normal="SourceSansPro", bold="SourceSansPro-Bold")
except:
    st.warning("Font not found, using default.")

# Helper lambdas simplified
r_q_h = lambda h: math.ceil(float(h)*4)/4 if h else 0.0
get_mon = lambda d: d - datetime.timedelta(days=d.weekday())
ext_short = lambda n: (str(n).split(",")[0] if "," in str(n) else str(n).split(" ")[0]).strip()
find_weeks = lambda cols: sorted(set(get_mon(c) for c in cols))

# Data processing functions condensed
def dedup_files(files):
    seen = set()
    return [f for f in files if (f.name, f.size) not in seen and not seen.add((f.name, f.size))]

def load_clean(f):
    c = f.read().decode("utf-8")
    h_row = next((i for i, l in enumerate(c.splitlines()) if 'START_DATE*' in l), None)
    if h_row is None: st.error(f"Header not found in {f.name}"); st.stop()
    df = pd.read_csv(StringIO(c), skiprows=h_row)
    return df[df['START_DATE*'].str.match(r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}')].drop(columns=[col for col in df.columns if any(x in col for x in ['CATEGORY','RATE','MILES'])], errors='ignore')

def parse_ts(df):
    df['start'], df['end'] = pd.to_datetime(df['START_DATE*']), pd.to_datetime(df['END_DATE*'])
    return df

def extract_sites(df):
    o = df['START*'].astype(str).str.rsplit('|', 1, True)
    d = df['STOP*'].astype(str).str.rsplit('|', 1, True)
    df = df.assign(on=o[0].str.strip(), ot=o[1].str.strip(), dn=d[0].str.strip(), dt=d[1].str.strip())
    df = df.assign(origin=np.where(df['ot'] == 'Homeowner', df['on'], df['ot']), 
                   destin=np.where(df['dt'] == 'Homeowner', df['dn'], df['dt']))
    return df.sort_values('start').reset_index(drop=True)

def clamp(df):
    df['cs'], df['ce'] = df.groupby(df['start'].dt.date)['start'].transform('min'), df.groupby(df['end'].dt.date)['end'].transform('max')
    df['cs'], df['ce'] = df[['start','cs']].max(axis=1), df[['end','ce']].min(axis=1)
    return df[df['ce']>df['cs']].assign(dur_hr=lambda x:(x['ce']-x['cs']).dt.total_seconds()/3600)

def build_segs(df):
    segs, recs = [], df.to_dict('records')
    for p,c in zip(recs,recs[1:]):
        segs.append({'s':p['cs'],'e':p['ce'],'dur':p['dur_hr'],'on':p['origin'],'dn':p['destin'],'ot':p['START*'],'dt':p['STOP*']})
        if c['cs']>p['ce']:
            segs.append({'s':p['ce'],'e':c['cs'],'dur':(c['cs']-p['ce']).total_seconds()/3600,'on':p['destin'],'dn':c['origin'],'ot':p['STOP*'],'dt':c['START*']})
    segs.append({'s':recs[-1]['cs'],'e':recs[-1]['ce'],'dur':recs[-1]['dur_hr'],'on':recs[-1]['origin'],'dn':recs[-1]['destin'],'ot':recs[-1]['START*'],'dt':recs[-1]['STOP*']})
    return segs

def alloc_hrs(segs):
    allocs=[]
    for s in segs:
        owner = 'Other' if 'sittler' in str([s['on'],s['dn'],s['ot'],s['dt']]).lower() else (s['on'] if 'Homeowner' in s['ot'] else s['dn'])
        allocs.append((s['s'].date(),owner,s['dur']))
    return pd.DataFrame(allocs,columns=['date','client','hours'])

def pivot_bills(df):
    df['client'] = df['client'].str.replace('(?i)sittler.*','Other',regex=True)
    return df.pivot_table('hours','client','date',aggfunc='sum',fill_value=0)

# PDF generation concise
def export_pdf(tbl_df,w_days,total_h):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
        doc=SimpleDocTemplate(tmp.name,pagesize=landscape(letter),leftMargin=0.5*inch,rightMargin=0.5*inch)
        elems=[
            Paragraph("HCB TIMESHEET",ParagraphStyle("H",fontSize=18,alignment=TA_CENTER,spaceAfter=28)),
            Paragraph(f"Employee: <b>Chad Barlow</b>",ParagraphStyle("L",fontSize=10,spaceAfter=10)),
            Paragraph(f"Week of: <b>{min(w_days):%B %-d, %Y}</b>",ParagraphStyle("L",fontSize=10,spaceAfter=10)),
            Paragraph(f'Total Hours: <b>{total_h}</b>',ParagraphStyle("L",fontSize=10,spaceAfter=10)),
            Spacer(1,0.18*inch)]
        data=[list(tbl_df.columns)]+tbl_df.values.tolist()
        tbl=Table(data,repeatRows=1)
        tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,colors.grey)]))
        elems.append(tbl)
        doc.build(elems)
        with open(tmp.name,"rb") as f:pdf=f.read()
    os.remove(tmp.name)
    return pdf

# Streamlit UI simplified
st.set_page_config(layout="wide")
st.title("MileIQ Billables Processor & PDF Export")
files=st.file_uploader("Upload CSVs",type=["csv"],accept_multiple_files=True)
if files:
    all_allocs=[alloc_hrs(build_segs(clamp(extract_sites(parse_ts(load_clean(f)))))) for f in dedup_files(files)]
    p=pivot_bills(pd.concat(all_allocs))
    weeks=find_weeks(p.columns)
    sel_weeks=st.multiselect("Select weeks",weeks,default=[get_mon(datetime.date.today())] or weeks[-1:])
    for wk in sel_weeks:
        days=[wk+datetime.timedelta(days=i) for i in range(6)]
        df_wk=pd.DataFrame({'Client':p.index}).join(p[days].T.reset_index(drop=True).T.reset_index(drop=True))
        edited=st.data_editor(df_wk,key=str(wk),use_container_width=True)
        total_h=edited.iloc[:,1:].sum().sum()
        pdf_bytes=export_pdf(edited,days,total_h)
        st.download_button(f"Download PDF ({wk:%Y-%m-%d})",pdf_bytes,f"Timesheet_{wk:%Y-%m-%d}.pdf")
