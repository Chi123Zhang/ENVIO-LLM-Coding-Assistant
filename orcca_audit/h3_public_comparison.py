"""Entirely fictional original-source fixtures for the public H3 demonstration."""
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from openpyxl import Workbook
from reportlab.pdfgen.canvas import Canvas
import streamlit as st

from .exports import safe_csv
from .h3_sources import MENUS, BRANCH, CHOICES


@st.cache_data(show_spinner=False)
def full_demo_sources():
    rng=np.random.default_rng(7331)
    tracker_book=Workbook();tracker=tracker_book.active;tracker.title='orcca_tracker_v2'
    lookup=tracker_book.create_sheet('Sheet1');lookup.append(['ID','County'])
    headers=['hcs_orc2record_id','redcap_event_name','hcs_orc2version','hcs_orc2a',
             'hcs_orc2b','hcs_orc2b1','hcs_orc2b2','hcs_orc2b3']
    for _,(n,start,stop,venue,start_date,_) in MENUS.items():
        base='hcs_orc2'+n
        for slot in range(1,5):
            headers.extend([f'{base}_{slot}',f'{base}{start}_{slot}',f'{base}{stop}_{slot}',
                            f'{base}{venue}_{slot}',f'{base}{start_date}_{slot}_yr',f'{base}{start_date}_{slot}_mo'])
    tracker.append(headers)
    dd_book=Workbook();dd=dd_book.active;dd.title='ORCCAT_DD'
    dd.append(['Version','Variable / Field Name','Field Label',BRANCH,CHOICES])
    for field in ['hcs_orc2b','hcs_orc2b1','hcs_orc2b2','hcs_orc2b3']:
        dd.append(['FICTIONAL_V1',field,"Have there been any changes to last month's ORCCAT entries?",'',
                   '1, Yes, progress is reported. | 0, No, there are no changes to report since last month.'])
    for _,(n,start,stop,venue,_,_) in MENUS.items():
        base='hcs_orc2'+n
        for slot in range(1,5):
            for suffix,label,choices in [(start,'Has implementation started?','1, Yes | 0, No'),
                (stop,'Has this strategy been discontinued?','1, Yes | 0, No'),
                (venue,'Venue','1, Fictional site A | 2, Fictional site B | 3, Fictional site C')]:
                dd.append(['FICTIONAL_V1',f'{base}{suffix}_{slot}',label,'',choices])
    monthly=[];annual=[];attendance=[]
    for cid in range(1,7):
        county=f'DEMO_{cid:02d}';lookup.append([cid,county]);fips=f'99{cid:03d}'
        total=0
        for month in range(1,13):
            deaths=int(rng.poisson(13+cid*.8));total+=deaths
            monthly.append({'County':county,'County Code':fips,'Month Code':f'2024/{month:02d}','Deaths':deaths})
            if month<=11:
                update=month in (1,4,7,10)
                row=dict(hcs_orc2record_id=cid,redcap_event_name=f'{pd.Timestamp(2024,month,1):%B}_2024_arm_1'.lower(),
                         hcs_orc2version='FICTIONAL_V1',hcs_orc2a=1,hcs_orc2b=int(update),
                         hcs_orc2b1=1 if update else None,hcs_orc2b2=1 if update else None,hcs_orc2b3=1 if update else None)
                if update:
                    for j,(_,(n,start,stop,venue,start_date,_)) in enumerate(MENUS.items()):
                        base='hcs_orc2'+n;active=1+(cid+month//3+j)%4
                        for slot in range(1,5):
                            on=int(slot<=active)
                            row.update({f'{base}_{slot}':1,f'{base}{start}_{slot}':on,f'{base}{stop}_{slot}':0,
                                f'{base}{venue}_{slot}':1+(slot+cid+j)%3,
                                f'{base}{start_date}_{slot}_yr':2024 if on else None,
                                f'{base}{start_date}_{slot}_mo':1 if on else None})
                tracker.append([row.get(h) for h in headers])
        annual.append({'County':county,'County Code':fips,'Year Code':2024,'Population':60000+cid*3000,'Deaths':total})
        for period in pd.period_range('2023-11','2024-10',freq='M'):
            if cid==1 and str(period)=='2024-04':
                continue # Deliberate fictional gap, documented by the bundled PDF.
            n=int(rng.integers(7,22))
            for person in range(n):
                attendance.append({'county':county,'date':str(period)+'-10',
                    'Name':f'Fictional person {person+1:02d}','Attendance':'present'})
    files={}
    for role,book in [('tracker',tracker_book),('dictionary',dd_book)]:
        stream=BytesIO();book.save(stream);files[role]=(f'fictional_{role}.xlsx',stream.getvalue())
    for role,rows in [('monthly',monthly),('annual',annual)]:
        files[role]=(f'fictional_{role}.tsv',pd.DataFrame(rows).to_csv(index=False,sep='\t').encode())
    return files,[('fictional_attendance.csv',safe_csv(attendance))]


@st.cache_data(show_spinner=False)
def demo_meeting_pdf():
    stream=BytesIO();pdf=Canvas(stream)
    for i,line in enumerate(['SYNTHETIC SOFTWARE DEMONSTRATION ONLY','County: DEMO_01. Meeting date: 2024-04-10.',
        'Coalition meeting attendance: 12 distinct people explicitly marked present.',
        'This is the only observed coalition meeting in this fictional month.',
        'All names, dates and counts in this file are fictional.']):
        pdf.drawString(45,760-25*i,line)
    pdf.save()
    return 'DEMO_01_2024-04-10.pdf',stream.getvalue()


def demo_input_archive():
    files,attendance=full_demo_sources();output=BytesIO()
    with ZipFile(output,'w',ZIP_DEFLATED) as z:
        for name,raw in [*files.values(),*attendance,demo_meeting_pdf()]:z.writestr(name,raw)
        z.writestr('README.txt','All inputs are fictional. No research observations or human labels are included. Use the Full H3 comparison demo in Workflow > H3 source reconstruction. Synthetic model results have no research interpretation.')
    return output.getvalue()


def render_demo_evidence(archive):
    from .h3_attendance_evidence import build_evidence_review
    st.subheader('Retrieve evidence for the fictional attendance gap')
    st.caption('BM25 uses the original ENVIO tokenizer and ranking on a bundled fictional meeting PDF. '
               'No real research files are loaded and no LLM is called. Source text and page references are retained.')
    stored=st.session_state.get('h3_full_demo_evidence')
    import hashlib
    fingerprint=hashlib.sha256(archive).hexdigest()
    if stored and stored[0]!=fingerprint:
        st.session_state.pop('h3_full_demo_evidence',None);stored=None
    if st.button('Retrieve synthetic attendance evidence',key='h3_full_demo_retrieve'):
        stored=fingerprint,build_evidence_review(archive,[demo_meeting_pdf()],synthetic=True)
        st.session_state['h3_full_demo_evidence']=stored
    if stored:
        manifest,queue,logs,memory,report=stored[1]
        st.write(f"**{manifest['actual_bm25_queries']} BM25 query with matched fictional source evidence.**")
        for record in logs:
            for doc in record['citations']:
                st.markdown(f"**{doc['source_file']} — page {doc['page']}**")
                st.text(doc['text'])
        st.caption('The evidence states a fictional count. It remains unapproved and is not inserted into a model. '
                   'Use Evidence Completion for the separate synthetic review exercise.')
        st.download_button('Download synthetic evidence review ZIP',report,'synthetic_h3_evidence_review.zip',
                           'application/zip',key='h3_full_demo_evidence_download')
