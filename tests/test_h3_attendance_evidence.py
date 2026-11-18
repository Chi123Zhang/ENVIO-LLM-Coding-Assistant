from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import pytest
from reportlab.pdfgen.canvas import Canvas

from orcca_audit.engine import AuditError
from orcca_audit.h3_attendance_evidence import build_evidence_review, apply_attendance_reviews
from orcca_audit.h3_attendance_evidence_ui import compare_approved


def fixture():
    rows=[]
    for county in ['DEMO_A', 'DEMO_B']:
        for menu in ['OEND', 'MOUD']:
            rows.append(dict(county=county, menu=menu, attendance_month='2024-01', outcome_month='2024-03',
                attendance=None if county=='DEMO_A' else 8, active_snapshot=1, venue_types=1,
                outcome_available=True, tracker_snapshot_available=True, attendance_available=county=='DEMO_B',
                all_three_available=county=='DEMO_B', venue_path_available=county=='DEMO_B'))
    stream=BytesIO()
    with ZipFile(stream,'w',ZIP_DEFLATED) as z:
        z.writestr('county_month.csv',pd.DataFrame(rows).to_csv(index=False))
        z.writestr('run_metadata.json',json.dumps(dict(classification='private research',county_months=2,counties=2)))
    raw=BytesIO();pdf=Canvas(raw);pdf.drawString(50,750,'Attendance at the coalition meeting: 12 people explicitly marked present.');pdf.save()
    archive=stream.getvalue()
    result=build_evidence_review(archive,[('DEMO_A_2024-01-15.pdf',raw.getvalue())])
    with ZipFile(BytesIO(result[4])) as z:
        reviews=pd.read_csv(z.open('A_independent_reviewer.csv'),dtype=str,keep_default_na=False)
    return archive,result,reviews


def approved(reviews,memory):
    reviews=reviews.copy();doc=memory['documents'][0]
    reviews.loc[0,['decision','reviewer','reviewer_type','definition_confirmed','review_reason']]=[
        'approved','Synthetic test reviewer','human','yes','Software fixture only; not a research label.']
    reviews.loc[0,'meeting_counts_json']=json.dumps([dict(date='2024-01-15',count=12,
        chunk_id=doc['chunk_id'],quote='12 people explicitly marked present.')])
    return reviews


def test_retrieval_never_fills_or_writes_human_answers():
    archive,result,reviews=fixture()
    manifest,queue,logs,memory,report=result
    assert manifest['actual_bm25_queries']==1
    assert queue.proposed_attendance.isna().all()
    assert all(not row for row in reviews.decision)
    assert all(not row for row in reviews.reviewer)
    assert logs[0]['citations'][0]['page']==1
    original=sha256(archive).hexdigest()
    data,ledger,derived=apply_attendance_reviews(archive,memory,reviews)
    assert not ledger and data.loc[data.county.eq('DEMO_A'),'attendance'].isna().all()
    assert sha256(archive).hexdigest()==original
    with pytest.raises(AuditError,match='No approved'):
        compare_approved(archive,memory,reviews)


def test_review_changes_missing_shared_exposure_only():
    archive,result,reviews=fixture();memory=result[3]
    data,ledger,derived=apply_attendance_reviews(archive,memory,approved(reviews,memory))
    assert len(ledger)==1 and len(data)==4
    assert set(data[data.county.eq('DEMO_A')].attendance)=={12}
    assert set(data[data.county.eq('DEMO_B')].attendance)=={8}
    assert data.all_three_available.all()
    with ZipFile(BytesIO(derived)) as z:
        assert json.loads(z.read('run_metadata.json'))['imputed_values']==1


@pytest.mark.parametrize('change', ['ai_reviewer','wrong_month','wrong_quote','stale_run','fractional_count'])
def test_invalid_approvals_are_rejected(change):
    archive,result,reviews=fixture();memory=result[3];reviews=approved(reviews,memory)
    if change=='ai_reviewer':reviews.loc[0,'reviewer_type']='AI'
    elif change=='stale_run':reviews.loc[0,'run_id']='different'
    else:
        counts=json.loads(reviews.loc[0,'meeting_counts_json'])
        if change=='wrong_month':counts[0]['date']='2024-02-15'
        if change=='wrong_quote':counts[0]['quote']='This is invented.'
        if change=='fractional_count':counts[0]['count']=12.5
        reviews.loc[0,'meeting_counts_json']=json.dumps(counts)
    with pytest.raises(AuditError):apply_attendance_reviews(archive,memory,reviews)


def test_memory_tampering_is_rejected():
    archive,result,reviews=fixture();memory=deepcopy(result[3]);memory['documents'][0]['text']='Changed'
    with pytest.raises(AuditError,match='memory changed'):
        apply_attendance_reviews(archive,memory,reviews)


def test_source_date_limits_retrieval():
    archive,result,reviews=fixture();docs=result[3]['documents'];assert docs[0]['month']=='2024-01'
    raw=BytesIO();pdf=Canvas(raw);pdf.drawString(50,750,'Attendance was 99 people.');pdf.save()
    other=build_evidence_review(archive,[('DEMO_A_2024-02-15.pdf',raw.getvalue())])
    assert other[0]['cases_with_same_county_month_pdf']==0
    assert not other[2][0]['citations']


def test_private_evidence_page_renders_review_controls_without_duplicate_keys():
    from streamlit.testing.v1 import AppTest
    archive,result,reviews=fixture()
    raw=BytesIO();pdf=Canvas(raw);pdf.drawString(50,750,'Attendance: 12 people present.');pdf.save()
    script=f'''
from unittest.mock import patch
from orcca_audit.h3_attendance_evidence_ui import render_evidence
class Uploaded:
    name='DEMO_A_2024-01-15.pdf'
    def getvalue(self): return bytes.fromhex('{raw.getvalue().hex()}')
def uploader(label,**kwargs):
    return [Uploaded()] if label=='Meeting PDFs for attendance verification' else None
with patch('streamlit.file_uploader',side_effect=uploader):
    render_evidence(bytes.fromhex('{archive.hex()}'))
'''
    at=AppTest.from_string(script,default_timeout=90).run()
    at.button(key='h3_attendance_retrieve').click().run()
    assert not at.exception
    assert at.button(key='h3_attendance_compare').disabled
    assert any(d.key=='h3_attendance_review_download' for d in at.get('download_button'))


def test_comparison_does_not_reuse_original_values_for_reviewed_fit(monkeypatch):
    import orcca_audit.h3_attendance_evidence_ui as ui
    from orcca_audit.h3_attendance_evidence import load_source
    archive,result,reviews=fixture();memory=result[3];seen=[]
    def analyse(raw):
        frame,meta=load_source(raw);n=int(frame.attendance.notna().sum());seen.append(n)
        path=pd.DataFrame([dict(menu='OEND',mediator='active_snapshot',term='path_product',county_months=n)])
        return meta,path,pd.DataFrame(),b'synthetic test analysis'
    monkeypatch.setattr(ui,'analyse_reconstruction',analyse)
    table,report=ui.compare_approved(archive,memory,approved(reviews,memory))
    assert seen==[2,4]
    assert table.county_months_original.iloc[0]==2
    assert table.county_months_reviewed.iloc[0]==4
    with ZipFile(BytesIO(report)) as z:
        assert len(json.loads(z.read('cell_changes.json')))==1


def test_run_binding_uses_source_content_not_archive_container():
    from orcca_audit.h3_attendance_evidence import source_run_id
    archive,result,reviews=fixture()
    stream=BytesIO()
    with ZipFile(BytesIO(archive)) as old, ZipFile(stream,'w',ZIP_DEFLATED) as new:
        for name in old.namelist():new.writestr(name,old.read(name))
        new.writestr('unrelated_output.txt','Not source evidence')
    assert archive!=stream.getvalue()
    assert source_run_id(archive)==source_run_id(stream.getvalue())
    data,ledger,derived=apply_attendance_reviews(stream.getvalue(),result[3],approved(reviews,result[3]))
    assert len(ledger)==1
