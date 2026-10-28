"""Synthetic bridge acceptance. Model output fixtures are not research labels."""
from copy import deepcopy
from io import BytesIO
import json
import zipfile

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path

from orcca_audit.envio_bridge import capture_run, transfer_segment
from orcca_audit.engine import AuditError
from orcca_audit.workflow import build_findings, decide_finding, export_project, start_project
from orcca_audit.workflow_ui import synthetic_project_inputs

ROOT = Path(__file__).resolve().parents[1]


def inputs(monkeypatch):
    monkeypatch.setenv('ORCCA_DEPLOYMENT_MODE','local_private')
    monkeypatch.setenv('ORCCA_ALLOW_PRIVATE_UPLOADS','1')
    state={}
    project=start_project(state,*synthetic_project_inputs(),private=True)
    quote='SYNTHETIC review only: transport made attendance difficult.'
    frame=pd.DataFrame([{'text':quote,'source_name':'synthetic_interview.txt','segment_index':1,
        'codes':'environmental_barrier','rationale':'SYNTHETIC MOCK RESPONSE, not an API call.'}])
    sources=[{'source_name':'synthetic_interview.txt','source_text':'SYNTHETIC transcript header\n'+quote}]
    capture_run(state,frame,sources,'fixture',{'model':'mock; no API calls','use_rag_context':False})
    return state,project,quote


def link(state,project,quote,**kwargs):
    return transfer_segment(state,project,state['envio_orcca_capture'],0,quote,
        'Synthetic reviewer','Software-only pilot for ORCCA relevance',True,**kwargs)


def test_handoff_new_version_then_review_and_export(monkeypatch):
    state,p,quote=inputs(monkeypatch)
    before=p['data']['tracker'].copy(deep=True)
    build_findings(p);decide_finding(p,'C1','approve','Synthetic reviewer','Software check')
    q=link(state,p,quote,county='DEMO_A',month='2024-01')
    assert q['revision']==2 and not q['finding_decisions'] and q['parent_run_id']==p['run_id']
    pd.testing.assert_frame_equal(before,q['data']['tracker'])
    b=build_findings(q)
    f=next(x for x in b['findings'] if x['finding_id']=='I1')
    assert f['level']=='method_review' and f['review_status']=='pending'
    d=next(d for d in b['documents'] if d['chunk_id']==f['evidence_ids'][0])
    assert d['text']==quote and d['line_start']==2 and d['page'] is None
    assert all('MOCK RESPONSE' not in d['text'] for d in b['documents'])
    decide_finding(q,'I1','approve','Synthetic reviewer','Approve as interpretation only')
    with zipfile.ZipFile(BytesIO(export_project(q,True))) as z:
        report=json.loads(z.read('audit_report.json'))
        ledger=report['results']['current_implementation_evidence'][0]
        assert ledger['scope_status']=='human_assigned_unverified'
        assert report['results']['research_findings'][0]['finding_id']=='I1'
        assert ledger['run_settings']['model']=='mock; no API calls'


def test_no_public_transfer_fabrication_ambiguous_source_or_stale_project(monkeypatch):
    state,p,quote=inputs(monkeypatch)
    with pytest.raises(AuditError,match='exactly'):link(state,p,'Invented source statement')
    with pytest.raises(AuditError,match='County'):link(state,p,quote,county='INVENTED')
    with pytest.raises(AuditError,match='month'):link(state,p,quote,month='2024-01')
    capture=state['envio_orcca_capture']
    with pytest.raises(AuditError,match='Confirm'):transfer_segment(state,p,capture,0,quote,'X','Reason',False)
    capture['sources']['synthetic_interview.txt']+=quote
    with pytest.raises(AuditError,match='more than once'):link(state,p,quote)
    capture['sources']['synthetic_interview.txt']='\n'+quote
    q=link(state,p,quote)
    with pytest.raises(AuditError,match='already linked'):link(state,q,quote)
    with pytest.raises(AuditError,match='changed'):link(state,p,quote)
    monkeypatch.setenv('ORCCA_DEPLOYMENT_MODE','public_demo')
    with pytest.raises(AuditError,match='private'):link(state,q,quote)
    capture_run(state,pd.DataFrame(),[],'fixture',{})
    assert 'envio_orcca_capture' not in state


def test_bridge_ui_cross_page_handoff_without_real_generation(monkeypatch):
    state,p,quote=inputs(monkeypatch)
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90)
    for k,v in state.items():at.session_state[k]=v
    at.run()
    assert not at.exception
    # The fixture is an earlier explicitly selected run, not current-input output.
    at.checkbox(key='envio_use_capture_'+state['envio_orcca_capture']['capture_id']).check().run()
    def element(items,label):return next(x for x in items if x.label==label)
    element(at.text_area,'Exact original quotation (must also occur in the source)').input(quote)
    element(at.checkbox,'I checked that this source concerns this ORCCA project, not an unrelated ENVIO participant').check()
    element(at.text_input,'Applicability reviewer').input('Synthetic reviewer')
    element(at.text_area,'Why this source and interpretation apply to ORCCA (English)').input('Synthetic software validation only')
    element(at.button,'Link candidate and create ORCCA version').click().run()
    assert not at.exception and at.session_state['workflow_project']['revision']==2
    at.session_state['workflow_workspace']='Legacy data audit'
    at.switch_page('pages/04_ORCCA_Workflow.py').run()
    assert not at.exception
    at.button(key='workflow_draft').click().run()
    at.switch_page('pages/02_ORCCA_Research_Findings.py').run()
    assert not at.exception
    assert any('environmental_barrier' in str(x.value) for x in at.markdown)
