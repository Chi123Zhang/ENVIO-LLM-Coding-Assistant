"""Synthetic H3 preparation tests, not research hypothesis tests."""
from io import BytesIO
import hashlib,json,zipfile
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from orcca_audit.study import sample, search, parse_inputs
from orcca_audit.workflow import start_project,build_findings,decide_finding,export_project,apply_completion
from orcca_audit.completion import identify_issues,prepare_evidence,evidence_candidate
from orcca_audit.engine import AuditError
from orcca_audit.workflow_ui import InputFile

ROOT=Path(__file__).resolve().parents[1]


def test_h3_requires_no_panel_and_preserves_gaps_through_completion_export():
    state={};p=start_project(state,*sample())
    assert 'panel' not in p['data'] and p['results']['structure']['counties']==3
    assert p['results']['h3_readiness'][0]['availability']=='Not supplied'
    data,issues=identify_issues(p['data']['tracker']);issue=next(x for x in issues if x['eligible'])
    candidate=evidence_candidate(issue,prepare_evidence(p['completion_evidence']),p['run_id'])['candidates'][0]
    decision={'candidate_id':candidate['candidate_id'],'run_id':p['run_id'],'decision':'approve','reviewer':'Synthetic reviewer','reason':'Explicit synthetic source'}
    q=apply_completion(state,p,data,issues,[candidate],[decision])
    assert q['results']['completion_overview']['unknown_fields']==0
    assert p['results']['completion_overview']['unknown_fields']==1
    assert 'original_tracker' in q['data'] and q['parent_run_id']==p['run_id']
    b=build_findings(q);assert len(b['hypotheses'])==1 and b['hypotheses'][0]['hypothesis']=='H3'
    assert all('causal' not in f.get('level','') for f in b['findings'])
    decide_finding(q,'S1','approve','Synthetic reviewer','Check current version only')
    with zipfile.ZipFile(BytesIO(export_project(q,True))) as z:
        report=json.loads(z.read('audit_report.json'))
        assert hashlib.sha256(z.read('synthetic_study_tracker.csv')).hexdigest()==q['manifest']['synthetic_tracker_file']['sha256']
        assert z.read('orcca_audit/H3_DEMO_DICTIONARY.csv')
        parsed=pd.read_csv(BytesIO(z.read('synthetic_completion_evidence.csv')))
        assert parsed.iloc[0]['text']==q['completion_evidence'].iloc[0]['text']
        assert report['results']['research_findings'][0]['finding_id']=='S1'
        assert report['results']['h3_readiness'][0]['availability']=='Not supplied'
        for d in b['documents']:
            if d['evidence_kind']=='deterministic_receipt':assert z.read(d['source_file']).decode()==d['text']


def test_exact_field_routing_and_bm25_are_distinct():
    _,docs,_,_=sample()
    text='Version: V1\nVariable: hcs_orc209l_1\nHas implementation started?'
    doc={**docs[0],'chunk_id':'exact','text':text,'sha256':hashlib.sha256(text.encode()).hexdigest(),'field':'hcs_orc209l_1','version':'V1'}
    assert search('hcs_orc209l_1',[doc],version='V1')['status']=='exact_field_definition'
    assert search('hcs_orc209l_1',[doc],version='V2')['status']=='field_definition_not_found'
    assert search('implementation started',[doc])['backend']=='bm25'
    assert search('hcs_orc209v_1',[doc])['citations']==[]


def test_minimal_workbook_starts_without_dictionary_and_blocks_duplicate_keys():
    import openpyxl
    book=openpyxl.Workbook();ws=book.active;ws.title='orcca_tracker_v2'
    ws.append(['hcs_orc2record_id','redcap_event_name','hcs_orc2version','hcs_orc209l_1'])
    ws.append([1,'january_2024_arm_1','V1',1])
    mapping=book.create_sheet('Sheet1');mapping.append(['record_id','county']);mapping.append([1,'DEMO_A'])
    out=BytesIO();book.save(out)
    data,docs,evidence,manifest=parse_inputs({'raw_tracker':InputFile('synthetic.xlsx',out.getvalue())},[])
    p=start_project({},data,docs,evidence,manifest)
    assert p['results']['structure']['records']==1 and 'tracker' not in data
    assert not p['results']['field_review'][0]['definition_found']
    ws.append([1,'january_2024_arm_1','V1',0]);out=BytesIO();book.save(out)
    with pytest.raises(AuditError,match='duplicate county-month'):parse_inputs({'raw_tracker':InputFile('synthetic.xlsx',out.getvalue())},[])


def test_default_h3_ui_navigation_reset_and_legacy_isolation(monkeypatch):
    monkeypatch.delenv('ORCCA_DEPLOYMENT_MODE',raising=False);monkeypatch.delenv('ORCCA_ALLOW_PRIVATE_UPLOADS',raising=False)
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
    at.switch_page('pages/04_ORCCA_Workflow.py').run()
    assert at.radio(key='workflow_workspace_widget').value=='H3 study preparation' and not at.exception
    assert not at.get('file_uploader')
    at.button(key='study_sample_build').click().run();assert not at.exception
    run=at.session_state['workflow_project']['run_id']
    at.text_input(key='study_query').input('coalition').run()
    at.button(key='study_search').click().run();assert not at.exception
    assert at.session_state['study_search_result'][1]['backend']=='bm25'
    at.button(key='study_reset_query').click().run()
    assert not at.exception and at.text_input(key='study_query').value==''
    at.button(key='study_draft').click().run();assert not at.exception
    next(x for x in at.text_input if x.label=='Reviewer name').input('Synthetic reviewer')
    next(x for x in at.selectbox if x.label=='Review decision').set_value('Accept')
    next(x for x in at.text_area if x.label=='Review reason (English)').input('Synthetic acceptance only')
    next(x for x in at.button if x.label=='Save finding review').click().run()
    assert not at.exception and any('S1 · Accepted' in str(x.value) for x in at.caption)
    at.switch_page('pages/01_ORCCA_Research_Audit.py').run();assert not at.exception and at.metric[0].value=='3'
    at.switch_page('pages/02_ORCCA_Research_Findings.py').run();assert not at.exception
    assert any('H3 remains a research question' in str(x.value) for x in at.subheader)
    at.switch_page('pages/03_ORCCA_Evidence_Completion.py').run();assert not at.exception
    assert at.session_state['completion_run']==run
    at.switch_page('pages/04_ORCCA_Workflow.py').run();assert not at.exception
    at.radio(key='workflow_workspace_widget').set_value('Legacy data audit').run()
    assert not at.exception and 'workflow_project' not in at.session_state
    at.button(key='workflow_build').click().run();assert not at.exception
    assert 'consistency' in at.session_state['workflow_project']['results']
