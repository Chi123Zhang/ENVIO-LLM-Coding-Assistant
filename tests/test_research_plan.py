"""Question/source lineage and safe planning, not causal evaluation."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest
from streamlit.testing.v1 import AppTest
from orcca_audit.research_plan import DEFAULT, normalize, save, requirements, fingerprint
from orcca_audit.study import sample
from orcca_audit.workflow import start_project, build_findings, decide_finding, export_project
from orcca_audit.engine import AuditError


def test_question_change_invalidates_reviews_and_preserves_data():
    state={}; p=start_project(state,*sample()); build_findings(p)
    decide_finding(p,'S1','approve','Synthetic reviewer','Inspect this version')
    raw=p['data']['raw_tracker'].copy(deep=True)
    changed={**DEFAULT,'question':'Is an implementation start month documented?', 'hypothesis':'Timing', 'mode':'Implementation record review'}
    q=save(state,changed)
    assert q['run_id']!=p['run_id'] and q['parent_run_id']==p['run_id']
    assert not q['finding_decisions'] and q['bundle'] is None
    assert q['data']['raw_tracker'].equals(raw)
    assert not any('Is an implementation start month' in d['text'] for d in q['documents'])
    assert state['study_query']==changed['question']


def test_presence_does_not_establish_a_measure_or_effect():
    state={}; p=start_project(state,*sample())
    rows=p['results']['question_requirements']
    assert next(x for x in rows if '(X)' in x['component'])['status']=='Not supplied'
    assert next(x for x in rows if '(Y)' in x['component'])['status']=='Not supplied'
    assert next(x for x in rows if x['component']=='Design and temporal order')['status']=='Not established'
    assert not any('validated'==x['status'] for x in rows)


def test_invalid_plan_does_not_replace_current_question():
    state={}; p=start_project(state,*sample())
    with pytest.raises(AuditError):save(state,{**DEFAULT,'question':''})
    assert state['workflow_project'] is p
    with pytest.raises(AuditError):normalize({**DEFAULT,'start_month':'2025-13'})
    with pytest.raises(AuditError):normalize({**DEFAULT,'start_month':'2025-06','end_month':'2025-01'})


def test_saved_question_and_requirements_in_readable_export():
    p=start_project({},*sample()); build_findings(p)
    with zipfile.ZipFile(BytesIO(export_project(p))) as z:
        report=json.loads(z.read('audit_report.json'))
        assert report['results']['research_plan']['question']==DEFAULT['question']
        assert 'question_requirements.csv' in z.namelist()
        assert 'Saved research question' in z.read('START_HERE.html').decode()
        assert all(f['review_status']=='pending' for f in report['results']['research_findings'])


def test_question_form_and_cross_page_current_version(monkeypatch):
    monkeypatch.setenv('ORCCA_DEPLOYMENT_MODE','public_demo')
    root=Path(__file__).resolve().parents[1]
    at=AppTest.from_file(str(root/'app.py'),default_timeout=90).run()
    at.switch_page('pages/04_ORCCA_Workflow.py').run()
    at.button(key='study_sample_build').click().run()
    old=at.session_state['workflow_project']['run_id']
    at.text_area(key='question_text').input('Which fictional implementation record requires review?')
    at.button(key='question_save').click().run()
    assert not at.exception
    new=at.session_state['workflow_project']['run_id']
    assert old!=new
    at.button(key='study_draft').click().run()
    at.switch_page('pages/02_ORCCA_Research_Findings.py').run()
    assert not at.exception and at.session_state['workflow_project']['run_id']==new
    assert any('Which fictional implementation' in x.value for x in at.markdown)
    assert not at.get('file_uploader')
