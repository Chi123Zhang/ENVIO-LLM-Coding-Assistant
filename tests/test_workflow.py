"""Synthetic end-to-end workflow checks, not research performance labels."""
from copy import deepcopy
from io import BytesIO
import json
import zipfile

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path

from orcca_audit.completion import evidence_candidate, identify_issues, prepare_evidence
from orcca_audit.engine import AuditError
from orcca_audit.workflow import (apply_completion, build_findings, current_project,
    decide_finding, export_project, findings_for_export, invalidate, start_project)
from orcca_audit.workflow_ui import synthetic_project_inputs

ROOT=Path(__file__).resolve().parents[1]


def project(state=None, variant='Baseline'):
    state={} if state is None else state
    return state,start_project(state,*synthetic_project_inputs(variant))


def test_versions_invalidate_decisions_and_never_substitute_saved_findings():
    state,p=project();bundle=build_findings(p)
    assert len(bundle['findings'])==4 and '18 county-month rows' in bundle['findings'][0]['statement']
    decide_finding(p,'C1','approve','Synthetic reviewer','Software acceptance only')
    assert len(findings_for_export(p,True)['findings'])==1
    state['findings_result']={'old':True};state['completion_candidates']={'old':True}
    invalidate(state,'Required file removed')
    assert current_project(state,False) is None
    assert 'findings_result' not in state and 'completion_candidates' not in state
    _,q=project(state,'Changed panel');build_findings(q)
    assert q['run_id']!=p['run_id'] and q['revision']==2
    assert '17 rows agree' in q['bundle']['findings'][0]['statement']
    with pytest.raises(AuditError,match='No findings'):findings_for_export(q,True)


def test_evidence_completion_changes_only_derived_tracker_and_reruns_audit():
    state,p=project();build_findings(p)
    original=p['data']['tracker'].copy(deep=True)
    data,issues=identify_issues(original)
    issue=next(x for x in issues if x['eligible'] and x['field']=='discontinued')
    result=evidence_candidate(issue,prepare_evidence(p['completion_evidence']),p['run_id'])
    assert result['status']=='source_candidate_unreviewed'
    candidate=result['candidates'][0]
    decision={'candidate_id':candidate['candidate_id'],'run_id':p['run_id'],
        'decision':'approve','reviewer':'Synthetic reviewer','reason':'Exact synthetic assertion; software test only'}
    q=apply_completion(state,p,data,issues,[candidate],[decision])
    assert q['parent_run_id']==p['run_id'] and q['bundle'] is None and not q['finding_decisions']
    assert p['results']['exposures']['summary']['uncertain_or_mismatched']==1
    assert q['results']['exposures']['summary']['uncertain_or_mismatched']==0
    pd.testing.assert_frame_equal(original,p['data']['tracker'])
    pd.testing.assert_frame_equal(original,q['data']['original_tracker'])
    build_findings(q);decide_finding(q,'C1','approve','Synthetic reviewer','Current version checked')
    with zipfile.ZipFile(BytesIO(export_project(q,True))) as z:
        exported=json.loads(z.read('audit_report.json'))
        assert exported['provenance']['run_id']==q['run_id']
        assert len(exported['results']['research_findings'])==1
        assert exported['results']['completion_review_ledger'][0]['candidate_id']==candidate['candidate_id']
        assert 'current_original_tracker.csv' in z.namelist()
        assert 'START_HERE.html' in z.namelist()
        for doc in exported['results']['evidence_excerpts']:
            if doc['evidence_kind']=='deterministic_receipt':
                assert z.read(doc['source_file']).decode()==doc['text']
        assert exported['provenance']['date_timezone']=='Asia/Shanghai'
    decision['run_id']='stale'
    with pytest.raises(AuditError,match='Stale'):apply_completion({},p,data,issues,[candidate],[decision])


def test_private_projects_do_not_cross_public_policy():
    state,p=project();p['classification']='private_research'
    assert current_project(state,False) is None and state['workflow_active']


def test_csv_string_tracker_can_accept_a_numeric_source_candidate():
    from orcca_audit.io import read_table
    state,p=project()
    original=read_table(p['data']['tracker'].to_csv(index=False).encode(),'synthetic_tracker.csv')
    p['data']['tracker']=original
    data,issues=identify_issues(original)
    issue=next(x for x in issues if x['eligible'])
    candidate=evidence_candidate(issue,prepare_evidence(p['completion_evidence']),p['run_id'])['candidates'][0]
    decision={'candidate_id':candidate['candidate_id'],'run_id':p['run_id'],'decision':'approve',
        'reviewer':'Synthetic test','reason':'Explicit synthetic assertion'}
    derived=apply_completion(state,p,data,issues,[candidate],[decision])
    assert derived['results']['exposures']['summary']['uncertain_or_mismatched']==0
    row=derived['data']['tracker'].iloc[1]
    assert row['discontinued']==0 and row['discontinued__original_value']==''
    assert original.iloc[1]['discontinued']==''


def test_browser_project_state_survives_page_navigation(monkeypatch):
    monkeypatch.delenv('ORCCA_DEPLOYMENT_MODE',raising=False)
    monkeypatch.delenv('ORCCA_ALLOW_PRIVATE_UPLOADS',raising=False)
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
    at.session_state['workflow_workspace']='Legacy data audit'
    at.switch_page('pages/04_ORCCA_Workflow.py').run()
    assert not at.exception and not at.get('file_uploader')
    at.button(key='workflow_build').click().run()
    assert not at.exception and 'workflow_project' in at.session_state
    at.button(key='workflow_draft').click().run()
    assert not at.exception
    run=at.session_state['workflow_project']['run_id']
    at.switch_page('pages/02_ORCCA_Research_Findings.py').run()
    assert not at.exception
    assert at.radio(key='workflow_findings_source').value=='Current web project'
    assert any('18 county-month' in str(x.value) for x in at.markdown)
    at.switch_page('pages/01_ORCCA_Research_Audit.py').run()
    assert not at.exception and at.metric[0].value=='18'
    at.switch_page('pages/03_ORCCA_Evidence_Completion.py').run()
    assert not at.exception and at.radio(key='completion_mode').value=='Current web project'
    assert at.session_state['completion_run']==run
    at.session_state['workflow_workspace']='Legacy data audit'
    at.switch_page('pages/04_ORCCA_Workflow.py').run()
    assert not at.exception and at.session_state['workflow_project']['run_id']==run
    at.selectbox(key='workflow_sample').select('Changed panel').run()
    assert not at.exception and 'workflow_project' not in at.session_state
    at.switch_page('pages/02_ORCCA_Research_Findings.py').run()
    assert not at.exception and not at.tabs
    assert any('No current draft' in x.value for x in at.warning)
    at.switch_page('pages/01_ORCCA_Research_Audit.py').run()
    assert not at.exception and not at.tabs
    assert any('No current project audit' in x.value for x in at.warning)
    at.switch_page('pages/03_ORCCA_Evidence_Completion.py').run()
    assert not at.exception and not at.tabs
    assert any('No current project inputs' in x.value for x in at.warning)


def test_passage_scope_requires_exact_source_and_preserves_applicability_review():
    from orcca_audit.workflow import add_scoped_evidence
    state,p=project();_,issues=identify_issues(p['data']['tracker']);issue=next(x for x in issues if x['eligible'])
    doc=deepcopy(p['documents'][0]);doc['text']='SYNTHETIC explicit record\ndiscontinued=0'
    import hashlib
    doc['sha256']=hashlib.sha256(doc['text'].encode()).hexdigest();p['documents']=[doc]
    with pytest.raises(AuditError,match='exact passage'):
        add_scoped_evidence(state,p,issue['issue_id'],doc['chunk_id'],'invented assignment','Tester','Synthetic test')
    q=add_scoped_evidence(state,p,issue['issue_id'],doc['chunk_id'],'discontinued=0','Tester','Explicit synthetic event applicability')
    assert q['run_id']!=p['run_id'] and q['parent_run_id']==p['run_id']
    row=q['completion_evidence'].iloc[-1]
    assert row['county']==issue['county'] and row['month']==issue['month']
    assert 'applicability_reviewer' in row['source_ref']
    assert int(row['_source_row'])>=3


def test_docx_and_pptx_extraction_do_not_invent_pdf_pages():
    from orcca_audit.io import extract_document
    def archive(files):
        buf=BytesIO()
        with zipfile.ZipFile(buf,'w') as z:
            for name,text in files.items():z.writestr(name,text)
        return buf.getvalue()
    doc=extract_document(archive({'word/document.xml':'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:r><w:t>SYNTHETIC paragraph</w:t></w:r></w:p></w:document>'}),'synthetic.docx')
    assert doc[0]['page'] is None and 'no rendered page' in doc[0]['locator']
    slide='<a:root xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:p><a:r><a:t>SYNTHETIC slide</a:t></a:r></a:p></a:root>'
    docs=extract_document(archive({'ppt/slides/slide1.xml':slide,'ppt/slides/slide2.xml':slide}),'synthetic.pptx')
    assert len({d['chunk_id'] for d in docs})==2
    assert [d['slide'] for d in docs]==[1,2] and all(d['page'] is None for d in docs)


def test_session_input_buffers_allow_single_file_replacement_after_navigation():
    from orcca_audit.workflow_ui import InputFile,retained_inputs,update_input,parse_uploads
    data,_,_,_=synthetic_project_inputs()
    state={}
    for role in ('panel','monthly','annual','tracker'):
        state['widget']=InputFile(role+'.csv',data[role].to_csv(index=False).encode())
        update_input(state,role,'widget')
    del state['widget']  # Streamlit cleans page-owned upload widgets on navigation.
    supplied,docs=retained_inputs(state)
    parsed,_,_,manifest=parse_uploads(supplied,docs)
    assert len(parsed['panel'])==18 and len(manifest)==5
    data['panel'].loc[0,'population']=0
    state['replacement']=InputFile('replacement.csv',data['panel'].to_csv(index=False).encode())
    update_input(state,'panel','replacement')
    supplied,_=retained_inputs(state)
    assert supplied['panel'].name=='replacement.csv' and supplied['annual'].name=='annual.csv'
    state['removed']=None;update_input(state,'annual','removed')
    with pytest.raises(AuditError,match='required together'):parse_uploads(*retained_inputs(state))
