"""Recording integrity and public gates, not scientific performance tests."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest
from streamlit.testing.v1 import AppTest
from orcca_audit import model_review as demo
from orcca_audit.engine import AuditError


def test_real_recording_and_current_bm25_inputs_match():
    r = demo.replay()
    assert len(r['runs']) == r['actual_bm25_calls_this_session'] == 3
    assert r['provenance']['runtime_version'] == '0.3.36'
    assert r['provenance']['external_model_calls'] == r['human_annotations'] == r['data_values_changed'] == 0
    assert r['provenance']['script_sha256'] == __import__('hashlib').sha256((Path(__file__).resolve().parents[1]/r['provenance']['executed_script_source']).read_bytes()).hexdigest()
    assert all(x['finish_reason'] == 'stop' and x['usage']['completion_tokens'] > 0 for x in r['runs'])
    for x in r['runs']:
        demo.validate_reply(x['output'], x['case_id'], x['payload']['evidence'])


def test_fabricated_quotes_and_wrong_cases_fail():
    r = demo.load_record()['runs'][0]
    a = deepcopy(r['output']); a['case_id'] = 'not-the-case'
    with pytest.raises(AuditError): demo.validate_reply(a, r['case_id'], r['payload']['evidence'])
    a = deepcopy(r['output']); a['evidence'] = [{'chunk_id': r['payload']['evidence'][0]['chunk_id'], 'quote': 'invented evidence'}]
    with pytest.raises(AuditError): demo.validate_reply(a, r['case_id'], r['payload']['evidence'])


def test_insufficient_evidence_with_date_stays_visible():
    r = demo.load_record()['runs'][0]
    a = deepcopy(r['output']); a['status'] = 'insufficient_evidence'; a['proposed_start_month'] = '2025-04'
    assert demo.validate_reply(a, r['case_id'], r['payload']['evidence'])


def test_changed_fixture_cannot_reuse_recording(monkeypatch):
    original = demo.fixture()
    original['documents'][0]['text'] = 'Changed input'
    monkeypatch.setattr(demo, 'fixture', lambda: original)
    with pytest.raises(AuditError, match='changed|current synthetic inputs'): demo.load_record()


def test_private_metadata_never_enters_demo_export():
    record = demo.replay(); record['private_notes'] = 'PRIVATE_SENTINEL'
    with zipfile.ZipFile(BytesIO(demo.export_record(record))) as z:
        assert z.testzip() is None and 'START_HERE.html' in z.namelist()
        assert all(b'PRIVATE_SENTINEL' not in z.read(n) for n in z.namelist())
    record['runs'][0]['output']['answer'] = 'PRIVATE_SENTINEL'
    with pytest.raises(AuditError): demo.export_record(record)


def test_public_browser_flow_and_changed_project(monkeypatch):
    monkeypatch.setenv('ORCCA_DEPLOYMENT_MODE', 'public_demo')
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root/'app.py'), default_timeout=90).run()
    at.switch_page('pages/04_ORCCA_Workflow.py').run()
    at.button(key='study_sample_build').click().run()
    at.button(key='model_demo_replay').click().run()
    assert not at.exception and not at.get('file_uploader')
    assert at.session_state['model_demo_result'][1]['actual_bm25_calls_this_session'] == 3
    at.selectbox(key='model_demo_case').set_value('DEMO_UNKNOWN').run()
    assert not at.exception
    at.selectbox(key='study_sample').set_value('Changed input').run()
    assert 'workflow_project' not in at.session_state
    assert not at.get('download_button')
