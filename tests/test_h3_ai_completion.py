"""Synthetic software checks. Fake generators are not research model runs."""
from copy import deepcopy
from io import BytesIO
import json

import pytest
from reportlab.pdfgen.canvas import Canvas

from orcca_audit.engine import AuditError
from orcca_audit.h3_sources import demo_sources, rebuild
from orcca_audit.h3_attendance_evidence import build_evidence_review, load_source
from orcca_audit.h3_ai_completion import (
    prepare_experiment, parse_candidate, run_completion, experimental_archive, compare_experiment,
)


def fixture(date='2024-02-10'):
    files, attendance = demo_sources()
    original = rebuild(files, attendance, '2024-03', '2024-05', synthetic=True)[2]
    raw = BytesIO()
    pdf = Canvas(raw)
    pdf.drawString(40, 740, 'The coalition meeting attendance was 12 people present.')
    pdf.save()
    memory = build_evidence_review(original, [(f'Demo 2_{date}.pdf', raw.getvalue())], synthetic=True)[3]
    return original, memory, prepare_experiment(original, memory)


def reply(task, count=12):
    return dict(status='candidate', count=count, chunk_id=task['evidence'][0]['chunk_id'],
                quote='attendance was 12 people present.', reason='Explicit synthetic headcount.')


def test_unreviewed_lane_fills_only_missing_derived_exposure_and_never_human_labels():
    archive, memory, prepared = fixture()
    original, _ = load_source(archive)
    result = run_completion(prepared, lambda task:dict(raw_response=json.dumps(reply(task))),
                            dict(runtime='fake_generator_for_software_test_only'))
    data, ledger, derived = experimental_archive(prepared, result)
    assert len(ledger) == 1 and result['summary']['human_confirmed_values'] == 0
    assert data[data.county.eq('Demo 2') & data.attendance_month.eq('2024-02')].attendance.eq(12).all()
    known = original.attendance.notna()
    assert data.loc[known,'attendance'].equals(original.loc[known,'attendance'])
    _, metadata = load_source(derived)
    assert metadata['approved_attendance_county_months'] == 0
    assert metadata['ai_unreviewed_attendance_county_months'] == 1
    assert metadata['eligible_for_confirmatory_analysis'] is False
    assert all(r['human_confirmed'] is False for r in result['candidates'])


def test_no_matching_date_never_calls_model():
    archive, memory, prepared = fixture('2024-03-10')
    def forbidden(task):
        raise AssertionError('No evidence should mean no model call.')
    result = run_completion(prepared, forbidden, dict(runtime='software_test'))
    assert result['summary']['llm_calls'] == result['summary']['candidate_county_months'] == 0
    assert not experimental_archive(prepared,result)[1]


@pytest.mark.parametrize('change', ['wrong_quote','wrong_count','wrong_id','negative','boolean','approximate'])
def test_invalid_model_numbers_are_not_experimental_measurements(change):
    _, _, prepared = fixture()
    task = deepcopy(prepared['tasks'][0]);value = reply(task)
    if change == 'wrong_quote': value['quote'] = 'Invented original quotation.'
    if change == 'wrong_count': value['count'] = 81
    if change == 'wrong_id': value['chunk_id'] = 'outside_retrieval'
    if change == 'negative': value['count'] = -1
    if change == 'boolean': value['count'] = True
    if change == 'approximate':
        value['quote'] = 'about 12 people present.'
        task['evidence'][0]['text'] = value['quote']
    with pytest.raises(AuditError):parse_candidate(json.dumps(value),task)


def test_unknown_and_format_failure_preserve_gaps():
    _, _, prepared = fixture()
    for raw in [json.dumps(dict(status='unknown',count=None,chunk_id='',quote='',reason='No exact count.')),
                '{"status":']:
        result=run_completion(prepared,lambda task:dict(raw_response=raw),dict(runtime='software_test'))
        assert result['summary']['candidate_county_months']==0
        assert not experimental_archive(prepared,result)[1]


def test_stale_memory_and_changed_results_are_rejected():
    archive,memory,prepared=fixture()
    changed=deepcopy(memory);changed['run_id']='different'
    with pytest.raises(AuditError,match='current source'):prepare_experiment(archive,changed)
    changed=deepcopy(memory);changed['documents'][0]['text']='Changed source'
    with pytest.raises(AuditError,match='memory changed'):prepare_experiment(archive,changed)
    result=run_completion(prepared,lambda task:dict(raw_response=json.dumps(reply(task))),dict(runtime='software_test'))
    result['candidates'][0]['ai_value']=999
    with pytest.raises(AuditError,match='frozen experiment'):experimental_archive(prepared,result)


def test_experimental_report_keeps_unreviewed_label_and_unique_zip_members():
    from zipfile import ZipFile
    _,_,prepared=fixture()
    result=run_completion(prepared,lambda task:dict(raw_response=json.dumps(reply(task))),dict(runtime='software_test'))
    summary,comparison,raw=compare_experiment(prepared,result)
    assert summary['applied_to_experimental_county_months']==1
    with ZipFile(BytesIO(raw)) as outer:
        with ZipFile(BytesIO(outer.read('AI_UNREVIEWED_analysis.zip'))) as inner:
            assert len(inner.namelist())==len(set(inner.namelist()))
            page=inner.read('START_HERE.html').decode()
            assert 'AI-unreviewed experimental attendance additions: 1' in page
            assert 'Evidence-reviewed additions: 1' not in page


def test_browser_experiment_requires_private_route_and_explicit_opt_in(monkeypatch):
    from streamlit.testing.v1 import AppTest
    archive,memory,prepared=fixture()
    script=f'''
import streamlit as st
from orcca_audit.h3_ai_completion_ui import render_ai_experiment
st.session_state['h3_attendance_evidence']=('fixture',(None,None,None,{memory!r},None))
render_ai_experiment(bytes.fromhex('{archive.hex()}'))
'''
    monkeypatch.delenv('ORCCA_DEPLOYMENT_MODE',raising=False)
    monkeypatch.delenv('ORCCA_ALLOW_PRIVATE_UPLOADS',raising=False)
    at=AppTest.from_string(script,default_timeout=60).run()
    assert not at.exception and not at.button
    monkeypatch.setenv('ORCCA_DEPLOYMENT_MODE','local_private')
    monkeypatch.setenv('ORCCA_ALLOW_PRIVATE_UPLOADS','1')
    at.run()
    assert not at.exception and at.button(key='h3_ai_run').disabled
    at.checkbox(key='h3_ai_opt_in').check().run()
    assert not at.exception and not at.button(key='h3_ai_run').disabled
