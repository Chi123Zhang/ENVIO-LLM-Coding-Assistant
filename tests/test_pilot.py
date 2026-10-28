"""Synthetic invariants, not study performance labels."""
from io import BytesIO
import json
import zipfile

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path

from orcca_audit.engine import AuditError
from orcca_audit.pilot import branching, run, exports, inventory
from orcca_audit.study import sample
from orcca_audit.workflow import start_project


def project():return start_project({},*sample())


def test_branching_three_valued_and_fail_closed():
    assert branching("[a]='1' or [b] = '1'",{'a':1,'b':None})=='applicable'
    assert branching("[a]='1' and [b]='1'",{'a':1,'b':None})=='unresolved_branching'
    assert branching("[a]='1' and [b]='1'",{'a':0,'b':None})=='structural_skip'
    assert branching("[a]='1' or [b]='1' and [c]='1'",{})=='unresolved_branching'
    assert branching('__import__("os")',{})=='unresolved_branching'


def test_no_reference_or_label_enters_memory_and_originals_unchanged():
    p=project();original=p['data']['raw_tracker'].copy(deep=True)
    # Deliberately supplied answer/model-like passages are not indexed.
    p['documents'].append({'field':'hcs_orc209l_1','evidence_kind':'model_output','text':'secret'})
    r=run(p)
    pd.testing.assert_frame_equal(original,p['data']['raw_tracker'])
    assert not set(r['summary']['development_counties']) & set(r['summary']['heldout_counties'])
    assert r['summary']['actual_bm25_calls']==24
    assert all(x['proposed_value'] is None for x in r['predictions'])
    assert all(x['accuracy_among_proposed'] is None for x in r['metrics'])
    assert 'secret' not in json.dumps(r['memory'])
    # Changing observed target values cannot change memory content.
    p['data']['raw_tracker']['hcs_orc209l_1']=0
    assert run(p)['memory']['memory_hash']==r['memory']['memory_hash']
    with zipfile.ZipFile(BytesIO(exports(r))) as z:
        a=pd.read_csv(BytesIO(z.read('A_annotator_1.csv')))
        assert a.human_answer.isna().all() and 'recorded_value' not in a and 'eligibility' not in a
        assert z.read('A_annotator_1.csv')==z.read('A_annotator_2.csv')
        assert 'sealed_recorded_references.csv' in z.namelist()


def test_input_bound_memory_derived_rejection_and_definition_gate():
    p=project();r=run(p);p['parent_run_id']='old'
    with pytest.raises(AuditError,match='original input'):run(p)
    p=project();p['data']['dictionary']=None
    with pytest.raises(AuditError,match='dictionary'):run(p)
    p=project();p['data']['dictionary'].loc[:,'Choices, Calculations, OR Slider Labels']='1, Other'
    assert all(x['eligibility']=='coding_requires_review' for x in inventory(p))


def test_public_pilot_app_flow(monkeypatch):
    monkeypatch.delenv('ORCCA_DEPLOYMENT_MODE',raising=False)
    root=Path(__file__).resolve().parents[1]
    at=AppTest.from_file(str(root/'app.py'),default_timeout=90).run()
    at.switch_page('pages/04_ORCCA_Workflow.py').run()
    at.button(key='study_sample_build').click().run()
    at.button(key='pilot_run_button').click().run()
    assert not at.exception and at.session_state['pilot_result']['summary']['masking_cases']==24
    assert not at.get('file_uploader')
    at.selectbox(key='study_sample').set_value('Changed input').run()
    assert not at.exception and 'workflow_project' not in at.session_state
