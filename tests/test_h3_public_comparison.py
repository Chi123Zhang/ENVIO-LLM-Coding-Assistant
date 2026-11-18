from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from streamlit.testing.v1 import AppTest

from orcca_audit.h3_public_comparison import full_demo_sources, demo_meeting_pdf, demo_input_archive
from orcca_audit.h3_sources import rebuild
from orcca_audit.h3_no_change_ui import compare_no_change
from orcca_audit.h3_attendance_evidence import build_evidence_review


def test_full_public_fixture_has_fictional_sources_and_real_computation():
    files,attendance=full_demo_sources()
    meta,coverage,archive=rebuild(files,attendance,'2024-01','2024-12',synthetic=True)
    assert meta['county_months']==72 and meta['counties']==6
    with ZipFile(BytesIO(archive)) as z:
        data=pd.read_csv(z.open('county_month.csv'))
    assert data.county.str.startswith('DEMO_').all()
    result=compare_no_change(archive,files['tracker'],files['dictionary'])
    assert result[0]['rule_reconstructed_mediator_values']>0
    assert result[0]['analysis_summary']['rule_derived']['models_fitted']==6
    assert result[0]['analysis_summary']['original']['inference_available']==6
    assert len(result[2])==6 and result[2].county_months.eq(65).all()
    evidence=build_evidence_review(archive,[demo_meeting_pdf()],synthetic=True)
    assert evidence[0]['classification']=='synthetic demonstration'
    assert evidence[0]['actual_bm25_queries']==1 and evidence[0]['applied_values']==0
    assert all(d['classification']=='synthetic' for d in evidence[3]['documents'])
    with ZipFile(BytesIO(demo_input_archive())) as z:
        assert z.testzip() is None and len(z.namelist())==7
        assert all(name.startswith(('fictional_','DEMO_','README')) for name in z.namelist())


def test_public_browser_flow_keeps_research_uploads_closed(monkeypatch):
    for key in ['ORCCA_DEPLOYMENT_MODE','ORCCA_ALLOW_PRIVATE_UPLOADS']:
        monkeypatch.delenv(key,raising=False)
    root=Path(__file__).resolve().parents[1]
    at=AppTest.from_file(str(root/'app.py'),default_timeout=90).run().switch_page('pages/04_ORCCA_Workflow.py').run()
    at.selectbox(key='h3_sources_demo_scenario').select('Full H3 comparison demo').run()
    at.button(key='h3_sources_run').click().run()
    at.button(key='h3_no_change_compare').click().run()
    at.button(key='h3_full_demo_retrieve').click().run()
    assert not at.exception and not at.get('file_uploader')
    keys={x.key for x in at.get('download_button')}
    assert {'h3_full_demo_inputs','h3_no_change_download','h3_full_demo_evidence_download'}<=keys
    assert not any('Generate' in x.label for x in at.button)
    at.button(key='h3_sources_clear').click().run()
    assert not at.exception
    assert not any(x.key in ['h3_no_change_download','h3_full_demo_evidence_download'] for x in at.get('download_button'))
