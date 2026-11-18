from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from streamlit.testing.v1 import AppTest

from orcca_audit.h3_sources import demo_sources, rebuild
from orcca_audit.h3_source_analysis import analyse_reconstruction


def test_incomplete_sources_export_diagnostics_without_invented_p_values():
    _, _, source = rebuild(*demo_sources(),'2024-03','2024-05',synthetic=True)
    info, paths, diagnostics, report = analyse_reconstruction(source)
    assert info['county_months'] == 6 and info['imputed_values'] == 0
    assert info['specifications_attempted'] == 6 and info['models_fitted'] == 0
    assert paths.empty and diagnostics.point_fit.eq('unavailable').all()
    with ZipFile(BytesIO(report)) as archive:
        assert archive.testzip() is None
        assert 'model_results.csv' in archive.namelist()
        assert 'source_reconstruction.html' in archive.namelist()
        assert json.loads(archive.read('run_metadata.json'))['models_fitted'] == 0


def test_statistical_button_runs_and_clear_removes_download():
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root/'app.py'),default_timeout=90).run().switch_page('pages/04_ORCCA_Workflow.py').run()
    at.button(key='h3_sources_run').click().run()
    at.button(key='h3_source_analysis_run').click().run()
    assert not at.exception
    assert any(x.key=='h3_source_analysis_download' for x in at.get('download_button'))
    at.button(key='h3_sources_clear').click().run()
    assert not at.exception
    assert not any(x.key=='h3_source_analysis_download' for x in at.get('download_button'))
