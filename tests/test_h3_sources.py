from io import BytesIO
import json
from zipfile import ZipFile

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from orcca_audit.engine import AuditError
from orcca_audit.h3_sources import demo_sources, rebuild


def unpack(archive, filename):
    with ZipFile(BytesIO(archive)) as z:
        assert z.testzip() is None
        return pd.read_csv(z.open(filename))


def test_raw_sources_and_unknowns_are_traceable():
    files, attendance = demo_sources()
    info, coverage, archive = rebuild(files, attendance, '2024-03', '2024-05', synthetic=True)
    assert info['county_months'] == 6 and info['models_fitted'] == info['imputed_values'] == 0
    table = unpack(archive, 'county_month.csv')
    assert len(table) == 18 and not table.duplicated(['county', 'outcome_month', 'menu']).any()
    oend = table[table.menu.eq('OEND')]
    # Missing stop status and an unreported month both remain unknown.
    assert oend.loc[oend.outcome_month.eq('2024-04'), 'active_snapshot'].isna().all()
    assert oend.loc[oend.outcome_month.eq('2024-04'), ['y_lo', 'y_hi']].eq([1, 9]).all().all()
    assert oend.loc[oend.outcome_month.eq('2024-05'), ['y_lo', 'y_hi']].eq(0).all().all()
    assert table.loc[table.menu.ne('OEND'), 'active_snapshot'].isna().all()
    assert coverage.set_index('menu').loc['OEND', 'all_three_available'] == 4
    assert len(unpack(archive, 'tracker_source_cells.csv')) == 7
    with ZipFile(BytesIO(archive)) as z:
        assert 'START_HERE.html' in z.namelist()
        assert len(json.loads(z.read('run_metadata.json'))['inputs']) == 5


def test_no_marks_are_not_zero_and_bad_inputs_fail():
    files, attendance = demo_sources()
    data = pd.read_csv(BytesIO(attendance[0][1])).fillna('')
    data['Attendance'] = ''
    _, coverage, _ = rebuild(files, [('blank.csv', data.to_csv(index=False).encode())],
                              '2024-03', '2024-05', synthetic=True)
    assert not coverage.attendance_available.any()
    files['annual'] = ('bad.csv', b'County,County Code,Year Code,Population,Deaths\nWrong,99001,2024,100,10\n')
    with pytest.raises(AuditError, match='disagree'):
        rebuild(files, attendance, '2024-03', '2024-05', synthetic=True)


def test_public_demo_download_and_clear():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / 'app.py'), default_timeout=90).run().switch_page('pages/04_ORCCA_Workflow.py').run()
    assert not at.exception
    assert not any(x.key.startswith('h3_sources_') for x in at.get('file_uploader'))
    at.button(key='h3_sources_run').click().run()
    assert not at.exception
    assert any(x.key == 'h3_sources_download' for x in at.get('download_button'))
    at.button(key='h3_sources_clear').click().run()
    assert not at.exception
    assert not any(x.key == 'h3_sources_download' for x in at.get('download_button'))


def test_replaced_or_removed_uploads_hide_previous_report(monkeypatch):
    import streamlit as st
    def source_ui():
        from orcca_audit.h3_sources_ui import render
        render(True)
    files, attendance = demo_sources()
    uploads = {}

    def file(pair):
        stream = BytesIO(pair[1])
        stream.name = pair[0]
        return stream

    def upload(label, *, key, **kwargs):
        return uploads.get(key, [] if kwargs.get('accept_multiple_files') else None)

    monkeypatch.setattr(st, 'file_uploader', upload)
    at = AppTest.from_function(source_ui, default_timeout=90).run()
    at.radio(key='h3_sources_route').set_value('Private research files').run()
    for role, pair in files.items():
        uploads['h3_sources_' + role + '0'] = file(pair)
    uploads['h3_sources_attendance0'] = [file(pair) for pair in attendance]
    at.text_input(key='h3_sources_start').set_value('2024-03')
    at.text_input(key='h3_sources_end').set_value('2024-05')
    at.run().button(key='h3_sources_run').click().run()
    assert not at.exception
    assert any(x.key == 'h3_sources_download' for x in at.get('download_button'))
    uploads['h3_sources_annual0'] = file(('changed.xls', files['annual'][1] + b'\n'))
    at.run()
    assert not any(x.key == 'h3_sources_download' for x in at.get('download_button'))
    uploads.pop('h3_sources_annual0')
    at.run()
    assert at.button(key='h3_sources_run').disabled
    assert not at.exception
