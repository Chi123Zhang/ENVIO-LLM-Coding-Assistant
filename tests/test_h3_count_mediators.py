from copy import deepcopy
from io import BytesIO
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from orcca_audit.engine import AuditError
from orcca_audit.h3_count_mediators import aggregate, analyse, extract, holm
from orcca_audit.h3_sources import demo_sources, rebuild


def fixtures():
    files,attendance=demo_sources()
    tracker=load_workbook(BytesIO(files['tracker'][1]))
    sheet=tracker['orcca_tracker_v2']
    for field in ['hcs_orc209p_1','hcs_orc209qb_1','hcs_orc209k_1']:
        sheet.cell(1,sheet.max_column+1,field)
    for r in range(2,sheet.max_row+1):
        sheet.cell(r,sheet.max_column-2,3)
        sheet.cell(r,sheet.max_column-1,2)
        sheet.cell(r,sheet.max_column,1)
    dictionary=load_workbook(BytesIO(files['dictionary'][1]))
    dd=dictionary['ORCCAT_DD']
    rule="([hcs_orc209k_1]='1' or [missing]='1') and [hcs_orc209l_1]='1'"
    dd.append(['DEMO','hcs_orc209p_1','How many partner organizations/practices are implementing the strategy?',rule,''])
    dd.append(['DEMO','hcs_orc209qb_1','How many fixed physical locations are currently delivering services to individuals?',rule,''])
    for role,book in [('tracker',tracker),('dictionary',dictionary)]:
        out=BytesIO();book.save(out);files[role]=(role+'.xlsx',out.getvalue())
    return files,attendance


def test_all_active_counts_required_and_inactive_zero_not_missing_zero():
    def row(state,value,usable):
        return dict(county='A',menu='OEND',mediator_month='2024-01',
                    mediator='organization_strategy_pairs',state=state,count=value,usable_active_count=usable)
    for rows,expected in [([row('active',3,True),row('active',2,True)],5),
                          ([row('not_started',None,False)],0),
                          ([row('active',0,True)],0),
                          ([row('active',None,False)],None),
                          ([row('active',3,True),row('unresolved',5,False)],None),
                          ([row('active',3,False)],None)]:
        value=aggregate(pd.DataFrame(rows)).iloc[0].value
        assert pd.isna(value) if expected is None else value==expected


def test_grouped_definitions_source_counts_and_no_carry():
    files,_=fixtures();before=deepcopy(files)
    snapshots,cells=extract(files['tracker'],files['dictionary'])
    assert files==before
    assert cells.loc[cells.usable_active_count,'count'].isin([2,3]).all()
    assert cells.source_cell.str.len().gt(0).all()
    assert cells.dictionary_row.notna().all()
    org=snapshots[snapshots.mediator.eq('organization_strategy_pairs')]
    assert org.loc[org.mediator_month.eq('2024-01'),'value'].eq(3).all()
    assert org.loc[org.county.eq('Demo 1')&org.mediator_month.eq('2024-03'),'value'].isna().all()
    assert not org.mediator_month.eq('2024-05').any()


def test_bundle_keeps_all_attempts_unavailable_and_rejects_changed_sources():
    files,attendance=fixtures()
    _,_,source=rebuild(files,attendance,'2024-03','2024-05',synthetic=True)
    meta,paths,checks,report=analyse(source,files['tracker'],files['dictionary'])
    assert len(paths)==len(checks)==6 and paths.p_approx.isna().all()
    assert paths.significant_unadjusted.isna().all()
    assert meta['imputed_values']==meta['original_values_changed']==0
    with ZipFile(BytesIO(report)) as z:
        assert z.testzip() is None
        assert len(pd.read_csv(z.open('all_link_and_path_tests.csv')))==18
        assert 'Unavailable' in z.read('START_HERE.html').decode()
        assert json.loads(z.read('protocol.json'))['planned_paths']==6
    with pytest.raises(AuditError,match='Rebuild'):
        analyse(source,('modified.xlsx',b'changed'),files['dictionary'])


def test_holm_counts_unavailable_attempts_without_reporting_them_as_null():
    actual=holm(pd.Series([.01,np.nan,.04,.005]))
    assert np.allclose(actual[[0,2,3]],[.03,.08,.02])
    assert np.isnan(actual[1])


def test_count_ui_exports_unavailable_paths():
    from streamlit.testing.v1 import AppTest
    files,attendance=fixtures()
    _,_,source=rebuild(files,attendance,'2024-03','2024-05',synthetic=True)
    def page(source,tracker,dictionary):
        from orcca_audit.h3_count_mediators import render
        render(source,tracker,dictionary)
    at=AppTest.from_function(page,args=(source,files['tracker'],files['dictionary']),default_timeout=90).run()
    at.button(key='h3_counts_run').click().run()
    assert not at.exception
    assert len(at.session_state['h3_count_mediators'][1][1])==6
    assert any(x.key=='h3_counts_download' for x in at.get('download_button'))
    assert any('6 paths have unavailable' in x.value for x in at.warning)
