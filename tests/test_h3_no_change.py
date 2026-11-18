from io import BytesIO
from hashlib import sha256
import json
from zipfile import ZipFile

import pandas as pd
import pytest

import orcca_audit.h3_no_change as module
from orcca_audit.engine import AuditError


def setup_source(monkeypatch, flags, dates=None, listed=None, observed=None):
    dates=dates or ['2024-01','2024-02','2024-03']
    listed=listed or [1,0,0]
    observed=observed or [2,None,None]
    rows=[];source=[]
    for i,(month,flag) in enumerate(zip(dates,flags)):
        source.append(dict(hcs_orc2record_id=1,redcap_event_name=month,hcs_orc2version='v1',
            hcs_orc2a=1,hcs_orc2b=flag,hcs_orc2b1='',hcs_orc2b2='',hcs_orc2b3='',_source_row=i+2))
        for menu in module.MENU_FIELDS:
            rows.append(dict(county='DEMO',menu=menu,mediator_month=month,
                outcome_month=str(pd.Period(month,freq='M')+1),attendance=5,
                attendance_available=True,active_snapshot=observed[i],venue_types=observed[i],
                listed_slots=listed[i],outcome_available=True))
    dd=[]
    for i,field in enumerate(['hcs_orc2b',*module.MENU_FIELDS.values()]):
        dd.append(dict(Version='v1',**{'Variable / Field Name':field,
             'Field Label':"Have there been any changes to last month's ORCCAT entries?",
             module.CHOICES:'1, Yes, progress | 0, No, there are no changes to report since last month.',
             module.BRANCH:'verified synthetic fixture only'},_source_row=i+2))
    monkeypatch.setattr(module,'event_month',lambda v:v)
    monkeypatch.setattr(module,'read_workbook',lambda *a,**k:dict(orcca_tracker_v2=pd.DataFrame(source),Sheet1=pd.DataFrame([[1,'DEMO','']])))
    monkeypatch.setattr(module,'read_table',lambda *a,**k:pd.DataFrame(dd))
    raw=b'synthetic tracker';dictionary=b'synthetic dictionary'
    metadata=dict(inputs=[dict(role=role,sha256=sha256(value).hexdigest()) for role,value in [('tracker',raw),('dictionary',dictionary)]])
    stream=BytesIO()
    with ZipFile(stream,'w') as z:
        z.writestr('county_month.csv',pd.DataFrame(rows).to_csv(index=False))
        z.writestr('run_metadata.json',json.dumps(metadata))
    return stream.getvalue(),('tracker.xlsx',raw),('dictionary.xlsx',dictionary)


def test_explicit_no_change_reconstructs_only_known_adjacent_history(monkeypatch):
    args=setup_source(monkeypatch,[1,0,0])
    meta,coverage,ledger,audits,report=module.reconstruct_no_change(*args)
    assert len(ledger)==12 and ledger.reconstructed_value.eq(2).all()
    assert ledger.tracker_month.min()=='2024-02'
    assert meta['imputed_values']==0 and not meta['rule_human_adjudicated']


@pytest.mark.parametrize('flags', [[1,None,0],[1,1,0]])
def test_missing_or_changed_update_response_does_not_carry(monkeypatch,flags):
    args=setup_source(monkeypatch,flags)
    assert module.reconstruct_no_change(*args)[2].empty


def test_no_carry_across_an_absent_tracker_month(monkeypatch):
    args=setup_source(monkeypatch,[1,0,0],dates=['2024-01','2024-03','2024-04'])
    assert module.reconstruct_no_change(*args)[2].empty


def test_unresolved_listed_strategies_are_not_repaired_by_no_change(monkeypatch):
    args=setup_source(monkeypatch,[1,0,0],listed=[1,1,0])
    assert module.reconstruct_no_change(*args)[2].empty


def test_original_observed_conflict_is_preserved_and_blocks_descendants(monkeypatch):
    args=setup_source(monkeypatch,[1,0,0],observed=[2,3,None],listed=[1,1,0])
    meta,coverage,ledger,audit,report=module.reconstruct_no_change(*args)
    assert ledger.empty and audit.conflicts.ne('').sum()==3
    with ZipFile(BytesIO(report)) as z:
        data=pd.read_csv(z.open('county_month.csv'))
    assert set(data.loc[data.mediator_month.eq('2024-02'),'active_snapshot'])=={3}


def test_initial_unknown_is_not_imputed(monkeypatch):
    args=setup_source(monkeypatch,[0,0,0],observed=[None,None,None],listed=[0,0,0])
    assert module.reconstruct_no_change(*args)[2].empty


def test_source_hash_mismatch_fails_closed(monkeypatch):
    args=setup_source(monkeypatch,[1,0,0])
    with pytest.raises(AuditError,match='frozen'):
        module.reconstruct_no_change(args[0],('wrong.xlsx',b'changed'),args[2])
