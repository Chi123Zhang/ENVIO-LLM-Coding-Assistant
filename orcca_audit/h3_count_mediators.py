"""Observed organization/site strategy-pair counts, with no numerical completion."""
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
import html
import json

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter

from .engine import AuditError
from .exports import safe_csv
from .h3_sources import BRANCH, code, tracker_measurements
from .h3_source_analysis import run_specification
from .io import read_workbook, read_table
from .pilot import branching
from .site_estimates import count

MEDIATORS = ('organization_strategy_pairs', 'fixed_site_strategy_pairs')
FIELDS = {'OEND': ('hcs_orc209p', 'hcs_orc209qb'),
          'MOUD': ('hcs_orc210pp', 'hcs_orc210qb'),
          'Safer Prescribing': ('hcs_orc211k', None)}


def aggregate(cells):
    """A missing active count or unresolved state invalidates the whole snapshot."""
    records=[]
    for (county, menu, month, mediator), group in cells.groupby(
            ['county','menu','mediator_month','mediator'], sort=True):
        active=group.state.eq('active')
        unresolved=~group.state.isin(['active','not_started','discontinued'])
        unusable=active & ~group.usable_active_count
        complete=not unresolved.any() and not unusable.any()
        records.append(dict(county=county,menu=menu,mediator_month=month,mediator=mediator,
            listed_slots=len(group),active_slots=int(active.sum()),unresolved_slots=int(unresolved.sum()),
            unusable_active_counts=int(unusable.sum()),complete=complete,
            value=float(group.loc[active,'count'].sum()) if complete else np.nan))
    return pd.DataFrame(records,columns=['county','menu','mediator_month','mediator','listed_slots',
        'active_slots','unresolved_slots','unusable_active_counts','complete','value'])


def extract(tracker, dictionary):
    _, states, _=tracker_measurements(tracker,dictionary)
    raw=read_workbook(tracker[1],positional_sheets=('Sheet1',))['orcca_tracker_v2']
    rows=raw.set_index('_source_row').to_dict('index')
    columns={name:get_column_letter(i) for i,name in enumerate(raw.columns,1)}
    dd=read_table(dictionary[1],dictionary[0],sheet='ORCCAT_DD')
    definitions={(str(r['Version']),r['Variable / Field Name']):r for r in dd.to_dict('records')}
    cells=[]
    for state in states.to_dict('records'):
        row=rows[state['tracker_row']]
        for mediator,prefix in zip(MEDIATORS,FIELDS[state['menu']]):
            field=f"{prefix}_{state['slot']}" if prefix else ''
            definition=definitions.get((state['version'],field),{})
            label=str(definition.get('Field Label','')).lower()
            label_matches=('partner organizations/practices are implementing' in label if mediator==MEDIATORS[0]
                           else 'fixed physical locations are currently delivering services' in label)
            rule=branching(definition.get(BRANCH),row) if label_matches and field in columns else 'definition_unavailable'
            value=count(row.get(field))
            cells.append(dict(county=state['county'],menu=state['menu'],slot=state['slot'],
                mediator_month=state['tracker_month'],mediator=mediator,state=state['state'],
                version=state['version'],field=field,recorded_value=code(row.get(field)),count=value,
                applicability=rule,usable_active_count=state['state']=='active' and rule=='applicable' and value is not None,
                source_file=tracker[0],source_sheet='orcca_tracker_v2',source_row=state['tracker_row'],
                source_cell=columns[field]+str(state['tracker_row']) if field in columns else '',
                dictionary_file=dictionary[0],dictionary_row=definition.get('_source_row'),
                dictionary_label=definition.get('Field Label',''),branch_expression=definition.get(BRANCH,'')))
    cells=pd.DataFrame(cells,columns=['county','menu','slot','mediator_month','mediator','state','version','field',
        'recorded_value','count','applicability','usable_active_count','source_file','source_sheet','source_row',
        'source_cell','dictionary_file','dictionary_row','dictionary_label','branch_expression'])
    return aggregate(cells),cells


def holm(values):
    """Retain unavailable rows while counting every planned test in the family."""
    values=pd.to_numeric(values,errors='coerce')
    valid=values.between(0,1).to_numpy()
    p=values.where(valid,1.).to_numpy()
    order=np.argsort(p,kind='stable')
    adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1.,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    adjusted[~valid]=np.nan
    return adjusted


def analyse(source_archive, tracker, dictionary):
    with ZipFile(BytesIO(source_archive)) as z:
        metadata=json.loads(z.read('run_metadata.json'))
        data=pd.read_csv(z.open('county_month.csv'))
    for role,pair in [('tracker',tracker),('dictionary',dictionary)]:
        matches=[r for r in metadata['inputs'] if r['role']==role]
        if len(matches)!=1 or matches[0]['sha256']!=sha256(pair[1]).hexdigest():
            raise AuditError('Rebuild source inputs: count definitions do not match the reconstruction.')
    snapshots,cells=extract(tracker,dictionary)
    if snapshots.empty:
        for mediator in MEDIATORS:data[mediator]=np.nan
    else:
        wide=snapshots.pivot(index=['county','menu','mediator_month'],columns='mediator',values='value').reset_index()
        data=data.merge(wide,on=['county','menu','mediator_month'],how='left',validate='many_to_one')
    results=[];checks=[];replicates=[];frames=[]
    for menu in FIELDS:
        for mediator in MEDIATORS:
            check,estimates,reps,frame=run_specification(data,menu,mediator)
            checks.append(check);replicates.extend(reps)
            frames.append(frame.assign(specification_mediator=mediator))
            for term in ['attendance_to_implementation','implementation_to_CDC_count_rate','path_product']:
                results.append(next((r for r in estimates if r['term']==term),None) or
                    dict(menu=menu,mediator=mediator,term=term,county_months=check['county_months'],
                         counties=check['counties'],p_approx=np.nan,estimate=np.nan,low=np.nan,high=np.nan))
    results=pd.DataFrame(results)
    results['p_holm_18_terms']=holm(results.p_approx)
    paths=results[results.term.eq('path_product')].copy()
    paths['p_holm_six_paths']=holm(paths.p_approx)
    paths['significant_unadjusted']=pd.array([bool(p<.05) if pd.notna(p) else None for p in paths.p_approx],dtype='boolean')
    diagnostics=pd.DataFrame(checks)
    protocol=dict(measurement='Observed strategy-pair counts, not unique organizations/sites or people served',
        timing='Recorded attendance t-2, observed counts t-1, CDC count t',
        aggregation='All listed states resolved; every active count applicable and recorded; no count carry-forward',
        planned_paths=6,inference_available=int(paths.p_approx.notna().sum()),
        significant_paths=int(paths.p_approx.lt(.05).sum()),original_values_changed=0,imputed_values=0,
        scope='Exploratory measurement sensitivity, not confirmatory causal mediation',
        inputs=metadata['inputs'],outcome_window=metadata['outcome_window'])
    page=('<!doctype html><html lang="en"><meta charset="utf-8"><title>Observed count H3 sensitivity</title>'
          '<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem}'
          'td,th{padding:.5rem;text-align:left}pre{white-space:pre-wrap}</style>'
          '<h1>H3 observed organization/site counts</h1>'
          '<p>Counts sum recorded organizations or fixed delivery sites across active strategy slots. '
          'Repeated organizations/sites across strategies may be counted more than once. These are '
          'strategy-pair measures, not unique entities or service reach. Original cells are unchanged; '
          'no attendance or mediator values are imputed. Inactive slots contribute zero only when their '
          'state is explicit. Missing active counts and unresolved states remain unknown.</p>'
          '<h2>All six indirect-path attempts</h2>'+paths.to_html(index=False,na_rep='Unavailable')+
          '<p>Indirect-path ratio exp(a×b), null 1. Approximate unadjusted and Holm-adjusted p-values '
          'are exploratory. Original county effects, linear trend, population offset, interval Poisson '
          'and delete-county jackknife are retained. Unavailable fits are not null results. '
          'Small sample, confounding and unconfirmed CDC endpoint filters limit interpretation.</p>'
          '<h2>All link and path tests</h2>'+results.to_html(index=False,na_rep='Unavailable')+
          '<h2>Diagnostics</h2>'+diagnostics.to_html(index=False,na_rep='')+
          '<h2>Protocol</h2><pre>'+html.escape(json.dumps(protocol,indent=2))+'</pre></html>')
    out=BytesIO()
    with ZipFile(out,'w',ZIP_DEFLATED) as z:
        z.writestr('START_HERE.html',page)
        z.writestr('protocol.json',json.dumps(protocol,indent=2))
        for name,table in [('paths',paths),('all_link_and_path_tests',results),('fit_diagnostics',diagnostics),
                           ('count_aggregates',snapshots),('count_cell_basis',cells),
                           ('delete_county_fits',pd.DataFrame(replicates)),('included_model_rows',pd.concat(frames))]:
            z.writestr(name+'.csv',safe_csv(table))
    return protocol,paths,diagnostics,out.getvalue()


def render(source_archive,tracker,dictionary):
    import streamlit as st
    from .h3_results_ui import render_path_results
    fingerprint=sha256(source_archive+tracker[1]+dictionary[1]).hexdigest()
    with st.expander('Test H3 with observed organization and delivery-site counts'):
        st.write('Use recorded implementing organizations and currently delivering fixed sites as alternative '
                 'intermediate measures. Counts sum organization–strategy or site–strategy pairs; '
                 'they do not measure unique entities or people reached. Missing counts stay unknown.')
        stored=st.session_state.get('h3_count_mediators')
        if stored and stored[0]!=fingerprint:
            st.session_state.pop('h3_count_mediators',None);stored=None
        if st.button('Run six observed-count H3 paths',key='h3_counts_run'):
            st.session_state.pop('h3_count_mediators',None);stored=None
            try:
                with st.spinner('Checking count definitions and all six model specifications…'):
                    result=analyse(source_archive,tracker,dictionary)
                stored=(fingerprint,result);st.session_state['h3_count_mediators']=stored
            except (AuditError,ValueError,KeyError,ArithmeticError) as error:
                st.error('Observed-count analysis failed. '+str(error))
        if stored:
            protocol,paths,checks,archive=stored[1]
            render_path_results(paths,key='h3_counts')
            st.dataframe(paths[['menu','mediator','p_holm_six_paths']],hide_index=True)
            st.caption('Holm adjustment covers all six attempted indirect paths. Unavailable fits stay visible.')
            st.dataframe(checks,hide_index=True)
            st.download_button('Download observed-count H3 report',archive,'h3_observed_counts.zip',
                               'application/zip',key='h3_counts_download')
