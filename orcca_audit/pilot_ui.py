"""Browser-operated, source-only feasibility pilot; no research values bundled."""
import pandas as pd
import streamlit as st

from .engine import AuditError
from .pilot import run, exports


def render(project):
    with st.expander('Completion evaluation · run a small pilot'):
        st.write('Compare a conservative rules baseline with ENVIO BM25 and frozen source memory. This first experiment checks whether the supplied evidence can support value recovery. It does not train a model.')
        st.caption('Only slot-1 implementation-start statuses are screened. Applicable observed cells can be masked; true unknowns and skip cases receive separate human-review forms. Whole counties are separated for development and heldout pilot review.')
        st.warning('Definitions explain a field; they cannot establish a county-month value. With definitions alone, both methods must retain unknowns. No recovery or accuracy gain is claimed.')
        if st.button('Run current-input completion pilot',key='pilot_run_button'):
            try:
                result=run(project)
                st.session_state['pilot_result']=result
            except AuditError as exc:
                st.session_state.pop('pilot_result',None);st.error(str(exc))
        r=st.session_state.get('pilot_result')
        if not r:return
        if r['project_run_id']!=project['run_id']:
            st.info('Inputs or project version changed. Rerun the pilot; previous results are not current.');return
        a,b,c=st.columns(3)
        a.metric('Observed cells masked',r['summary']['masking_cases'])
        b.metric('Actual BM25 calls',r['summary']['actual_bm25_calls'])
        c.metric('Independent human labels',0)
        st.subheader('Field applicability review')
        labels={'observed_applicable':'Recorded and applicable','unknown_applicable':'Unknown; applicable',
            'structural_skip':'Display condition not met','unresolved_branching':'Applicability unresolved'}
        st.dataframe(pd.DataFrame([{'Classification':labels.get(k,k.replace('_',' ')),'Cells':v} for k,v in r['summary']['eligibility_counts'].items()]),hide_index=True)
        st.subheader('Recorded-value masking results')
        table=pd.DataFrame(r['metrics'])[['split','method','masked_cases','proposed','unknown']]
        table['split']=table['split'].replace({'development':'Development','heldout_county_pilot':'Heldout counties'})
        table['method']=table['method'].replace({'rules_only':'Exact rules','bm25_source_memory':'BM25 + source memory'})
        st.dataframe(table.rename(columns={'split':'Group','method':'Method','masked_cases':'Masked','proposed':'Proposed','unknown':'Unknown'}),hide_index=True)
        st.caption('Correct and incorrect are counts against recorded values. Accuracy among proposed values is undefined when neither method proposes a value. This is not an independent human accuracy evaluation.')
        st.subheader('Frozen evidence memory')
        st.code(r['memory']['memory_hash'],language=None)
        st.write(r['memory']['policy'])
        st.caption('Memory contains source definitions only; it is not LSTM memory, model training, or learned county status. Test answers never update it.')
        case=st.selectbox('Inspect a retrieval case',[x['case_id'] for x in r['retrievals']],key='pilot_case_'+project['run_id'])
        log=next(x for x in r['retrievals'] if x['case_id']==case)
        st.write('BM25 status: '+log['status'])
        for d in log['citations']:
            st.caption(d['source_file']+' · '+str(d.get('locator',d['line_start'])))
            st.code(d['text'],language=None)
        st.download_button('Download pilot and separate review forms',exports(r),'orcca_completion_pilot.zip','application/zip',key='pilot_download')
        st.caption('Coordinator package: give annotators only their A file and original sources. Keep system outputs and sealed recorded references separate until both reviews are submitted. Private research exports remain private.')
