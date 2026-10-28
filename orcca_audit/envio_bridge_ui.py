"""Small UI handoff; does not run or reconfigure ENVIO coding."""
import streamlit as st

from .engine import AuditError
from .envio_bridge import transfer_segment
from .policy import private_uploads_allowed
from .workflow import current_project


def render(signature):
    st.divider()
    st.subheader('Continue to ORCCA')
    st.page_link('pages/04_ORCCA_Workflow.py', label='Open ORCCA project workflow')
    if not private_uploads_allowed():
        st.caption('Public ENVIO and ORCCA demonstrations remain separate. Research coding handoff requires the private workspace; no research data are transferred here.')
        return
    capture = st.session_state.get('envio_orcca_capture')
    project = current_project(st.session_state, True)
    if not capture:
        st.info('Complete an authorized ENVIO coding run first. The handoff makes no new model call.')
        return
    with st.expander('Captured coding run and settings'):
        st.write('Run '+capture['capture_id'][:12]+' · '+capture['created_at_utc'])
        st.json(capture['settings'])
        st.caption('RAG settings describe the requested ENVIO run. This bridge does not retain its retrieved context or certify successful retrieval.')
    if capture['input_signature'] != signature:
        st.warning('This captured run differs from the current page inputs or the upload widgets were cleared during navigation. It is not a result for the inputs currently shown.')
        if not st.checkbox('Use this explicitly selected earlier coding run',key='envio_use_capture_'+capture['capture_id']):return
    if not project or project['classification'] != 'private_research':
        st.info('Load your private ORCCA project in Workflow, then return here. Coding results stay in this browser session.')
        return
    st.caption('ENVIO labels are model outputs from its existing codebook, not validated ORCCA variables. Only an exact source quotation enters evidence search; labels and rationale stay in the separate review ledger. No new model call is made by this handoff.')
    from .study import counties
    rows = capture['rows']
    if not rows:return
    selected = st.selectbox('Coded segment to link', range(len(rows)),
        format_func=lambda i: f"{rows[i]['source_name']} · segment {rows[i].get('segment_index',i+1)}", key='envio_link_segment')
    row = rows[selected]
    st.write('ENVIO codes: '+str(row.get('codes','')))
    st.code(str(row.get('text','')),language=None)
    with st.form('envio_handoff_'+project['run_id']+'_'+capture['capture_id']):
        quote = st.text_area('Exact original quotation (must also occur in the source)')
        relevant = st.checkbox('I checked that this source concerns this ORCCA project, not an unrelated ENVIO participant')
        county = st.selectbox('Applicable county (optional; human assigned)', ['']+counties(project))
        month = st.text_input('Applicable event month (optional, YYYY-MM; not the interview date by default)')
        reviewer = st.text_input('Applicability reviewer')
        reason = st.text_area('Why this source and interpretation apply to ORCCA (English)')
        save = st.form_submit_button('Link candidate and create ORCCA version')
    if save:
        try:
            transfer_segment(st.session_state,project,capture,selected,quote,reviewer,reason,relevant,county,month)
            st.success('Candidate linked. Generate new draft findings in Workflow and review them before export. Prior finding approvals have expired. No numeric field was filled.')
        except AuditError as exc:st.error(str(exc))
