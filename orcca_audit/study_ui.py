"""H3 preparation UI; upload, inspect, retrieve, review and export."""
import json
import hashlib
import pandas as pd
import streamlit as st

from .engine import AuditError
from .policy import private_uploads_allowed
from .workflow import current_project, start_project, invalidate, export_project, build_findings
from .study import parse_inputs, sample, search


def render_audit(project):
    r=project['results'];s=r['structure']
    a,b,c=st.columns(3);a.metric('Counties',s['counties']);b.metric('Tracker records',s['records']);c.metric('Event months',s['event_months'])
    st.caption('Repeated records within counties. These are not independent outcome samples.')
    st.subheader('H3 data requirements')
    st.dataframe(pd.DataFrame(r['h3_readiness']),hide_index=True)
    st.subheader('Representative field and version review')
    profiles=r['field_review']
    if profiles:
        versions=sorted(set(x['version'] for x in profiles))
        chosen=st.selectbox('Questionnaire version',versions,key='study_audit_version')
        st.dataframe(pd.DataFrame([x for x in profiles if x['version']==chosen]),hide_index=True)
    else:st.info('No representative field definitions were available. Add a source dictionary; no definitions are guessed.')
    st.caption('Raw nonblank counts describe the export. Branching, carry-forward rules and strategy identity still require review. Slot 1 is a review example, not the complete strategy population.')
    if r.get('completion_overview'):st.json(r['completion_overview'])


def _reset_search():
    st.session_state['study_query']=''
    st.session_state.pop('study_search_result',None)


def render_search(project):
    st.subheader('Find evidence for a specific question')
    st.caption('An explicit field ID uses exact dictionary lookup. Other questions use source retrieval. Neither operation generates an answer or fills a field.')
    q=st.text_input('Question or field ID',key='study_query')
    versions=['All supplied versions']+sorted(set(d['version'] for d in project['documents'] if d.get('version')))
    version=st.selectbox('Evidence version',versions,key='study_evidence_version')
    backend=st.selectbox('Retrieval method',['bm25','hybrid'],key='study_backend')
    finger=hashlib.sha256(json.dumps([project['run_id'],q,version,backend]).encode()).hexdigest()
    if st.button('Find source evidence',key='study_search'):
        try:result=search(q,project['documents'],backend,None if version==versions[0] else version)
        except Exception:result={'status':'retrieval_error_no_fallback','citations':[],'backend':backend}
        st.session_state['study_search_result']=(finger,result)
    cached=st.session_state.get('study_search_result')
    if cached and cached[0]==finger:
        result=cached[1];st.write('Status: '+result['status']);st.caption('Method: '+result.get('backend',backend))
        if not result['citations']:st.info('No supporting passage was found. Leave the question unresolved.')
        for d in result['citations']:
            st.markdown('**'+d['source_file']+'**');st.caption(d.get('locator',f"Lines/rows {d['line_start']}–{d.get('line_end',d['line_start'])}"));st.code(d['text'],language=None)
    elif cached:st.caption('Inputs or query changed. Previous retrieval is not current.')
    st.button('Reset evidence query',key='study_reset_query',on_click=_reset_search)
    st.write('For a missing status, open Evidence Completion, inspect the candidate and its source, and record an explicit review decision.')
    st.page_link('pages/03_ORCCA_Evidence_Completion.py',label='Review source-backed completion candidates')
    st.page_link('app.py',label='Optional: review an authorized ENVIO coding result')
    st.caption("Interviews, minutes and attendance remain auxiliary. Reuse Eric's existing work where available. No generic ENVIO label becomes a validated H3 measure automatically.")


def render():
    from .workflow_ui import navigation, project_header, render_review, update_input, retained_inputs
    st.title('ORCCA H3 Study Workspace')
    st.caption('Prepare one source-linked mediation question. H4 and H5 are deferred.')
    navigation()
    st.sidebar.page_link('pages/01_ORCCA_Research_Audit.py',label='Current data review')
    st.sidebar.page_link('pages/02_ORCCA_Research_Findings.py',label='Current findings')
    st.sidebar.page_link('pages/03_ORCCA_Evidence_Completion.py',label='Evidence completion')
    a,b,c=st.columns(3)
    a.markdown('**1 · Coalition process (X)**');a.caption('One defined exposure, source and applicable time')
    b.markdown('**2 · EBP implementation / reach (M)**');b.caption('A validated intermediate measure from the Tracker or reach data')
    c.markdown('**3 · Later overdose outcome (Y)**');c.caption('A later, consistently defined outcome and confounders')
    st.info('This workspace prepares and reviews the pathway. It does not estimate mediation effects or recover suppressed death counts.')
    private=private_uploads_allowed()
    st.caption('Private session: research uploads stay in this server session; external generation is disabled.' if private else 'Public practice: synthetic data only. Research uploads and external generation are disabled.')
    with st.expander('Suggested demo route'):
        st.write('Load the synthetic project → inspect available and missing H3 components → retrieve a source passage → review one status candidate → create a derived version → generate and review current findings → download the report.')
        st.caption('Synthetic examples demonstrate the workflow, not empirical findings. Missing coalition or outcome data remain visible. A real model call is not required for this demo.')
    if st.button('Clear current study project',key='study_clear'):
        invalidate(st.session_state,'Study project cleared.');st.session_state.pop('workflow_input_files',None)
        st.session_state['workflow_upload_epoch']=st.session_state.get('workflow_upload_epoch',0)+1;st.rerun()
    tabs=st.tabs(['1 · Project inputs','2 · Data review','3 · Evidence and completion','4 · Findings and review','5 · Export'])
    with tabs[0]:
        st.subheader('Start with the Tracker')
        st.write('No H2a panel is required. The Tracker is the primary implementation source; the other files add definitions, potential outcomes or auxiliary evidence.')
        if private and st.radio('Input source',['Synthetic practice','Private research upload'],key='study_input_route',on_change=invalidate,args=(st.session_state,'Input source changed. Load the intended project again.'))=='Private research upload':
            epoch=str(st.session_state.get('workflow_upload_epoch',0))
            for role,label,types in [('raw_tracker','Original Tracker workbook',['xlsx']),('dictionary','REDCap dictionary (recommended)',['xlsx']),
                ('monthly','Monthly mortality (optional)',['csv','tsv','xls','xlsx']),('annual','Annual mortality (optional)',['csv','tsv','xls','xlsx']),
                ('coalition_process','Defined coalition-process table (optional)',['csv','tsv']),('completion_evidence','Scoped completion source excerpts (optional)',['csv','tsv'])]:
                key='study_upload_'+role+'_'+epoch
                st.file_uploader(label,type=types,key=key,on_change=update_input,args=(st.session_state,role,key))
                kept=st.session_state.get('workflow_input_files',{}).get(role)
                if kept:
                    st.caption('Retained: '+kept[0]['name'])
                    if st.button('Remove '+label,key='study_remove_'+role+'_'+epoch):
                        st.session_state['workflow_input_files'].pop(role,None);invalidate(st.session_state,'Input removed. Reload before reviewing findings.')
                        st.session_state['workflow_upload_epoch']=int(epoch)+1;st.rerun()
            key='study_upload_documents_'+epoch
            st.file_uploader('Auxiliary documents (optional; PDF, text, DOCX or PPTX)',type=['pdf','txt','md','docx','pptx'],accept_multiple_files=True,key=key,
                on_change=update_input,args=(st.session_state,'documents',key,True))
            kept=st.session_state.get('workflow_input_files',{}).get('documents',[])
            if kept:
                st.caption('Retained auxiliary sources: '+', '.join(x['name'] for x in kept))
                if st.button('Remove auxiliary documents',key='study_remove_documents_'+epoch):
                    st.session_state['workflow_input_files'].pop('documents',None);invalidate(st.session_state,'Auxiliary evidence removed. Reload the project.')
                    st.session_state['workflow_upload_epoch']=int(epoch)+1;st.rerun()
            st.caption('Coalition table: county, month, measure, value, source_ref. Supply only defined, authorized measures; attendance is not selected automatically. An interview date is not automatically the event time.')
            if st.button('Load Tracker and review inputs',key='study_build'):
                try:
                    supplied,docs=retained_inputs(st.session_state);invalidate(st.session_state,'Parsing study inputs.')
                    start_project(st.session_state,*parse_inputs(supplied,docs),private=True);st.success('Study inputs loaded. Continue to Data review.')
                except Exception as exc:
                    invalidate(st.session_state,'Study inputs could not be loaded. Previous findings are not current.')
                    st.error(str(exc) if isinstance(exc,AuditError) else 'Parsing failed. No old project or sample was substituted.')
        st.subheader('Practice without research data')
        variant=st.selectbox('Synthetic practice case',['Unresolved status','Changed input'],key='study_sample',on_change=invalidate,args=(st.session_state,))
        if st.button('Load synthetic H3 project',key='study_sample_build'):
            start_project(st.session_state,*sample(variant));st.success('Synthetic practice project loaded.')
        project=current_project(st.session_state,private)
        if project:project_header(project)
        elif st.session_state.get('workflow_active'):st.warning(st.session_state.get('workflow_invalid_reason','Load the Tracker first.'))
    if not project:
        for tab in tabs[1:]:
            with tab:st.info('Load a Tracker or the synthetic practice project in step 1.')
        return
    with tabs[1]:render_audit(project)
    with tabs[2]:
        render_search(project)
        st.caption(project['data'].get('completion_status','Status completion unavailable.'))
        from .pilot_ui import render as render_pilot
        render_pilot(project)
    with tabs[3]:
        if st.button('Generate current study findings',key='study_draft'):
            build_findings(project);st.success('Draft preparation findings generated. Review each one; no H3 effect was estimated.')
        render_review(project,show_findings=True)
        st.page_link('pages/02_ORCCA_Research_Findings.py',label='Open current findings and sources')
    with tabs[4]:
        if not project.get('bundle'):st.info('Generate and review current findings before export.')
        else:
            approved=st.checkbox('Export accepted findings only',value=True,key='study_approved_'+project['run_id'])
            try:st.download_button('Download H3 study review',export_project(project,approved),'orcca_h3_study_review.zip','application/zip',key='study_export')
            except AuditError as exc:st.info(str(exc))
        st.caption('The report contains source references, readiness gaps and review lineage. Exported preparation findings are not hypothesis-testing results.')
