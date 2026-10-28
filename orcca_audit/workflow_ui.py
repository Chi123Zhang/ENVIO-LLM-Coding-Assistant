"""Browser-operated intake and handoff; private uploads remain server-gated."""
import hashlib
import json

import pandas as pd
import streamlit as st

from .adapters import tracker_workbooks_to_tidy
from .completion import identify_issues
from .demo import demo_data, public_documents
from .engine import AuditError, compare_specifications
from .io import read_table, read_workbook, normalize_mortality, extract_document
from .policy import private_uploads_allowed
from .workflow import (build_findings, current_project, decide_finding, export_project,
    invalidate, start_project)


class InputFile:
    def __init__(self, name, data):self.name, self.data = name, data
    def getvalue(self):return self.data


def update_input(state, role, key, multiple=False):
    """Durable session buffers survive Streamlit's cross-page widget cleanup."""
    value=state.get(key)
    values=(value or []) if multiple else ([value] if value is not None else [])
    pool=state.setdefault('workflow_input_files',{})
    if values:pool[role]=[{'name':f.name,'data':f.getvalue()} for f in values]
    else:pool.pop(role,None)
    invalidate(state,'Inputs changed. Rerun the audit; previous findings and approvals are no longer current.')


def retained_inputs(state):
    pool=state.get('workflow_input_files',{})
    total=sum(len(f['data']) for files in pool.values() for f in files)
    if total>100*1024*1024:raise AuditError('Combined project inputs exceed the 100 MB limit.')
    supplied={k:InputFile(v[0]['name'],v[0]['data']) for k,v in pool.items() if k!='documents' and v}
    docs=[InputFile(v['name'],v['data']) for v in pool.get('documents',[])]
    return supplied,docs


def navigation():
    st.sidebar.page_link("pages/04_ORCCA_Workflow.py", label="Start / manage project")


def project_header(project):
    st.caption(f"Current web project · version {project['revision']} · {project['run_id'][:12]} · "
        + ("Private session" if project['classification'] == 'private_research' else 'Synthetic practice'))
    st.caption("Tracker → data review → source evidence → human review → H3 preparation findings → export" if project['data'].get('_study_kind')=='h3'
        else "Upload → audit → evidence completion → rerun → findings review → export")
    st.page_link("pages/04_ORCCA_Workflow.py", label="Manage inputs and project versions")


def synthetic_project_inputs(variant="Baseline"):
    data = demo_data()
    if variant == "Changed panel":data['panel'].loc[0, ['Deaths', 'y_lo', 'y_hi']] = 999
    evidence = pd.DataFrame([{"county": "DEMO_A", "month": "2024-01", "menu": "MOUD", "slot": "1", "version": "",
        "text": "SYNTHETIC source record for workflow practice\ndiscontinued=0", "source_ref": "Invented practice record, not study evidence",
        "source_file": "synthetic_completion_evidence.csv", "_source_row": 2}])
    return data, public_documents(), evidence, {"synthetic_case": variant, "synthetic": True}


def parse_uploads(supplied, evidence_files):
    required = ['panel', 'monthly', 'annual']
    if not all(supplied.get(k) is not None for k in required):raise AuditError("Panel, monthly and annual sources are required together.")
    if (supplied.get('raw_tracker') is None) != (supplied.get('dictionary') is None):raise AuditError("Supply both Tracker workbook and dictionary.")
    if supplied.get('tracker') is not None and supplied.get('raw_tracker') is not None:raise AuditError("Choose a tidy tracker or the workbook pair, not both.")
    data = {k: read_table(supplied[k].getvalue(), supplied[k].name) for k in required}
    for k in ('monthly', 'annual'):data[k] = normalize_mortality(data[k])
    if supplied.get('tracker') is not None:data['tracker'] = read_table(supplied['tracker'].getvalue(), supplied['tracker'].name)
    elif supplied.get('raw_tracker') is not None:
        dd = read_table(supplied['dictionary'].getvalue(), supplied['dictionary'].name, sheet='ORCCAT_DD')
        data['tracker'], refs = tracker_workbooks_to_tidy(read_workbook(supplied['raw_tracker'].getvalue(), positional_sheets=('Sheet1',)), dd, supplied['raw_tracker'].name)
        data['field_references'] = refs.to_dict('records')
    data['specifications'] = []
    if supplied.get('specs') is not None:
        try:data['specifications'] = json.loads(supplied['specs'].getvalue())
        except (ValueError, UnicodeError) as exc:raise AuditError("Model specification JSON is invalid.") from exc
        compare_specifications(data['specifications'])
    if supplied.get('results') is not None:data['reported_results'] = read_table(supplied['results'].getvalue(), supplied['results'].name)
    evidence = read_table(supplied['completion_evidence'].getvalue(), supplied['completion_evidence'].name) if supplied.get('completion_evidence') is not None else None
    if evidence is not None:
        from .completion import prepare_evidence
        prepare_evidence(evidence)  # Fail before replacing the project.
    documents = []
    for uploaded in evidence_files:documents.extend(extract_document(uploaded.getvalue(), uploaded.name))
    if supplied.get('results') is not None:
        documents.extend(extract_document(supplied['results'].getvalue(), supplied['results'].name))
    names = [f.name for f in [v for v in supplied.values() if v is not None] + evidence_files]
    if len(names) != len(set(names)):raise AuditError("Source filenames must be unique within a project.")
    manifest = {role: {"source_file": f.name, "sha256": hashlib.sha256(f.getvalue()).hexdigest()}
        for role, f in supplied.items() if f is not None}
    manifest['evidence_files'] = [{"source_file": f.name, "sha256": hashlib.sha256(f.getvalue()).hexdigest()} for f in evidence_files]
    return data, documents, evidence, manifest


def render_review(project, show_findings=False):
    bundle = project.get('bundle')
    if not bundle:
        st.info("Generate draft findings after the audit. No saved study conclusions will be substituted.")
        return
    status_labels={}
    for finding in bundle['findings']:
        if show_findings:
            st.subheader(finding['finding_id'] + ' · ' + finding['title'])
            st.write(finding['statement'])
            st.caption('Basis: ' + finding['verification_method'])
            st.caption('Limits: ' + finding['limitations'])
        decision = project['finding_decisions'].get(finding['finding_id'])
        status_labels[finding['finding_id']]=st.empty()
        status_labels[finding['finding_id']].caption(finding['finding_id'] + ' · ' + ('Accepted' if decision and decision['decision']=='approve' else 'Rejected' if decision else 'Pending review'))
    with st.form('workflow_review_form_' + project['run_id']):
        chosen = st.selectbox('Finding to review', [f['finding_id'] for f in bundle['findings']],
            format_func=lambda key: next(f['title'] for f in bundle['findings'] if f['finding_id']==key))
        reviewer = st.text_input('Reviewer name')
        decision = st.selectbox('Review decision', ['Not decided', 'Accept', 'Reject'])
        reason = st.text_area('Review reason (English)')
        note = st.text_area('Researcher note (optional, English; kept separate from calculated findings)')
        save = st.form_submit_button('Save finding review')
    if save:
        try:
            decide_finding(project, chosen, {'Accept':'approve','Reject':'reject'}.get(decision), reviewer, reason, note)
            status_labels[chosen].caption(chosen+' · '+('Accepted' if decision=='Accept' else 'Rejected'))
            st.success('Review saved for this input version. It is not an independent evaluation label.')
        except AuditError as exc:st.error(str(exc))
    st.page_link('pages/03_ORCCA_Evidence_Completion.py', label='Review missing fields and source candidates')


def _render_legacy():
    st.title('ORCCA Project Workflow')
    st.caption('One browser session, versioned inputs, linked audit and review. No effect models or external LLM calls.')
    navigation()
    st.sidebar.page_link('app.py',label='ENVIO coding → ORCCA evidence')
    for file,label in [('01_ORCCA_Research_Audit.py','Data audit'),('02_ORCCA_Research_Findings.py','Research findings'),('03_ORCCA_Evidence_Completion.py','Evidence completion')]:
        st.sidebar.page_link('pages/'+file,label=label)
    private = private_uploads_allowed()
    if not private:st.info('Public practice: synthetic inputs only. Real research uploads and external generation remain disabled.')
    else:st.info('Private browser workspace: files remain in this server session, are not committed or sent to an external model. Download your work before closing the session.')
    st.write('1. Load inputs → 2. Inspect the audit → 3. Resolve evidence-backed missing fields → 4. Rerun and review findings → 5. Download the current version.')
    if st.button('Clear current web project',key='workflow_clear'):
        invalidate(st.session_state,'Project cleared. Load inputs to continue.')
        st.session_state.pop('workflow_input_files',None)
        st.session_state['workflow_upload_epoch'] = st.session_state.get('workflow_upload_epoch',0)+1
        st.rerun()
    tabs=st.tabs(['1 · Inputs','2 · Audit','3 · Completion','4 · Findings and review','5 · Export'])
    with tabs[0]:
        if private:
            st.write('Upload replacements here; no configuration-file edits are needed. Changing an input invalidates the earlier run and approvals immediately.')
            supplied={};epoch=str(st.session_state.get('workflow_upload_epoch',0))
            for role,label,types in [('panel','Panel',['csv','tsv','xls','xlsx']),('monthly','Monthly mortality',['csv','tsv','xls','xlsx']),('annual','Annual mortality',['csv','tsv','xls','xlsx']),
                ('tracker','Tidy tracker (optional)',['csv','tsv']),('raw_tracker','Original Tracker workbook (optional)',['xlsx']),('dictionary','REDCap dictionary (with original Tracker)',['xlsx']),
                ('results','Reported model results (optional; no refitting)',['csv','tsv']),('specs','Two source-linked specifications JSON (optional)',['json']),
                ('completion_evidence','Completion source excerpts CSV (optional)',['csv','tsv'])]:
                key='workflow_upload_'+role+'_'+epoch
                supplied[role]=st.file_uploader(label,type=types,key=key,on_change=update_input,args=(st.session_state,role,key))
                retained=st.session_state.get('workflow_input_files',{}).get(role,[])
                if retained:
                    st.caption('Retained in this session: '+retained[0]['name'])
                    if st.button('Remove retained '+label,key='workflow_remove_'+role+'_'+epoch):
                        st.session_state['workflow_input_files'].pop(role,None)
                        invalidate(st.session_state,'A source was removed. Rerun the audit with valid required inputs.')
                        st.session_state['workflow_upload_epoch']=int(epoch)+1
                        st.rerun()
            evidence_files=st.file_uploader('Evidence documents (optional)',type=['pdf','txt','md','docx','pptx'],accept_multiple_files=True,
                key='workflow_upload_documents_'+epoch,on_change=update_input,args=(st.session_state,'documents','workflow_upload_documents_'+epoch,True))
            if st.session_state.get('workflow_input_files',{}).get('documents'):
                st.caption('Retained evidence: '+', '.join(x['name'] for x in st.session_state['workflow_input_files']['documents']))
                if st.button('Remove retained evidence documents',key='workflow_remove_documents_'+epoch):
                    st.session_state['workflow_input_files'].pop('documents',None)
                    invalidate(st.session_state,'Evidence removed; rerun before reviewing findings.')
                    st.session_state['workflow_upload_epoch']=int(epoch)+1
                    st.rerun()
            st.caption('Completion CSV columns: county, month (event-applicable), menu, slot, version, text, source_ref. Narrative documents are searchable but do not acquire county/month labels automatically.')
            if st.button('Parse files and run audit',key='workflow_build'):
                try:
                    invalidate(st.session_state,'Parsing new inputs.')
                    retained,retained_docs=retained_inputs(st.session_state)
                    data,documents,evidence,manifest=parse_uploads(retained,retained_docs)
                    start_project(st.session_state,data,documents,evidence,manifest,private=True)
                    st.success('Inputs parsed and audit run. Continue to tab 2.')
                except Exception as exc:
                    invalidate(st.session_state,'Parsing failed. No previous findings are current.')
                    st.error(str(exc) if isinstance(exc,AuditError) else 'Input parsing failed; no demo or previous result was substituted.')
        else:
            variant=st.selectbox('Synthetic sample',['Baseline','Changed panel'],key='workflow_sample',on_change=invalidate,args=(st.session_state,))
            if st.button('Load sample and run audit',key='workflow_build'):
                start_project(st.session_state,*synthetic_project_inputs(variant));st.success('Synthetic project loaded. Continue to tab 2.')
        project=current_project(st.session_state,private)
        if project:
            project_header(project)
            with st.expander('Input manifest and hashes'):st.json(project['manifest'])
        elif st.session_state.get('workflow_active'):st.warning(st.session_state.get('workflow_invalid_reason','Run an audit first.'))
    if not project:
        for tab in tabs[1:]:
            with tab:st.info('Load valid inputs in tab 1. Earlier results are not current.')
        return
    with tabs[1]:
        project_header(project)
        for section,value in project['results'].items():
            st.subheader(section.replace('_',' ').title())
            if isinstance(value,dict):
                st.json(value.get('summary',value))
            else:st.dataframe(pd.DataFrame(value),hide_index=True)
        st.page_link('pages/01_ORCCA_Research_Audit.py',label='Open detailed audit with these inputs')
    with tabs[2]:
        st.page_link('app.py', label='Link an existing ENVIO coding result to this project')
        st.caption('Private workspace only: exact source quotations can support implementation candidates after explicit relevance review. ENVIO labels are not automatically ORCCA variables or missing-value replacements.')
        if project['data'].get('implementation_evidence'):
            st.dataframe(pd.DataFrame(project['data']['implementation_evidence'])[['source_file','envio_codes','county','applicable_month','scope_status','status']],hide_index=True)
        if 'tracker' in project['data']:
            issues=[]
            try:
                _,issues=identify_issues(project['data']['tracker'])
                st.write(f"{len(issues)} missing or unknown fields; {sum(x['eligible'] for x in issues)} eligible for source review.")
            except AuditError as exc:st.warning(str(exc))
            st.page_link('pages/03_ORCCA_Evidence_Completion.py',label='Open completion on this project')
            st.caption('Accepted source values can create a derived project version and rerun the audit. Statistical scenarios never change source-based audit values.')
            if private and project['documents'] and any(x['eligible'] for x in issues):
                with st.expander('Link an uploaded passage to a missing field'):
                    eligible={x['issue_id']:x for x in issues if x['eligible']}
                    target=st.selectbox('Target field',list(eligible),format_func=lambda k:' / '.join(str(eligible[k][f]) for f in ('county','month','menu','slot','version','field')))
                    docs={d['chunk_id']:d for d in project['documents']}
                    selected=st.selectbox('Source passage',list(docs),format_func=lambda k:f"{docs[k]['source_file']} · {docs[k].get('locator', 'page '+str(docs[k].get('page')))} · lines {docs[k]['line_start']}–{docs[k]['line_end']}")
                    st.code(docs[selected]['text'],language=None)
                    st.caption('Assign event-applicable scope only after checking the source. An interview date is not automatically an event month. Exact source text is required; do not add invented field assignments.')
                    with st.form('workflow_scope_form_'+project['run_id']):
                        quote=st.text_area('Exact supporting passage')
                        reviewer=st.text_input('Applicability reviewer')
                        reason=st.text_area('Why this passage applies to this county, month and strategy (English)')
                        save_scope=st.form_submit_button('Save scoped evidence and create new version')
                    if save_scope:
                        try:
                            from .workflow import add_scoped_evidence
                            add_scoped_evidence(st.session_state,project,target,selected,quote,reviewer,reason)
                            st.rerun()
                        except AuditError as exc:st.error(str(exc))
        else:st.info('Add a tracker before reviewing missing fields. No sample tracker is substituted.')
    with tabs[3]:
        if st.button('Generate draft findings from this audit',key='workflow_draft'):
            build_findings(project);st.success('Draft findings generated from current calculations. Human review is required.')
        render_review(project, show_findings=True)
        st.page_link('pages/02_ORCCA_Research_Findings.py',label='Open current findings and evidence search')
    with tabs[4]:
        if project.get('bundle'):
            approved=st.checkbox('Export accepted findings only',value=True,key='workflow_approved_only_'+project['run_id'])
            try:
                payload=export_project(project,approved)
                st.download_button('Download current project report',payload,'orcca_current_project.zip','application/zip',key='workflow_export')
                st.caption('Includes version manifest, audit tables, current derived tracker, source evidence and review ledger. A draft export is labelled with each finding’s review status.')
            except AuditError as exc:st.info(str(exc))
        else:st.info('Generate findings in tab 4 before exporting the project report.')


def _change_workspace():
    st.session_state['workflow_workspace']=st.session_state['workflow_workspace_widget']
    st.session_state.pop('workflow_input_files',None)
    st.session_state['workflow_upload_epoch']=st.session_state.get('workflow_upload_epoch',0)+1
    invalidate(st.session_state,'Workspace changed. Load inputs for this study; earlier findings are not current.')


def render():
    options=['H3 study preparation','Legacy data audit']
    saved=st.session_state.get('workflow_workspace',options[0])
    choice=st.radio('Workspace',options,index=options.index(saved),key='workflow_workspace_widget',on_change=_change_workspace)
    st.session_state['workflow_workspace']=choice
    if choice==options[0]:
        from .study_ui import render as study_render
        study_render()
    else:
        st.caption('Historical data and model-specification checks. H2a is not refitted; use H3 study preparation for the current research direction.')
        _render_legacy()
