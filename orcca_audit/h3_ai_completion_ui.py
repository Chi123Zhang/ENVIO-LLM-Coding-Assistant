"""Private, explicitly unreviewed on-server RAG experiment; never a human approval."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import pandas as pd
import streamlit as st

from .engine import AuditError
from .h3_ai_completion import prepare_experiment, run_completion, compare_experiment, import_generation_memory, generation_key
from .h3_attendance_evidence import source_run_id
from .policy import private_uploads_allowed


def render_ai_experiment(archive):
    if not private_uploads_allowed():
        return
    st.subheader('Experimental RAG + local LLM completion')
    st.warning('AI-unreviewed experiment. Eligible candidates may enter a separate exploratory dataset '
               'without human confirmation. They are not verified measurements or confirmatory results.')
    st.caption('ENVIO BM25 retrieves same-county, same-month meeting excerpts. A fixed open-weight Qwen '
               'model runs on this application server CPU. Public model weights may download; research '
               'text is not sent to an external model API. Source memory holds original text only.')
    evidence = st.session_state.get('h3_attendance_evidence')
    if not evidence or evidence[1][3]['run_id'] != source_run_id(archive):
        st.info('First retrieve attendance evidence for the current source inputs above.')
        st.session_state.pop('h3_ai_experiment', None)
        return
    memory = evidence[1][3]
    baseline = archive
    no_change = st.session_state.get('h3_no_change_comparison')
    options = ['Observed source records']
    if no_change:
        options.insert(0, 'Documented no-change reconstruction')
    chosen = st.selectbox('Fixed baseline for the AI comparison', options, key='h3_ai_baseline')
    if chosen == 'Documented no-change reconstruction':
        with ZipFile(BytesIO(no_change[1][4])) as z:
            baseline = z.read('documented_no_change_analysis.zip')
    st.caption('This isolates attendance additions: baseline implementation rules, outcome, lags and '
               'statistical specification stay fixed. Evidence gaps remain unknown. Mechanical quote '
               'checks do not establish factual accuracy or complete meeting coverage.')
    prior = st.file_uploader('Previous RAG experiment ZIP (optional)',type=['zip'],
                             key='h3_prior_rag_memory')
    prior_bytes = prior.getvalue() if prior else b''
    st.caption('An earlier experiment can supply raw responses when model settings and exact source '
               'contexts match. Imported answers pass the current citation checks again.')
    permitted = st.checkbox('Use eligible AI candidates only in a separate unreviewed experiment',
                            value=False, key='h3_ai_opt_in')
    fingerprint = sha256(archive + baseline + memory['memory_sha256'].encode() + prior_bytes).hexdigest()
    stored = st.session_state.get('h3_ai_experiment')
    if stored and stored[0] != fingerprint:
        st.session_state.pop('h3_ai_experiment', None)
        stored = None
    if st.button('Run experimental RAG + local LLM completion', disabled=not permitted,
                 key='h3_ai_run'):
        st.session_state.pop('h3_ai_experiment', None)
        stored = None
        try:
            with st.spinner('Preparing source-bound retrieval and loading the fixed local model…'):
                prepared = prepare_experiment(archive, memory, baseline,
                    query_cache=st.session_state.setdefault('h3_query_memory',{}))
                answer_memory = st.session_state.setdefault('h3_generation_memory',{})
                if prior_bytes:
                    answer_memory.update(import_generation_memory(prior_bytes,archive,baseline))
                needs_generation = any(generation_key(t,prepared['protocol']) not in answer_memory
                                       for t in prepared['tasks'])
                if needs_generation:
                    from .h3_ai_runtime import generator
                    generate, model = generator()
                else:
                    generate = None
                    model = dict(status='previous_outputs_rechecked_or_no_matching_context')
            progress = st.progress(0.0)
            status = st.empty()
            def update(done, total, record):
                progress.progress(done / total)
                status.caption(f'Local generation {done}/{total}: {record["case_id"]}')
            with st.spinner('Generating AI candidates and fitting the unchanged exploratory models…'):
                result = run_completion(prepared, generate, model, progress=update,
                    output_cache=st.session_state.setdefault('h3_generation_memory',{}))
                comparison = compare_experiment(prepared, result)
            stored = fingerprint, result, comparison
            st.session_state['h3_ai_experiment'] = stored
            progress.empty()
            status.empty()
        except Exception as error:
            # A failed model never turns into a recorded or fabricated completion.
            message = str(error) if isinstance(error, (AuditError, ValueError)) else type(error).__name__
            st.error('AI experiment did not complete. No candidates were applied. ' + message)
    if stored:
        result, (summary, comparison, report) = stored[1], stored[2]
        st.write(f'**{summary["missing_county_months"]} attendance gaps; '
                 f'{summary["llm_calls"]} real local-model calls; '
                 f'{summary["candidate_county_months"]} AI candidate county-months; '
                 f'{summary["applied_to_experimental_county_months"]} experimental additions; '
                 '0 human confirmations.**')
        st.caption(f"Generation memory reused {summary.get('generation_memory_hits',0)} previous responses after checking the exact model, prompt and source context.")
        st.dataframe(pd.DataFrame(result['candidates'])[['case_id','ai_value','status','model_calls','reason']],
                     hide_index=True)
        st.dataframe(comparison, hide_index=True)
        with st.expander('Runtime, retrieval and output checks'):
            st.json(summary)
            st.dataframe(pd.DataFrame([dict(case_id=r['case_id'], meeting_date=r['meeting_date'],
                raw_response=r['raw_response'], automatic_checks=r['automatic_checks'],
                rejection_reason=r.get('rejection_reason','')) for r in result['model_outputs']]), hide_index=True)
        st.caption('Results are exploratory and unreviewed. Zero additions means no eligible source-supported '
                   'completion was produced, not that the system filled the gaps. There are no independent '
                   'labels here, so accuracy and time-saving benefits have not been measured.')
        st.download_button('Download AI-unreviewed RAG experiment', report,
            'orcca_h3_AI_UNREVIEWED_RAG_experiment.zip', 'application/zip', key='h3_ai_download')
        st.subheader('RAG and attendance interpolation')
        st.caption('Run six exploratory paths. Where dated meeting evidence was searched, the model '
                   'uses an eligible unreviewed RAG count first. Otherwise it uses observed '
                   'meetings in the same county within six months and generates 40 estimates. '
                   'Automatic selection compares interpolation and local donors on masked attendance. AI counts are treated as fixed. '
                   'Other gaps stay unknown.')
        estimator = st.selectbox('Attendance estimator',
            ['select_by_masked_error','interpolation','local_donors'],
            help='Choose using attendance prediction error, or run a named sensitivity method.')
        if st.button('Run H3 with RAG and interpolation', disabled=not permitted):
            from .h3_hybrid_attendance import run_hybrid
            result.pop('hybrid_attendance', None)
            with st.spinner('Comparing six H3 pathways…'):
                result['hybrid_attendance'] = (estimator,run_hybrid(baseline, result['candidates'],method=estimator))
        hybrid = result.get('hybrid_attendance')
        if permitted and hybrid and hybrid[0]==estimator:
            paths, audit, package = hybrid[1]
            st.dataframe(paths, hide_index=True)
            st.dataframe(audit, hide_index=True)
            st.download_button('Download hybrid H3 comparison', package, 'h3_hybrid.zip', 'application/zip')
