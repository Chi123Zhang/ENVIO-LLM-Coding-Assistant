"""Read-only public presentation of a real local run on fictional evidence."""
import streamlit as st
from .engine import AuditError
from .model_review import replay, load_record, export_record, validate_reply, rules_baseline


def render(project):
    with st.expander('Local Qwen review · recorded synthetic demonstration'):
        st.write('See how a local language model reads retrieved evidence and either proposes a value or leaves it unknown. These three fictional examples are separate from your current project and from the 24-case masking pilot.')
        st.info('Recorded local generation, not live cloud inference. The button reruns ENVIO BM25 and checks the saved model response against the same inputs. No model API is called.')
        st.caption('All examples and source passages here are invented. This recording contains no Tracker records, meeting transcripts or human answers. It does not update your project or Findings.')
        if st.button('Replay synthetic Qwen review', key='model_demo_replay'):
            try:
                st.session_state['model_demo_result'] = (project['run_id'], replay())
            except (AuditError, OSError, ValueError, KeyError):
                st.session_state.pop('model_demo_result', None)
                st.error('Recording unavailable or changed. No substitute response was generated.')
        cached = st.session_state.get('model_demo_result')
        if not cached: return
        if cached[0] != project['run_id']:
            st.info('Project changed. Replay the separate synthetic demonstration again.'); return
        try:
            record = cached[1]
            if record['fixture_sha256'] != load_record()['fixture_sha256']:
                raise AuditError('Inputs changed.')
            package = export_record(record)
        except (AuditError, OSError, ValueError, KeyError):
            st.error('Saved recording is no longer current. Replay after restoring the bundled inputs.'); return
        a, b, c = st.columns(3)
        a.metric('BM25 calls on replay', record['actual_bm25_calls_this_session'])
        b.metric('Recorded model responses', len(record['runs']))
        c.metric('Independent human labels', 0)
        chosen = st.selectbox('Synthetic model case', [r['case_id'] for r in record['runs']], key='model_demo_case')
        run = next(r for r in record['runs'] if r['case_id'] == chosen)
        st.write('**Question:** ' + run['payload']['question'])
        st.write('**Model status:** ' + run['output']['status'].replace('_', ' '))
        st.write(run['output']['answer'])
        st.write('**Proposed month:** ' + (run['output']['proposed_start_month'] or 'None'))
        baseline = rules_baseline(run)
        st.write('**Exact-rule source check:** ' + (baseline['month'] or 'Unknown'))
        if baseline['month'] and not run['output']['proposed_start_month']:
            st.warning('The fictional source explicitly records a start date, but this model response retained unknown. This run demonstrates a model limitation, not accurate recovery.')
        for flag in validate_reply(run['output'], chosen, run['payload']['evidence']): st.warning(flag)
        st.write('**Sources supplied to Qwen**')
        byid = {e['chunk_id']: e for e in run['output']['evidence']}
        for d in run['payload']['evidence']:
            st.caption(d['chunk_id'] + ' · ' + d['source_file'] + ' · ' + d['locator'])
            st.code(d['text'], language=None)
            if d['chunk_id'] in byid: st.write('Model quote: ' + byid[d['chunk_id']]['quote'])
        for limit in run['output']['limitations']: st.caption('Model limitation: ' + limit)
        st.warning('A matching quote does not prove the interpretation. A candidate remains unconfirmed; no value is written to your data.')
        with st.expander('Model settings and run provenance'):
            st.json(record['provenance'])
            st.caption('Source memory contains only the six fictional source lines. Model responses are never indexed as evidence. There is no weight training or automatic memory update.')
        st.download_button('Download synthetic Qwen run', package, 'orcca_synthetic_qwen_review.zip', 'application/zip', key='model_demo_download')
