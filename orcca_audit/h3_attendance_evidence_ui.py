"""Private browser controls for evidence retrieval and independently reviewed additions."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import streamlit as st

from .engine import AuditError
from .exports import safe_csv
from .h3_attendance_evidence import build_evidence_review, apply_attendance_reviews
from .h3_source_analysis import analyse_reconstruction


def compare_approved(archive, memory, reviews):
    data, ledger, derived = apply_attendance_reviews(archive, memory, reviews)
    if not ledger:
        raise AuditError('No approved human reviews were supplied. No model comparison was run.')
    before = analyse_reconstruction(archive)
    after = analyse_reconstruction(derived)
    keys = ['menu', 'mediator', 'term']
    comparison = before[1].merge(after[1], on=keys, how='outer', suffixes=('_original', '_reviewed'))
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as z:
        z.writestr('original_analysis.zip', before[3])
        z.writestr('reviewed_analysis.zip', after[3])
        z.writestr('path_comparison.csv', safe_csv(comparison))
        z.writestr('cell_changes.json', json.dumps(ledger, indent=2))
    return comparison, output.getvalue()


def render_evidence(archive):
    st.subheader('Find evidence for missing attendance')
    st.markdown('**3 · RAG evidence: targeted attendance verification**')
    st.write('The primary tables, dictionary and attendance CSVs have already been read above. '
             'This upload adds supporting documents only; it does not replace the structured inputs.')
    st.caption('This step accepts meeting PDFs. Interviews and other documents can be searched in the '
               'project evidence workflow, but are not automatically converted to attendance values here.')
    st.caption('County and date in each filename restrict retrieval to the '
               'missing exposure month. BM25 retrieves original excerpts and preserves page references. '
               'A retrieved passage is not a verified headcount. No external model is called.')
    st.caption('Source memory retains original excerpts, locators and retrieval logs. It does not train a '
               'Transformer or generate missing values. Accepted corrections require an explicit review.')
    epoch = str(st.session_state.get('h3_source_epoch', 0))
    uploads = st.file_uploader('Meeting PDFs for attendance verification', type=['pdf'],
                               accept_multiple_files=True, key='h3_attendance_pdfs_' + epoch)
    files = [(f.name, f.getvalue()) for f in uploads]
    fingerprint = sha256(archive + json.dumps(sorted(
        (name, sha256(raw).hexdigest()) for name, raw in files)).encode()).hexdigest()
    stored = st.session_state.get('h3_attendance_evidence')
    if stored and stored[0] != fingerprint:
        st.session_state.pop('h3_attendance_evidence', None)
        st.session_state.pop('h3_attendance_comparison', None)
        stored = None
    if st.button('Retrieve attendance evidence for current gaps', disabled=not files,
                 key='h3_attendance_retrieve'):
        st.session_state.pop('h3_attendance_evidence', None)
        st.session_state.pop('h3_attendance_comparison', None)
        stored = None
        try:
            with st.spinner('Reading PDFs and retrieving county-month evidence…'):
                result = build_evidence_review(archive, files)
            stored = fingerprint, result
            st.session_state['h3_attendance_evidence'] = stored
        except (AuditError, ValueError, KeyError, TypeError) as error:
            st.error('Evidence retrieval failed. ' + str(error))
    if not stored:
        return
    manifest, queue, logs, memory, report = stored[1]
    st.write(f"**{manifest['missing_attendance_cases']} missing county-months; "
             f"{manifest['cases_with_same_county_month_pdf']} have matching PDF evidence. "
             'No values have been filled.**')
    st.dataframe(queue[['case_id', 'potential_menu_paths', 'source_pdfs', 'retrieved_chunks',
                         'retrieval_status', 'human_confirmed']], hide_index=True)
    matches = [r for r in logs if r['citations']]
    if matches:
        case_id = st.selectbox('Inspect retrieved source evidence', [r['case_id'] for r in matches],
                               key='h3_attendance_evidence_case_' + fingerprint[:12])
        result = next(r for r in matches if r['case_id'] == case_id)
        for doc in result['citations']:
            st.markdown(f"**{doc['source_file']} — page {doc['page']}**")
            st.text(doc['text'])
            st.caption('Source ID: ' + doc['chunk_id'])
    st.download_button('Download attendance review package', report, 'orcca_attendance_evidence_review.zip',
                        'application/zip', key='h3_attendance_review_download')
    st.caption('The independent reviewer answer columns are blank. AI outputs must never be entered as human '
               'approvals. The package keeps original-source memory separate from decisions.')
    upload = st.file_uploader('Completed independent attendance review CSV (optional)', type=['csv'],
                               key='h3_attendance_decisions_' + fingerprint[:12])
    review_raw = upload.getvalue() if upload else b''
    review_fingerprint = sha256(fingerprint.encode() + review_raw).hexdigest()
    previous = st.session_state.get('h3_attendance_comparison')
    if previous and previous[0] != review_fingerprint:
        st.session_state.pop('h3_attendance_comparison', None)
        previous = None
    if st.button('Compare reviewed additions with the original data', disabled=not review_raw,
                 key='h3_attendance_compare'):
        st.session_state.pop('h3_attendance_comparison', None)
        previous = None
        try:
            reviews = pd.read_csv(BytesIO(review_raw), dtype=str, keep_default_na=False)
            with st.spinner('Checking review decisions and fitting the same specification…'):
                result = compare_approved(archive, memory, reviews)
            previous = review_fingerprint, result
            st.session_state['h3_attendance_comparison'] = previous
        except (AuditError, ValueError, KeyError, TypeError, ArithmeticError) as error:
            st.error('Comparison was not completed. ' + str(error))
    if previous:
        comparison, report = previous[1]
        st.dataframe(comparison, hide_index=True)
        st.caption('Both fits use the same exploratory specification. Differences reflect reviewed input changes, '
                   'not proof of improved accuracy or causal identification. Read the fit diagnostics for both datasets.')
        st.download_button('Download original and reviewed comparison', report, 'orcca_h3_reviewed_comparison.zip',
                            'application/zip', key='h3_attendance_comparison_download')
