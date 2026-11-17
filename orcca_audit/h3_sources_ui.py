"""Original-source reconstruction controls with private uploads and public fixtures."""

import hashlib
import json

import streamlit as st

from .engine import AuditError
from .h3_sources import demo_sources, rebuild

LABELS = {
    'menu': 'Strategy menu', 'attendance_available': 'Attendance available',
    'tracker_snapshot_available': 'Tracker snapshot available', 'outcome_available': 'Death and population available',
    'all_three_available': 'Complete active-strategy path', 'venue_path_available': 'Complete venue-breadth path',
}


@st.cache_data(show_spinner=False)
def synthetic_sources():
    return demo_sources()


def clear_sources():
    st.session_state.pop('h3_source_report', None)
    st.session_state['h3_source_epoch'] = st.session_state.get('h3_source_epoch', 0) + 1


def render(private):
    st.subheader('Build H3 inputs from original sources')
    st.write('Start with the original Tracker, its REDCap dictionary, monthly and annual CDC tables, '
             'and meeting attendance CSVs. The report shows the source coverage for each part of the pathway.')
    options = ['Synthetic source files', 'Private research files'] if private else ['Synthetic source files']
    route = st.radio('Source reconstruction data', options, key='h3_sources_route')
    synthetic = route == options[0]
    if synthetic:
        st.info('All files and values in this demonstration are synthetic.')
        files, attendance = synthetic_sources()
        start, end = '2024-03', '2024-05'
        st.caption('Example outcome window · March–May 2024. Attendance precedes the outcome by two months; '
                   'the Tracker snapshot precedes it by one month.')
    else:
        st.caption('Upload authorized files to this private server session.')
        epoch = str(st.session_state.get('h3_source_epoch', 0))
        files = {}
        for role, label, types in [
            ('tracker', 'Original Tracker workbook', ['xlsx']),
            ('dictionary', 'REDCap dictionary', ['xlsx']),
            ('monthly', 'Original CDC monthly table', ['csv', 'tsv', 'xls', 'xlsx']),
            ('annual', 'Original CDC annual table', ['csv', 'tsv', 'xls', 'xlsx']),
        ]:
            file = st.file_uploader(label, type=types, key='h3_sources_' + role + epoch)
            if file:
                files[role] = (file.name, file.getvalue())
        uploads = st.file_uploader('Original attendance records', type=['csv'], accept_multiple_files=True,
                                   key='h3_sources_attendance' + epoch)
        attendance = [(file.name, file.getvalue()) for file in uploads]
        a, b = st.columns(2)
        start = a.text_input('Outcome start month (YYYY-MM)', '2021-02', key='h3_sources_start')
        end = b.text_input('Outcome end month (YYYY-MM)', '2022-06', key='h3_sources_end')
    ready = len(files) == 4 and bool(attendance)
    fingerprint = hashlib.sha256(json.dumps([
        route, start, end,
        [(role, name, hashlib.sha256(raw).hexdigest()) for role, (name, raw) in sorted(files.items())],
        [(name, hashlib.sha256(raw).hexdigest()) for name, raw in attendance],
    ]).encode()).hexdigest()
    stored = st.session_state.get('h3_source_report')
    if stored and stored[0] != fingerprint:
        st.session_state.pop('h3_source_report', None)
        stored = None
    if st.button('Rebuild H3 from source files', disabled=not ready, key='h3_sources_run'):
        st.session_state.pop('h3_source_report', None)
        stored = None
        try:
            with st.spinner('Checking definitions and aligning source records…'):
                result = rebuild(files, attendance, start, end, synthetic=synthetic)
            stored = (fingerprint, result)
            st.session_state['h3_source_report'] = stored
        except (AuditError, ValueError, KeyError, TypeError) as error:
            st.error('Source reconstruction failed. ' + str(error))
    if stored:
        metadata, coverage, archive = stored[1]
        st.write(f"**{metadata['county_months']} county-months across {metadata['counties']} counties.**")
        st.dataframe(coverage.rename(columns=LABELS), hide_index=True, width='stretch')
        st.caption('Coverage is counted separately for each strategy menu. Unreported Tracker months, unresolved status '
                   'and missing attendance remain unknown. Only explicit present marks establish recorded attendance.')
        st.write('The rebuilt inputs and source references are ready for review. Model fitting and statistical imputation '
                 'have not run in this source-reconstruction report.')
        st.download_button('Download source reconstruction report', archive, 'orcca_h3_source_report.zip',
                           'application/zip', key='h3_sources_download')
    st.button('Clear source reconstruction', on_click=clear_sources, key='h3_sources_clear')
