"""Original-source reconstruction controls with private uploads and public fixtures."""

import hashlib
import json

import streamlit as st

from .engine import AuditError
from .h3_sources import demo_sources, rebuild, mortality

LABELS = {
    'menu': 'Strategy menu', 'attendance_available': 'Attendance available',
    'tracker_snapshot_available': 'Tracker snapshot available', 'outcome_available': 'Death and population available',
    'all_three_available': 'Complete active-strategy path', 'venue_path_available': 'Complete venue-breadth path',
}


@st.cache_data(show_spinner=False)
def synthetic_sources():
    return demo_sources()


def clear_sources():
    for key in ['h3_support_corpus','h3_support_search','h3_support_parse_memory',
                'h3_meeting_parse_memory','h3_query_memory','h3_generation_memory']:
        st.session_state.pop(key,None)
    st.session_state.pop('implementation_replay_result', None)
    st.session_state.pop('h3_site_path', None)
    st.session_state.pop('h3_site_estimates', None)
    st.session_state.pop('site_retrospective', None)
    st.session_state.pop('h3_source_report', None)
    st.session_state.pop('h3_source_analysis', None)
    st.session_state.pop('h3_attendance_evidence', None)
    st.session_state.pop('h3_attendance_comparison', None)
    st.session_state.pop('h3_no_change_comparison', None)
    st.session_state.pop('h3_full_demo_evidence', None)
    st.session_state.pop('h3_ai_experiment', None)
    st.session_state['h3_source_epoch'] = st.session_state.get('h3_source_epoch', 0) + 1


def render(private):
    st.subheader('Build H3 inputs from original sources')
    st.write('Start with three original tables: the Tracker, monthly CDC deaths and annual CDC deaths. '
             'Add the dictionary and attendance records to define and measure the selected H3 pathway. '
             'Meeting documents provide supporting evidence in a separate retrieval step.')
    st.markdown('**How the files are used**')
    st.table([
        {'Input group': '1 · Primary data', 'Files': 'Tracker + monthly CDC + annual CDC',
         'Use': 'Implementation records, monthly death counts, annual population and consistency checks'},
        {'Input group': '2 · Definitions and process data', 'Files': 'REDCap dictionary + attendance CSVs',
         'Use': 'Field and status definitions; recorded coalition attendance (X) for the current pathway'},
        {'Input group': '3 · RAG evidence', 'Files': 'Meeting PDFs for targeted attendance verification',
         'Use': 'Retrieve original passages and page references; review evidence before accepting a value'},
    ])
    st.caption('The three primary tables alone do not supply coalition attendance. Attendance CSVs are '
               'structured model inputs, not just retrieval documents. No H2a panel is required.')
    options = ['Synthetic source files', 'Private research files'] if private else ['Synthetic source files']
    route = st.radio('Source reconstruction data', options, key='h3_sources_route')
    synthetic = route == options[0]
    full_demo=False
    if synthetic:
        st.info('All files and values in this demonstration are synthetic.')
        scenario=st.selectbox('Synthetic source scenario',
            ['Small source-check example','Full H3 comparison demo'],key='h3_sources_demo_scenario')
        full_demo=scenario=='Full H3 comparison demo'
        if full_demo:
            from .h3_public_comparison import full_demo_sources, demo_input_archive
            files,attendance=full_demo_sources();start,end='2024-01','2024-12'
            st.caption('Six fictional counties, original-source reconstruction, a fixed model comparison, '
                       'BM25 evidence retrieval and report ZIP downloads. Synthetic estimates are software demonstrations.')
            st.download_button('Download full synthetic H3 inputs',demo_input_archive(),
                'synthetic_h3_comparison_inputs.zip','application/zip',key='h3_full_demo_inputs')
        else:
            files, attendance = synthetic_sources()
            start, end = '2024-03', '2024-05'
            st.caption('Example outcome window · March–May 2024. For the modelling and retrieval demonstration, '
                       'select Full H3 comparison demo above.')
    else:
        st.caption('Upload authorized files to this private server session.')
        epoch = str(st.session_state.get('h3_source_epoch', 0))
        from .implementation_replay import render as render_replay
        render_replay(epoch)
        files = {}
        st.subheader('1 · Primary data: three original tables')
        st.caption('Upload the original records here. Death counts and mortality rates are not interchangeable.')
        for role, label, types in [
            ('tracker', 'Original Tracker workbook', ['xlsx']),
            ('monthly', 'Original CDC monthly table', ['csv', 'tsv', 'xls', 'xlsx']),
            ('annual', 'Original CDC annual table', ['csv', 'tsv', 'xls', 'xlsx']),
        ]:
            file = st.file_uploader(label, type=types, key='h3_sources_' + role + epoch)
            if file:
                files[role] = (file.name, file.getvalue())
        st.subheader('2 · Definitions and coalition-process data')
        st.caption('The dictionary defines version-specific fields and status rules. Attendance records supply '
                   'the exposure for the attendance → implementation → death-count pathway. Both are required '
                   'by this reconstruction; meeting PDFs do not replace them automatically.')
        dictionary = st.file_uploader('REDCap dictionary', type=['xlsx'], key='h3_sources_dictionary' + epoch)
        if dictionary:
            files['dictionary'] = (dictionary.name, dictionary.getvalue())
        uploads = st.file_uploader('Original attendance records', type=['csv'], accept_multiple_files=True,
                                   key='h3_sources_attendance' + epoch)
        attendance = [(file.name, file.getvalue()) for file in uploads]
        st.caption('Next: rebuild the structured inputs. The separate RAG evidence upload appears below '
                   'after reconstruction, so retrieval targets the gaps in these files.')
        a, b = st.columns(2)
        start = a.text_input('Outcome start month (YYYY-MM)', '2021-02', key='h3_sources_start')
        end = b.text_input('Outcome end month (YYYY-MM)', '2022-06', key='h3_sources_end')
    if not synthetic and 'monthly' in files and st.checkbox(
            'Use the entire supplied CDC period', value=True, key='h3_sources_full_period'):
        try:
            table = mortality(files['monthly'], 'Month Code', r'\d{4}/(0[1-9]|1[0-2])', ['Deaths'])
            months = table['Month Code'].str.replace('/', '-', regex=False)
            if months.empty:
                raise AuditError('No valid CDC months found.')
            start, end = months.min(), months.max()
            st.caption(f'Full outcome window: {start} to {end}.')
        except AuditError as error:
            st.error(str(error))
            files.pop('monthly')
    ready = len(files) == 4 and bool(attendance)
    if not synthetic and not ready:
        required = {'tracker': 'Tracker', 'monthly': 'monthly CDC', 'annual': 'annual CDC',
                    'dictionary': 'REDCap dictionary'}
        missing = [label for role, label in required.items() if role not in files]
        if not attendance:
            missing.append('attendance CSVs')
        st.info('Before reconstruction, add: ' + ', '.join(missing) + '. Meeting PDFs are optional evidence, '
                'not a substitute for these inputs.')
    fingerprint = hashlib.sha256(json.dumps([
        route, full_demo, start, end,
        [(role, name, hashlib.sha256(raw).hexdigest()) for role, (name, raw) in sorted(files.items())],
        [(name, hashlib.sha256(raw).hexdigest()) for name, raw in attendance],
    ]).encode()).hexdigest()
    stored = st.session_state.get('h3_source_report')
    if stored and stored[0] != fingerprint:
        st.session_state.pop('h3_source_report', None)
        st.session_state.pop('h3_source_analysis', None)
        stored = None
    previous = st.session_state.get('h3_site_estimates')
    if previous and previous[0] != fingerprint:
        st.session_state.pop('h3_site_estimates')
        previous = None
    with st.expander('Estimate missing delivery-site counts'):
        st.write('Compare planned site counts with estimates from similar, earlier Tracker reports. '
                 'Accuracy is measured on recorded values held out by reporting month.')
        if st.button('Evaluate site estimates', disabled=not {'tracker','dictionary'} <= files.keys()):
            from .site_estimates import report
            try:
                previous = (fingerprint, report(files['tracker'], files['dictionary']))
                st.session_state['h3_site_estimates'] = previous
            except (ValueError, KeyError, AuditError) as error:
                previous = None
                st.session_state.pop('h3_site_estimates', None)
                st.error(str(error))
        if previous:
            stats, detail, download = previous[1]
            st.write(f"{stats['applicable_cells']} applicable cells · {stats['missing']} missing · "
                     f"{stats['estimated']} statistical estimates · {stats['still_unknown']} still unknown")
            if stats['applicable_cells']:
                st.dataframe(stats['metrics'], hide_index=True)
                plan = stats['plan_validation']
                st.write(f"Across all {plan['recorded_pairs']} available pairs, planned and delivering counts "
                         f"agreed in {plan['exact']}. The download includes {stats['full_delivery_scenarios']} "
                         "missing-cell scenarios assuming all planned sites delivered.")

                st.dataframe(detail[detail.status != 'recorded'], hide_index=True)
                st.caption('Donor cells identify the original records. These estimates are exploratory and stay outside the H3 models.')
            else:
                st.info('These inputs contain no eligible delivery-site fields with a matching dictionary definition.')
            st.download_button('Download site estimates and validation', download,
                               'delivery_site_validation.zip', 'application/zip')
    if private and not synthetic:
        from .retrospective_sites import render as render_sites
        render_sites(files, fingerprint)
    else:
        st.session_state.pop('site_retrospective', None)
    if st.button('Rebuild H3 from source files', disabled=not ready, key='h3_sources_run'):
        st.session_state.pop('h3_source_report', None)
        st.session_state.pop('h3_source_analysis', None)
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
        render_analysis(archive, fingerprint)
        if private and not synthetic:
            from .site_path import render as render_site_path
            render_site_path(archive, files, fingerprint)
            from .h3_no_change_ui import render_no_change
            render_no_change(archive,files['tracker'],files['dictionary'])
            from .h3_attendance_evidence_ui import render_evidence
            render_evidence(archive)
            from .h3_ai_completion_ui import render_ai_experiment
            render_ai_experiment(archive)
        elif full_demo:
            from .h3_no_change_ui import render_no_change
            render_no_change(archive,files['tracker'],files['dictionary'])
            from .h3_public_comparison import render_demo_evidence
            render_demo_evidence(archive)
        st.download_button('Download source reconstruction report', archive, 'orcca_h3_source_report.zip',
                           'application/zip', key='h3_sources_download')
    st.button('Clear source reconstruction', on_click=clear_sources, key='h3_sources_clear')


def render_analysis(archive, fingerprint):
    from .h3_source_analysis import analyse_reconstruction
    st.subheader('Test the observed H3 paths')
    st.caption('Attendance at t−2 → reported implementation at t−1 → CDC counts at t. '
               'OLS and interval Poisson models use county fixed effects and a linear time trend. '
               'Unknown inputs remain unknown. Approximate intervals use delete-county jackknife inference.')
    stored = st.session_state.get('h3_source_analysis')
    if stored and stored[0] != fingerprint:
        st.session_state.pop('h3_source_analysis', None)
        stored = None
    if st.button('Run H3 tests on current source data', key='h3_source_analysis_run'):
        st.session_state.pop('h3_source_analysis', None)
        stored = None
        try:
            with st.spinner('Fitting observed paths and checking each county…'):
                result = analyse_reconstruction(archive)
            stored = (fingerprint, result)
            st.session_state['h3_source_analysis'] = stored
        except (ValueError, ArithmeticError) as error:
            st.error('Analysis failed. ' + str(error))
    if stored:
        metadata, paths, diagnostics, report = stored[1]
        st.write(f"**{metadata['models_fitted']} of {metadata['specifications_attempted']} specifications fitted; "
                 f"{metadata['inference_available']} completed uncertainty checks.**")
        if not paths.empty:
            st.dataframe(paths[['menu','mediator','county_months','counties','estimate','low','high','p_approx']]
                         .rename(columns={'menu':'Menu','mediator':'Intermediate measure','county_months':'County-months',
                                          'counties':'Counties','estimate':'exp(a × b)','low':'95% lower',
                                          'high':'95% upper','p_approx':'Approximate p-value'}),hide_index=True)
        else:
            st.info('The current records do not support interval estimates. See the diagnostics for each specification.')
        with st.expander('All specifications and stability checks'):
            st.dataframe(diagnostics,hide_index=True)
        st.caption('These exploratory path tests depend on the selected measures and available records. '
                   'Small county counts and suppressed outcomes limit inference. CDC endpoint filters, service reach, '
                   'nonfatal outcomes and causal assumptions still require review.')
        st.download_button('Download H3 statistical report',report,'orcca_h3_statistical_report.zip',
                           'application/zip',key='h3_source_analysis_download')
