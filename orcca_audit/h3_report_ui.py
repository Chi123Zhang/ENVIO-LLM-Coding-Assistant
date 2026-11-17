"""Streamlit controls for running and downloading the H3 analysis."""
import hashlib

import streamlit as st

from .h3_report import METHOD, create_report, demo_inputs


def render(private):
    st.subheader('Run H3 analysis')
    st.caption(METHOD)
    mode = st.radio('Analysis data', ['Synthetic demo', 'Private CSV inputs'] if private else ['Synthetic demo'],
                    horizontal=True, key='h3_model_mode')
    synthetic = mode == 'Synthetic demo'
    if synthetic:
        st.info('The demo uses generated data. Run it to see the model results and download an example report.')
        inputs = demo_inputs()
    else:
        st.write('Upload the full mortality and strategy panel and the county-month attendance table. '
                 'The report shows how many attendance values were observed or statistically imputed.')
        panel = st.file_uploader('County-month panel CSV', type='csv', key='h3_model_panel')
        attendance = st.file_uploader('Attendance CSV', type='csv', key='h3_model_attendance')
        inputs = (panel.getvalue(), attendance.getvalue()) if panel and attendance else None
    fingerprint = (mode, *(hashlib.sha256(raw).hexdigest() for raw in inputs)) if inputs else None
    stored = st.session_state.get('h3_model_report')
    if stored and stored[0] != fingerprint:
        st.session_state.pop('h3_model_report', None)
        stored = None
    if st.button('Run full-panel H3 analysis', disabled=inputs is None, key='h3_fit'):
        st.session_state.pop('h3_model_report', None)
        stored = None
        try:
            with st.spinner('Fitting models and generating the report…'):
                stored = (fingerprint, create_report(*inputs, synthetic=synthetic))
                st.session_state['h3_model_report'] = stored
        except (ValueError, ArithmeticError) as error:
            st.error(str(error))
    if stored:
        info, reference, coverage, archive = stored[1]
        st.write(f"**{info['county_months']} county-months across {info['counties']} counties.** "
                 f"{info['observed']} attendance values observed and {info['imputed']} statistically imputed.")
        st.markdown('**Indirect path summary · reference imputation scenario**')
        st.dataframe(reference.rename(columns={'menu':'Strategy', 'estimate':'Ratio', 'low':'95% CI lower',
                     'high':'95% CI upper', 'p_approx':'Approximate p-value'}), hide_index=True)
        with st.expander('Attendance coverage by county'):
            st.dataframe(coverage, hide_index=True)
        st.download_button('Download H3 analysis report', archive, 'orcca_h3_analysis_report.zip',
                           'application/zip', key='h3_download')
