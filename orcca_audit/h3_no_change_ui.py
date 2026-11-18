"""Private browser comparison for dictionary-supported no-change reconstruction."""
from hashlib import sha256
from io import BytesIO
import html
import json
from zipfile import ZipFile, ZIP_DEFLATED

import streamlit as st

from .engine import AuditError
from .exports import safe_csv
from .h3_no_change import reconstruct_no_change
from .h3_source_analysis import analyse_reconstruction


def compare_no_change(archive, tracker, dictionary):
    metadata, coverage, ledger, audit, derived = reconstruct_no_change(archive, tracker, dictionary)
    original = analyse_reconstruction(archive)
    completed = analyse_reconstruction(derived)
    metadata['source_reconstruction_models_fitted'] = metadata.pop('models_fitted', 0)
    metadata['analysis_summary'] = {
        'original': {key: original[0][key] for key in
                     ['models_fitted', 'specifications_attempted', 'inference_available']},
        'rule_derived': {key: completed[0][key] for key in
                         ['models_fitted', 'specifications_attempted', 'inference_available']},
    }
    comparison = original[1].merge(completed[1], on=['menu','mediator','term'], how='outer',
                                    suffixes=('_original','_rule_derived'))
    output = BytesIO()
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 no-change comparison</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;line-height:1.6}'
            'td,th{padding:.5rem;text-align:left}</style><h1>H3 documented no-change comparison</h1>'
            f'<p>{html.escape(metadata["classification"])}</p>'
            f'<p>Rule-derived mediator cells: {len(ledger)}. Original observed values changed: 0. '
            'Attendance values filled: 0. Human labels created: 0.</p>'
            '<p>Only explicit, dictionary-defined no-change reports with an approved plan and known '
            'immediately preceding state are used. No unresolved listed strategy, initial unknown, '
            'missing month or conflict is filled. These are rule-derived sensitivity inputs, '
            'not independently adjudicated truth or an accuracy evaluation.</p><h2>Compared path tests</h2>'
            + comparison.to_html(index=False, float_format=lambda x:f'{x:.4f}')
            + '<p>Same exploratory attendance at t−2, implementation at t−1 and CDC counts at t '
            'specification in both datasets. County effects, linear trend, population offset and '
            'approximate delete-county jackknife inference are unchanged. No result-driven model selection.</p>'
            '<p>Complete H3 causal mediation is not established. County coverage, endpoint query filters, '
            'reach/nonfatal outcomes and the causal confounder design require review. '
            'Synthetic examples have no research interpretation. Nonsignificance is not proof of no effect.</p>'
            '<p>Read the full diagnostics in each nested analysis ZIP and the source-linked reconstruction '
            'ledger before interpreting the estimates. Original workbooks remain unchanged.</p></html>')
    with ZipFile(output,'w',ZIP_DEFLATED) as z:
        z.writestr('START_HERE.html',page)
        z.writestr('original_analysis.zip',original[3])
        z.writestr('documented_no_change_analysis.zip',completed[3])
        z.writestr('path_comparison.csv',safe_csv(comparison))
        z.writestr('reconstruction_ledger.csv',safe_csv(ledger))
        z.writestr('rule_audit.csv',safe_csv(audit))
        z.writestr('run_metadata.json',json.dumps(metadata,indent=2))
    return metadata, coverage, completed[1], comparison, output.getvalue()


def render_no_change(archive, tracker, dictionary):
    st.subheader('Compare documented no-change reconstruction')
    st.caption('ORCB and menu-specific ORCB1–3 distinguish an explicit no-change report from an unknown. '
               'This secondary analysis carries a known previous-month implementation snapshot only when '
               'the original response and version-specific dictionary support it. It does not fill attendance.')
    fingerprint=sha256(archive+tracker[1]+dictionary[1]).hexdigest()
    stored=st.session_state.get('h3_no_change_comparison')
    if stored and stored[0]!=fingerprint:
        st.session_state.pop('h3_no_change_comparison',None)
        stored=None
    if st.button('Run documented no-change comparison',key='h3_no_change_compare'):
        st.session_state.pop('h3_no_change_comparison',None)
        stored=None
        try:
            with st.spinner('Checking dictionary rules and fitting the unchanged specifications…'):
                result=compare_no_change(archive,tracker,dictionary)
            stored=fingerprint,result
            st.session_state['h3_no_change_comparison']=stored
        except (AuditError,ValueError,KeyError,TypeError,ArithmeticError) as error:
            st.error('No-change comparison failed. '+str(error))
    if stored:
        metadata,coverage,paths,comparison,report=stored[1]
        st.write(f"**{metadata['rule_reconstructed_mediator_values']} rule-derived mediator cells; "
                 '0 original recorded values changed; 0 attendance values filled.**')
        st.dataframe(paths[['menu','mediator','county_months','counties','estimate','low','high','p_approx']]
            .rename(columns={'menu':'Menu','mediator':'Intermediate measure','county_months':'County-months',
                             'counties':'Counties','estimate':'exp(a × b)','low':'95% lower',
                             'high':'95% upper','p_approx':'Approximate p-value'}),hide_index=True)
        with st.expander('Original versus rule-derived estimates'):
            st.dataframe(comparison,hide_index=True)
        st.caption('Interpret these as conditional exploratory paths. A no-change reconstruction is not '
                   'independent validation or proof of an AI benefit. Read all fit diagnostics. '
                   'Complete causal H3 remains unresolved.')
        st.download_button('Download documented no-change H3 comparison',report,
            'orcca_h3_documented_no_change_comparison.zip','application/zip',key='h3_no_change_download')
