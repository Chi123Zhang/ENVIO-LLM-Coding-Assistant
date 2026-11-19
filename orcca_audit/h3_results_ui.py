"""Readable exploratory path results shared by both H3 sensitivity routes."""
import pandas as pd


def statistical_summary(paths):
    data = paths.reindex(columns=['menu', 'mediator', 'county_months', 'estimate',
                                 'low', 'high', 'p_approx']).copy()
    p = pd.to_numeric(data.pop('p_approx'), errors='coerce')
    valid = p.between(0, 1)
    data['Approx. p-value'] = p.where(valid)
    data['Significant (p < 0.05)'] = [
        ('Yes' if value < .05 else 'No') if ok else 'Unavailable'
        for value, ok in zip(p, valid)]
    data['mediator'] = data['mediator'].replace({
        'active_snapshot': 'Active strategies', 'venue_types': 'Venue types'})
    return data.rename(columns={'menu': 'Strategy', 'mediator': 'Intermediate measure',
        'county_months': 'County-months', 'estimate': 'Indirect-path ratio',
        'low': '95% interval: lower', 'high': '95% interval: upper'})


def render_path_results(paths, *, key):
    import streamlit as st
    from .exports import safe_csv
    summary = statistical_summary(paths)
    p = summary['Approx. p-value'].dropna()
    count = int((p < .05).sum())
    st.markdown(f'**Statistically significant paths (p < 0.05): {count} of {len(p)} evaluable paths.**')
    if len(p):
        st.write(f'Minimum approximate p-value: {p.min():.4f}.')
    if len(p) < len(summary):
        st.warning(f'{len(summary) - len(p)} paths have unavailable p-values; see fit diagnostics.')
    st.dataframe(summary, hide_index=True, column_config={
        'Approx. p-value': st.column_config.NumberColumn(format='%.4f')})
    st.caption('Approximate, unadjusted p-values for exploratory indirect paths. '
               'A p-value below 0.05 alone does not establish causal H3 mediation. '
               'Estimated attendance remains distinct from observed attendance.')
    st.download_button('Download H3 p-values and significance (CSV)', safe_csv(summary),
        'h3_p_values_and_significance.csv', 'text/csv', key=key + '_statistics')
    with st.expander('Full model results and fit diagnostics'):
        st.dataframe(paths, hide_index=True)
