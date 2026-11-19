"""Exploratory H3 path through fixed-site delivery per applicable strategy slot."""
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
import json

import pandas as pd

from .exports import safe_csv
from .h3_source_analysis import run_specification
from .retrospective_sites import report as estimate_sites


def compare(source_zip, site_zip):
    with ZipFile(BytesIO(source_zip)) as z:
        panel = pd.read_csv(z.open('county_month.csv'), dtype={'fips': str})
        inputs = json.loads(z.read('run_metadata.json'))['inputs']
    with ZipFile(BytesIO(site_zip)) as z:
        sites = pd.read_csv(z.open('site_review.csv'))
        metadata = json.loads(z.read('summary.json'))
    for role in ('tracker', 'dictionary'):
        if [i['sha256'] for i in inputs if i['role'] == role] != [metadata[role + '_sha256']]:
            raise ValueError('The source and site reports use different input versions.')
    if sites.cell.duplicated().any():
        raise ValueError('Duplicate site cells in the estimate report.')
    keys = ['county', 'mediator_month', 'menu']
    sites = sites.rename(columns={'month': 'mediator_month'})
    checks, estimates, used = [], [], []
    for scenario, column in [('Observed', 'observed'), ('Retrospective Qwen estimates', 'derived_site_count')]:
        grouped = sites.groupby(keys)[column].agg(['mean', 'count', 'size'])
        # Keep the denominator constant within a report. Unrecorded slots are never zeros.
        grouped['sites_per_slot'] = grouped['mean'].where(grouped['size'] == grouped['count'])
        data = panel.merge(grouped[['sites_per_slot']], on=keys, how='left', validate='one_to_one')
        for menu in ('OEND', 'MOUD'):
            check, terms, _, frame = run_specification(data, menu, 'sites_per_slot')
            subset = data[data.menu.eq(menu)]
            known = subset.sites_per_slot.notna()
            checks.append(dict(scenario=scenario, known_mediator_months=int(known.sum()),
                               with_attendance=int((known & subset.attendance.notna()).sum()), **check))
            estimates.extend(dict(scenario=scenario, **term) for term in terms)
            used.append(frame.assign(scenario=scenario))
    checks = pd.DataFrame(checks)
    terms = pd.DataFrame(estimates, columns=["scenario", "menu", "mediator", "term", "county_months", "counties", "coefficient", "se_jackknife", "estimate", "low", "high", "p_approx"])
    paths = terms[terms.term.eq('path_product')] if len(terms) else terms
    archive = BytesIO()
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        z.writestr('site_validation.zip', site_zip)
        z.writestr('inputs.json', json.dumps(inputs, indent=2))
        z.writestr('path_results.csv', safe_csv(terms))
        z.writestr('diagnostics.csv', safe_csv(checks))
        z.writestr('model_rows.csv', safe_csv(pd.concat(used)))
        z.writestr('README.txt', 'Exploratory H3 sensitivity analysis. Attendance two months earlier, mean delivering fixed sites per applicable reported slot one month earlier, then CDC count. Only complete slot groups qualify. Sites may overlap across strategies. The retrospective scenario can use later reports and carries unconfirmed estimates. Intervals are conditional on the completed inputs and omit imputation uncertainty. Missing intervals mean the stability check failed. See site_validation.zip for observed values and cited donors.')
    return checks, paths, archive.getvalue()


def render(source_zip, files, fingerprint):
    import streamlit as st
    st.subheader('Test the H3 pathway through fixed service sites')
    st.caption('Compare observed reports with retrospective estimates from linked Tracker records and on-server Qwen. The mediator is the mean number of delivering sites per applicable strategy slot. Each included report needs a value for every applicable slot. Later reports can inform estimates, so interpret the comparison as exploratory.')
    key = 'h3_site_path'
    if st.session_state.get(key, (None,))[0] != fingerprint:
        st.session_state.pop(key, None)
    if st.button('Run H3 with site-count completion'):
        st.session_state.pop(key, None)
        try:
            with st.spinner('Retrieving site counts and fitting the two scenarios…'):
                cached = st.session_state.get('site_retrospective')
                if not cached or cached[0] != fingerprint:
                    cached = (fingerprint, estimate_sites(files['tracker'], files['dictionary']))
                    st.session_state['site_retrospective'] = cached
                st.session_state[key] = (fingerprint, compare(source_zip, cached[1][1]))
        except Exception as error:
            st.error(f'Site pathway could not finish ({type(error).__name__}). Check source inputs and model availability.')
    if key in st.session_state:
        checks, paths, archive = st.session_state[key][1]
        st.dataframe(checks, hide_index=True)
        if not paths.empty:
            st.table(paths[['scenario', 'menu', 'county_months', 'counties', 'estimate', 'low', 'high', 'p_approx']])
        else:
            st.info('No pathway completed the leave-one-county uncertainty check.')
        st.download_button('Download H3 site pathway comparison', archive, 'h3_site_path.zip', 'application/zip')
