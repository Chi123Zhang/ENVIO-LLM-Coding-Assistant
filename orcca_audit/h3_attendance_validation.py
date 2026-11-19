"""Retrospective attendance estimators with a frozen, blocked evaluation split.

Only county, month and observed attendance enter this module. Outer test blocks
are removed before inner method selection. Both tasks mask two calendar months
at once; this is a reconstruction benchmark, not a prospective forecast.
"""
import numpy as np
import pandas as pd

METHODS = ('interpolation', 'local_donors', 'time_weighted_donors')
PROTOCOL_VERSION = 'attendance-blocked-validation-1'


def neighbours(observed, month):
    distances = np.abs(pd.PeriodIndex(observed.index, freq='M').asi8 -
                       pd.Period(month, freq='M').ordinal)
    selected = (distances > 0) & (distances <= 6)
    return observed[selected], distances[selected]


def interpolate(observed, month):
    observed = observed.sort_index()
    delta = pd.PeriodIndex(observed.index, freq='M').asi8 - pd.Period(month, freq='M').ordinal
    left = np.flatnonzero((delta < 0) & (delta >= -6))
    right = np.flatnonzero((delta > 0) & (delta <= 6))
    if not len(left) or not len(right):
        return None
    a, b = left[-1], right[0]
    return float(np.interp(0, delta[[a, b]], observed.iloc[[a, b]]))


def predictions(observed, month):
    ref, distances = neighbours(observed, month)
    return dict(interpolation=interpolate(observed, month),
        local_donors=float(ref.mean()) if len(ref) >= 3 else None,
        time_weighted_donors=float(np.average(ref, weights=1 / distances)) if len(ref) >= 3 else None)


def blocked_validation(observed):
    rows = []
    for county, group in observed.groupby('county', sort=True):
        obs = group.set_index('attendance_month').attendance.sort_index()
        blocks = pd.PeriodIndex(obs.index, freq='M').asi8 // 2
        # Every fifth fixed two-month block is outer test data, independent of
        # values, missingness patterns, implementation, outcomes and p-values.
        test = blocks % 5 == 0
        training = obs[~test]
        for i, (month, actual) in enumerate(obs.items()):
            inner_blocks = pd.PeriodIndex(training.index, freq='M').asi8 // 2
            reference = training if test[i] else training[inner_blocks != blocks[i]]
            predicted = predictions(reference, month)
            refs, _ = neighbours(reference, month)
            rows.append(dict(county=county, month=month, observed=float(actual),
                split='held_out_test' if test[i] else 'method_selection',
                block=int(blocks[i]), donor_months=' | '.join(refs.index), **predicted))
    columns = ['county', 'month', 'observed', 'split', 'block', 'donor_months', *METHODS]
    frame = pd.DataFrame(rows, columns=columns)
    scores = []
    for split, group in frame.groupby('split', sort=True):
        common = group.dropna(subset=list(METHODS))
        for method in METHODS:
            errors = common[method] - common.observed
            county_mae = common.assign(error=errors.abs()).groupby('county').error.mean()
            scores.append(dict(split=split, method=method, evaluated_rows=len(common),
                candidate_rows=len(group), counties=common.county.nunique(),
                mae=float(errors.abs().mean()), rmse=float(np.sqrt((errors**2).mean())),
                bias=float(errors.mean()), county_balanced_mae=float(county_mae.mean())))
    metrics = pd.DataFrame(scores, columns=['split','method','evaluated_rows','candidate_rows',
        'counties','mae','rmse','bias','county_balanced_mae'])
    selection = metrics[metrics.split.eq('method_selection')].dropna(subset=['county_balanced_mae'])
    chosen = (selection.sort_values(['county_balanced_mae','method']).iloc[0]['method']
              if len(selection) else 'local_donors')
    protocol = dict(version=PROTOCOL_VERSION, selected_method=chosen,
        selection_metric='method_selection county_balanced_mae on common eligible rows',
        selection_status='selected_on_masked_training' if len(selection) else 'predeclared_fallback_no_validation',
        split='Two-month calendar blocks; block ordinal modulo 5 == 0 held out; test values excluded from all method-selection donors',
        scope='Same county, within six months; retrospective, may use later meetings',
        warning='Known-record masking does not establish accuracy on truly missing attendance. '
                'Draw ranges are not calibrated confidence intervals. Mortality and p-values are not used.')
    return frame, metrics, protocol


def render_validation(package, *, key):
    from io import BytesIO
    from zipfile import ZipFile
    import json
    import streamlit as st
    from .exports import safe_csv
    with ZipFile(BytesIO(package)) as z:
        protocol = json.loads(z.read('protocol.json'))
        st.write('**Attendance estimator used:** ' + protocol['method'])
        if 'blocked_validation_scores.csv' not in z.namelist():
            return
        metrics = pd.read_csv(z.open('blocked_validation_scores.csv'))
    with st.expander('Attendance estimation validation', expanded=True):
        st.caption('Prediction errors are in people. Methods share the same evaluable records. '
                   'Held-out test attendance is excluded from method selection and its donor records. '
                   'After selection, the final reconstruction uses all observed attendance.')
        view = metrics[['split','method','evaluated_rows','candidate_rows','mae','rmse','county_balanced_mae']].copy()
        view['Records evaluated'] = [f'{a} / {b}' for a, b in zip(view.pop('evaluated_rows'),view.pop('candidate_rows'))]
        view['split'] = view['split'].replace({'held_out_test':'Held-out test','method_selection':'Method selection'})
        view['method'] = view['method'].replace({'interpolation':'Interpolation',
            'local_donors':'Local donors','time_weighted_donors':'Time-weighted donors'})
        view = view.rename(columns={'split':'Split','method':'Estimator','mae':'MAE (people)',
            'rmse':'RMSE (people)','county_balanced_mae':'County-balanced MAE'})
        st.dataframe(view, hide_index=True)
        if protocol['blocked_validation']['selection_status'] != 'selected_on_masked_training':
            st.warning('Insufficient training validation: the declared local-donor fallback was used.')
        st.caption(protocol['blocked_validation']['warning'])
        st.download_button('Download attendance validation scores (CSV)',safe_csv(view),
            'h3_attendance_validation.csv','text/csv',key=key + '_validation')
