"""Exploratory complete-case H3 models from the current source reconstruction."""
from io import BytesIO
import html
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from scipy.stats import t

from .h3_model import fit
from .exports import safe_csv

CONFIG = {
    'attendance_lag_months': 2,
    'mediator_lag_months': 1,
    'mediators': ['active_snapshot', 'venue_types'],
    'menus': ['OEND', 'MOUD', 'Safer Prescribing'],
    'exposure_unit': '10 additional recorded attendees per observed meeting',
    'controls': 'County fixed effects and linear calendar trend',
    'first_equation': 'OLS',
    'second_equation': 'Interval Poisson likelihood with log(population / 100000) offset',
    'inference': 'Delete-one-county jackknife; approximate t reference on G-1 degrees of freedom',
    'missing_policy': 'Retain unknowns; complete-case estimation for each mediator and menu',
    'imputation': 'none',
    'scope': 'Exploratory attendance-implementation-CDC count paths; confounding and endpoint query unresolved',
}


def design(frame):
    controls = pd.get_dummies(frame.county, drop_first=True, dtype=float).to_numpy()
    time = (frame.month_index - frame.month_index.mean()) / 12
    first = np.column_stack([np.ones(len(frame)), time, controls, frame.attendance / 10])
    second = np.column_stack([first, frame.H3_L1])
    return first, second


def run_specification(data, menu, mediator):
    subset = data[data.menu.eq(menu)].copy()
    valid = subset.attendance.notna() & subset[mediator].notna() & subset.outcome_available
    frame = subset[valid].copy()
    frame['month_index'] = pd.PeriodIndex(frame.outcome_month, freq='M').astype('int64')
    frame['H3_L1'] = frame[mediator]
    estimates, failures = [], []
    if len(frame) < 4 or frame.county.nunique() < 3:
        return dict(menu=menu, mediator=mediator, county_months=len(frame), counties=frame.county.nunique(),
                    point_fit='unavailable', reason='Fewer than four complete records or three counties.'), estimates, failures, frame
    first, second = design(frame)
    check = dict(menu=menu, mediator=mediator, county_months=len(frame), counties=frame.county.nunique(),
                 first_parameters=first.shape[1], first_rank=np.linalg.matrix_rank(first),
                 outcome_parameters=second.shape[1], outcome_rank=np.linalg.matrix_rank(second),
                 suppressed_outcomes=int(frame.cdc_deaths.eq('Suppressed').sum()),
                 first_outcome_month=frame.outcome_month.min(), last_outcome_month=frame.outcome_month.max())
    estimates, failures = [], []
    try:
        point = fit(frame, menu='H3')
        check['point_fit'] = 'finite_fit'
        check.update(a=float(point[0]), b=float(point[1]), a_times_b=float(point[2]))
    except (ValueError, np.linalg.LinAlgError, ArithmeticError) as error:
        check.update(point_fit='unavailable', reason=str(error))
        return check, estimates, failures, frame
    for county in sorted(frame.county.unique()):
        try:
            values = fit(frame[frame.county.ne(county)], menu='H3')
            failures.append(dict(menu=menu, mediator=mediator, removed_county=county, status='fit',
                                 a=values[0], b=values[1], product=values[2]))
        except (ValueError, np.linalg.LinAlgError, ArithmeticError) as error:
            failures.append(dict(menu=menu, mediator=mediator, removed_county=county,
                                 status='unavailable', reason=str(error)))
    if any(row['status'] != 'fit' for row in failures):
        check['inference'] = 'Unavailable: at least one delete-county fit failed'
        return check, estimates, failures, frame
    if len(failures) < 3:
        check['inference'] = 'Unavailable: fewer than three county clusters'
        return check, estimates, failures, frame
    reps = np.array([[row['a'], row['b'], row['product']] for row in failures])
    g = len(reps)
    se = np.sqrt((g - 1) / g * ((reps - reps.mean(axis=0)) ** 2).sum(axis=0))
    margin = t.ppf(.975, g - 1) * se
    for i, term in enumerate(['attendance_to_implementation', 'implementation_to_CDC_count_rate', 'path_product']):
        transform = float if i == 0 else np.exp
        estimates.append(dict(menu=menu, mediator=mediator, term=term, county_months=len(frame), counties=g,
                              coefficient=float(point[i]), se_jackknife=float(se[i]),
                              estimate=float(transform(point[i])), low=float(transform(point[i] - margin[i])),
                              high=float(transform(point[i] + margin[i])),
                              p_approx=float(2 * t.sf(abs(point[i] / se[i]), g - 1)) if se[i] > 0 else None))
    check['inference'] = 'Approximate jackknife'
    return check, estimates, failures, frame

def analyse_reconstruction(source_archive):
    with ZipFile(BytesIO(source_archive)) as archive:
        data = pd.read_csv(archive.open('county_month.csv'), dtype={'fips': str})
        metadata = json.loads(archive.read('run_metadata.json'))
        source_files = {name: archive.read(name) for name in archive.namelist()}
    checks, results, replicates, included = [], [], [], []
    for menu in CONFIG['menus']:
        for mediator in CONFIG['mediators']:
            check, estimates, failures, frame = run_specification(data, menu, mediator)
            checks.append(check)
            results.extend(estimates)
            replicates.extend(failures)
            included.append(frame.assign(specification_mediator=mediator))
    diagnostics = pd.DataFrame(checks)
    estimates = pd.DataFrame(results, columns=[
        'menu', 'mediator', 'term', 'county_months', 'counties', 'coefficient', 'se_jackknife',
        'estimate', 'low', 'high', 'p_approx'])
    path_results = estimates[estimates.term.eq('path_product')].copy()
    metadata.update(
        models_fitted=int(diagnostics.point_fit.eq('finite_fit').sum()),
        specifications_attempted=len(checks),
        inference_available=int(sum(c.get('inference') == 'Approximate jackknife' for c in checks)),
        specification=(dict(CONFIG, input_completion='AI-unreviewed source-based candidates',
                            imputation='Experimental AI attendance additions; no human confirmation')
                       if 'ai_unreviewed_attendance_county_months' in metadata else CONFIG),
    )
    addition_label = ('AI-unreviewed experimental attendance additions' if
                      metadata.get('eligible_for_confirmatory_analysis') is False and
                      'ai_unreviewed_attendance_county_months' in metadata
                      else 'Evidence-reviewed additions')
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 source analysis</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:1rem;line-height:1.5}'
        'td,th{padding:.5rem;text-align:left}table{border-collapse:collapse}</style>'
        '<h1>H3 exploratory source analysis</h1><p>' + html.escape(metadata['classification']) + '</p>'
        f'<p>{metadata["county_months"]} county-months across {metadata["counties"]} counties. '
        f'{metadata["models_fitted"]} of {len(checks)} specifications produced finite point estimates. '
        f'{metadata["inference_available"]} completed all delete-county fits.</p>'
        '<p>Attendance at t−2 → implementation at t−1 → CDC death counts at t. '
        f'Unknown values remain unknown. {addition_label}: {metadata.get("imputed_values", 0)}.</p>'
        f'<p>Dictionary-supported rule-derived mediator cells: {metadata.get("rule_reconstructed_mediator_values", 0)}. '
        'These are conditional state reconstructions, not human-adjudicated attendance additions.</p>'
        '<h2>Exploratory path tests</h2>' + path_results.to_html(index=False, float_format=lambda x: f'{x:.4f}')
        + '<p>The null is a × b = 0, equivalent to exp(a × b) = 1. The exposure unit is ten additional recorded '
        'attendees per meeting. Intervals and p-values use approximate delete-county jackknife inference. '
        'Small county counts, sparse observed records and suppressed outcomes limit inference. '
        'P-values are exploratory and unadjusted.</p>'
        '<h2>All attempted specifications</h2>' + diagnostics.to_html(index=False)
        + '<p>Confirm the CDC query filters, questionnaire timing, strategy coverage, attendance definitions '
        'and confounder strategy. Complete H3 also requires verified service reach and nonfatal outcomes. '
        f'Approved attendance county-month additions: {metadata.get("approved_attendance_county_months", 0)}.</p>'
        '<p>source_reconstruction.html and tracker_source_cells.csv preserve source evidence. '
        'county_month.csv retains every candidate row. model_rows.csv shows only included records. '
        'model_results.csv contains regression links and path products. delete_county_fits.csv records failures.</p>'
        '</html>'
    )
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, content in source_files.items():
            if name == 'START_HERE.html':
                archive.writestr('source_reconstruction.html', content)
            elif name != 'run_metadata.json':
                archive.writestr(name, content)
        archive.writestr('START_HERE.html', page)
        archive.writestr('run_metadata.json', json.dumps(metadata, indent=2))
        archive.writestr('model_results.csv', safe_csv(estimates))
        archive.writestr('model_diagnostics.csv', safe_csv(diagnostics))
        archive.writestr('delete_county_fits.csv', safe_csv(replicates))
        archive.writestr('model_rows.csv', safe_csv(pd.concat(included)))
    return metadata, path_results, diagnostics, output.getvalue()
