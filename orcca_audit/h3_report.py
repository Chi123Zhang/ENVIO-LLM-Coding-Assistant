"""Prepare H3 inputs and produce a self-contained exploratory report."""
import hashlib
import html
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .h3_model import DRAWS, analyse

METHOD = (
    'Attendance two months earlier → active strategy intensity one month earlier → monthly fatal opioid overdose. '
    'The first path uses linear regression and the death count path uses interval Poisson regression with a population offset. '
    'Both use county fixed effects and a linear time trend. Intervals and p-values are approximate. '
    'The path product is an exploratory association. Results depend on assumptions about missing attendance.'
)


def read_csv(raw, columns):
    if len(raw) > 2_000_000:
        raise ValueError('Each input CSV must be under 2 MB.')
    data = pd.read_csv(BytesIO(raw))
    if not set(columns).issubset(data):
        raise ValueError('Required columns: ' + ', '.join(columns))
    return data[list(columns)].copy()


def prepare(panel_bytes, attendance_bytes):
    numeric = ['OEND_L1', 'MOUD_L1', 'population', 'y_lo', 'y_hi']
    panel = read_csv(panel_bytes, ['county', 'date', *numeric])
    attendance = read_csv(attendance_bytes, ['county', 'month', 'value'])
    if len(panel) > 500 or not len(panel):
        raise ValueError('The panel must contain 1 to 500 county-month records.')
    for frame in (panel, attendance):
        frame['county'] = frame.county.astype('string').str.strip()
        if frame.county.isna().any() or frame.county.eq('').any():
            raise ValueError('Every record requires a county.')
    panel['outcome_month'] = pd.to_datetime(panel.date, format='mixed', errors='raise').dt.to_period('M')
    if panel.outcome_month.isna().any() or panel.duplicated(['county', 'outcome_month']).any():
        raise ValueError('Panel county and month must be present and unique.')
    panel[numeric] = panel[numeric].apply(pd.to_numeric, errors='raise')
    lo, hi = panel.y_lo, panel.y_hi
    if (not np.isfinite(panel[numeric].to_numpy(dtype=float)).all()
        or (panel.population <= 0).any() or (panel[['OEND_L1', 'MOUD_L1']] < 0).any().any()
        or ((lo < 0) | (hi < lo) | (hi > 1000) | (lo % 1 != 0) | (hi % 1 != 0)).any()):
        raise ValueError('Check strategy counts, population and integer mortality bounds (0 to 1,000).')
    if not attendance.month.astype('string').str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', na=False).all():
        raise ValueError('Attendance month must use YYYY-MM.')
    attendance['attendance'] = pd.to_numeric(attendance.pop('value'), errors='raise')
    if not np.isfinite(attendance.attendance.dropna()).all() or attendance.attendance.lt(0).any():
        raise ValueError('Attendance must be nonnegative and finite when supplied.')
    if attendance.duplicated(['county', 'month']).any():
        raise ValueError('Provide one attendance value per county and month.')
    panel['month_index'] = panel.outcome_month.astype('int64')
    panel['attendance_month'] = (panel.outcome_month - 2).astype(str)
    panel = panel.merge(attendance.rename(columns={'month':'attendance_month'}),
        how='left', on=['county', 'attendance_month'], validate='one_to_one')
    if panel.county.nunique() < 4 or panel.county.nunique() > 12:
        raise ValueError('This report supports 4 to 12 counties.')
    observed = panel[panel.attendance.notna()]
    if len(observed) < 12 or observed.county.nunique() < 3 or observed.attendance.nunique() < 2:
        raise ValueError('Need at least 12 observed attendance values with variation across at least three counties.')
    return panel


def demo_inputs():
    rng = np.random.default_rng(83611)
    records, attendance = [], []
    for c in range(5):
        county = f'Demo county {c+1}'
        for j, month in enumerate(pd.period_range('2024-01', periods=12, freq='M')):
            pop = 75000 + 15000 * c
            oend, moud = rng.integers(0, 6, size=2)
            count = rng.poisson(3 + .4 * c + .15 * oend)
            lo, hi = (1, 9) if 1 <= count <= 9 and j % 3 == 0 else (count, count)
            records.append(dict(county=county, date=str(month.start_time.date()), population=pop,
                                OEND_L1=oend, MOUD_L1=moud, y_lo=lo, y_hi=hi))
            if c != 4 and (j+c) % 4:
                attendance.append(dict(county=county, month=str(month-2), value=int(rng.integers(5, 27))))
    return tuple(pd.DataFrame(rows).to_csv(index=False).encode() for rows in (records, attendance))


def create_report(panel_bytes, attendance_bytes, *, synthetic):
    panel = prepare(panel_bytes, attendance_bytes)
    results = analyse(panel)
    coverage = panel.assign(observed=panel.attendance.notna(), imputed=panel.attendance.isna()).groupby(
        'county', as_index=False).agg(county_months=('county','size'), observed=('observed','sum'),
                                    imputed=('imputed','sum'))
    info = dict(classification='synthetic demonstration' if synthetic else 'private research',
        county_months=len(panel), counties=len(coverage), observed=int(coverage.observed.sum()),
        imputed=int(coverage.imputed.sum()), interval_outcomes=int(panel.y_lo.ne(panel.y_hi).sum()),
        imputations_per_scenario=DRAWS, input_sha256=[hashlib.sha256(b).hexdigest()
        for b in (panel_bytes, attendance_bytes)], method=METHOD)
    reference = results.loc[results.scenario.eq('reference') & results.term.eq('path_product'),
                            ['menu','estimate','low','high','p_approx']]
    title = 'H3 analysis report'
    disclosure = ('The Erie scenarios apply only to a county named Erie. Input bounds and strategy fields '
                  'are used as supplied. Statistical imputation estimates missing attendance. '
                  'The analysis sends no records to an external service.')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 analysis report</title>'
            '<style>body{font:16px system-ui;max-width:1000px;margin:3rem auto;padding:0 1rem}'
            'th,td{padding:.35rem .6rem;text-align:left}table{border-collapse:collapse}</style>'
            f'<h1>{title}</h1><p>{html.escape(info["classification"].capitalize())}</p>'
            f'<p>{info["county_months"]} county-months across {info["counties"]} counties. '
            f'{info["observed"]} attendance values observed and {info["imputed"]} statistically imputed.</p>'
            f'<p>{html.escape(METHOD)}</p><p>{html.escape(disclosure)}</p>'
            '<h2>Reference scenario</h2>' + reference.to_html(index=False, float_format='%.4f') +
            '<h2>Attendance coverage</h2>' + coverage.to_html(index=False) + '</html>')
    buffer = BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('START_HERE.html', page)
        archive.writestr('model_results.csv', results.to_csv(index=False))
        archive.writestr('attendance_coverage.csv', coverage.assign(
            county=coverage.county.map(lambda value: "'" + value if value.startswith(('=', '+', '-', '@')) else value)
        ).to_csv(index=False))
        archive.writestr('run_metadata.json', json.dumps(info, indent=2))
    return info, reference, coverage, buffer.getvalue()
