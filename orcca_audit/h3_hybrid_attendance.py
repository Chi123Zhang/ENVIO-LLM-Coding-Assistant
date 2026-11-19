"""Exploratory H3 attendance sensitivity using the current session's RAG candidates."""
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from .h3_ai_completion import source_only_archive
from .h3_model import fit_with_variance, pool
from .h3_source_analysis import CONFIG, analyse_reconstruction


def interpolate(observed, month):
    months = pd.PeriodIndex(observed.index, freq='M')
    delta = months.asi8 - pd.Period(month, freq='M').ordinal
    left = np.flatnonzero((delta < 0) & (delta >= -6))
    right = np.flatnonzero((delta > 0) & (delta <= 6))
    if not len(left) or not len(right):
        return None
    a, b = left[-1], right[0]
    return float(np.interp(0, delta[[a, b]], observed.iloc[[a, b]]))


def run_hybrid(archive, candidates):
    with ZipFile(BytesIO(archive)) as src:
        data = pd.read_csv(src.open('county_month.csv'))
    before = analyse_reconstruction(source_only_archive(archive))[1]
    keys = ['county', 'attendance_month']
    observed = data[keys + ['attendance']].drop_duplicates().dropna()
    rng = np.random.default_rng(20261120)
    sampled, ledger = {}, []
    eligible = data[data.attendance.isna() & data.outcome_available &
                    data[CONFIG['mediators']].notna().any(axis=1)]
    relevant = set(eligible[keys].itertuples(index=False, name=None))
    for row in sorted(candidates, key=lambda r: (r['county'], r['attendance_month'])):
        county, month = key = (row['county'], row['attendance_month'])
        if key not in relevant or not row['supplied_meetings']:
            continue
        obs = observed[observed.county.eq(county)].set_index('attendance_month').attendance.sort_index()
        value = row['ai_value']
        basis = 'Unknown'
        if row['status'] == 'AI_unreviewed_candidate' and pd.notna(value):
            sampled[key] = np.full(40, value)
            basis = 'Source-linked AI estimate (unreviewed)'
        elif row['model_calls'] > 0:
            value = interpolate(obs, month)
            if value is not None:
                errors = []
                for held, actual in obs.items():
                    prediction = interpolate(obs, held)
                    if prediction is not None:
                        errors.append(actual - prediction)
                if len(errors) >= 3:
                    sampled[key] = np.maximum(0, value + rng.choice(errors, size=40))
                    basis = 'Statistical interpolation'
        ledger.append(dict(county=county, attendance_month=month, basis=basis,
                           estimate=value if key in sampled else None))
    results = []
    for menu in CONFIG['menus']:
        for mediator in CONFIG['mediators']:
            frame = data[data.menu.eq(menu) & data[mediator].notna() & data.outcome_available].copy()
            frame['month_index'] = pd.PeriodIndex(frame.outcome_month, freq='M').asi8
            frame['H3_L1'] = frame[mediator]
            fits = []
            diagnostic = ''
            n = g = 0
            for i in range(40):
                additions = [sampled[k][i] if k in sampled else np.nan
                             for k in frame[keys].itertuples(index=False, name=None)]
                current = frame.assign(attendance=frame.attendance.fillna(pd.Series(additions,index=frame.index))).dropna(subset=['attendance'])
                n, g = len(current), current.county.nunique()
                if n < 4 or g < 3:
                    diagnostic = "Insufficient complete rows or counties"
                    break
                try:
                    fits.append(fit_with_variance(current, 'H3'))
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    diagnostic = str(exc)
                    break
            result = pool(fits, 'Hybrid', menu, n, g)[-1] if len(fits) == 40 else {}
            results.append(dict(result, menu=menu, mediator=mediator, county_months=n, counties=g, diagnostic=diagnostic))
    paths = pd.DataFrame(results).merge(before[['menu','mediator','county_months','p_approx']].rename(
        columns={'county_months':'baseline_county_months','p_approx':'baseline_p'}), how='left', on=['menu','mediator'])
    audit = pd.DataFrame(ledger, columns=keys+['basis','estimate'])
    buf = BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED) as z:
        z.writestr('paths.csv', paths.to_csv(index=False))
        z.writestr('attendance_basis.csv', audit.to_csv(index=False))
    return paths, audit, buf.getvalue()
