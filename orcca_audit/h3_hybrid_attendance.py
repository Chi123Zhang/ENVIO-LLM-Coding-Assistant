"""Exploratory H3 attendance sensitivity using the current session's RAG candidates."""
from io import BytesIO
from hashlib import sha256
import json
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


def run_hybrid(archive, candidates, *, method="interpolation"):
    with ZipFile(BytesIO(archive)) as src:
        data = pd.read_csv(src.open('county_month.csv'))
    before = analyse_reconstruction(source_only_archive(archive))[1]
    keys = ['county', 'attendance_month']
    observed = data[keys + ['attendance']].drop_duplicates().dropna()
    rng = np.random.default_rng(20261120)
    sampled, ledger = {}, []
    validation = []
    for county, group in observed.groupby('county'):
        obs = group.set_index('attendance_month').attendance.sort_index()
        for held, actual in obs.items():
            distances = np.abs(pd.PeriodIndex(obs.index,freq='M').asi8-pd.Period(held,freq='M').ordinal)
            ref = obs[(distances>0)&(distances<=6)]
            predicted = interpolate(obs,held)
            if len(ref)>=3 and predicted is not None:
                validation.append(dict(county=county,month=held,observed=actual,
                                       interpolation=predicted,local_donors=ref.mean()))
    validation = pd.DataFrame(validation,columns=['county','month','observed','interpolation','local_donors'])
    scores = {name:float((validation[name]-validation.observed).abs().mean())
              for name in ['interpolation','local_donors']} if len(validation) else {}
    if method == 'select_by_masked_error':
        method = min(scores,key=scores.get) if scores else 'interpolation'
    if method not in {'interpolation','local_donors'}:
        raise ValueError('Unsupported attendance estimator.')
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
        elif row.get('evidence_assessments',row['model_calls']) > 0:
            distances = np.abs(pd.PeriodIndex(obs.index,freq='M').asi8-pd.Period(month,freq='M').ordinal)
            ref = obs[(distances>0)&(distances<=6)]
            if method == 'local_donors' and len(ref)>=3:
                value = float(ref.mean())
                sampled[key] = np.array([rng.choice(ref.values,p=rng.dirichlet(np.ones(len(ref)))) for _ in range(40)])
                basis = 'Statistical local donors'
            else:
                value = interpolate(obs, month) if method=='interpolation' else None
            if method == 'interpolation' and value is not None:
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
        from .h3_results_ui import statistical_summary
        z.writestr('results_summary.csv', statistical_summary(paths).to_csv(index=False))
        z.writestr('masked_validation.csv',validation.to_csv(index=False))
        z.writestr('protocol.json',json.dumps(dict(method=method,masked_mae=scores,
            validation_rows=len(validation),seed=20261120,draws=40,
            source_archive_sha256=sha256(archive).hexdigest(),
            method_selection='Attendance prediction error; mortality and p-values not used',
            source_candidates=candidates),indent=2))
        z.writestr('draws.csv',pd.DataFrame([
            dict(county=k[0],attendance_month=k[1],draw=i,value=v)
            for k, values in sampled.items() for i,v in enumerate(values)]).to_csv(index=False))
        z.writestr('attendance_basis.csv', audit.to_csv(index=False))
        z.writestr('README.txt','Open results_summary.csv for approximate p-values, 95% intervals and significance at p < 0.05. '
                   'paths.csv retains all raw model columns and fit diagnostics. '
                   'Exploratory sensitivity analysis. Statistical attendance uses 40 masked-error draws. '
                   'AI counts are unreviewed and treated as fixed; extraction uncertainty is not calibrated. '
                   'Interpolation uses meetings before and after the gap and is retrospective. '
                   'Six menu-mediator paths are reported together. Complete causal H3 is unresolved.')
    return paths, audit, buf.getvalue()
