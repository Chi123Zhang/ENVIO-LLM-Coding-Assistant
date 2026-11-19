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
from .h3_attendance_validation import blocked_validation, interpolate, neighbours


def run_hybrid(archive, candidates, *, method="interpolation"):
    with ZipFile(BytesIO(archive)) as src:
        data = pd.read_csv(src.open('county_month.csv'))
    before = analyse_reconstruction(source_only_archive(archive))[1]
    keys = ['county', 'attendance_month']
    observed = data[keys + ['attendance']].drop_duplicates().dropna()
    blocked_predictions, blocked_scores, blocked_protocol = blocked_validation(observed)
    requested_method = method
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
    if method == 'select_by_blocked_validation':
        method = blocked_protocol['selected_method']
    if method not in {'interpolation','local_donors','time_weighted_donors'}:
        raise ValueError('Unsupported attendance estimator.')
    eligible = data[data.attendance.isna() & data.outcome_available &
                    data[CONFIG['mediators']].notna().any(axis=1)]
    relevant = set(eligible[keys].itertuples(index=False, name=None))
    for row in sorted(candidates, key=lambda r: (r['county'], r['attendance_month'])):
        county, month = key = (row['county'], row['attendance_month'])
        if key not in relevant or not row['supplied_meetings']:
            continue
        obs = observed[observed.county.eq(county)].set_index('attendance_month').attendance.sort_index()
        ref, distances = neighbours(obs, month)
        value = row['ai_value']
        basis = 'Unknown'
        if row['status'] == 'AI_unreviewed_candidate' and pd.notna(value):
            sampled[key] = np.full(40, value)
            basis = 'Source-linked AI estimate (unreviewed)'
            ref = ref.iloc[:0]
        elif row.get('evidence_assessments',row['model_calls']) > 0:
            if method in {'local_donors','time_weighted_donors'} and len(ref)>=3:
                weights = np.ones(len(ref)) if method == 'local_donors' else 1 / distances
                weights = weights / weights.sum()
                value = float(np.average(ref, weights=weights))
                sampled[key] = np.array([rng.choice(ref.values,p=rng.dirichlet(weights * len(ref))) for _ in range(40)])
                basis = 'Statistical local donors' if method == 'local_donors' else 'Statistical time-weighted donors'
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
        draws = sampled.get(key)
        ledger.append(dict(county=county, attendance_month=month, basis=basis,
            estimate=value if draws is not None else None,
            estimator='fixed_unreviewed_AI' if basis.startswith('Source-linked AI') else method,
            supplied_meetings=row['supplied_meetings'], donor_count=len(ref),
            donor_months=' | '.join(ref.index),
            uses_later_donors=bool(any(m > month for m in ref.index)),
            draw_lower=float(np.quantile(draws,.025)) if draws is not None else None,
            draw_upper=float(np.quantile(draws,.975)) if draws is not None else None))
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
    audit = pd.DataFrame(ledger, columns=keys+['basis','estimate','estimator','supplied_meetings',
        'donor_count','donor_months','uses_later_donors','draw_lower','draw_upper'])
    buf = BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED) as z:
        z.writestr('paths.csv', paths.to_csv(index=False))
        from .h3_results_ui import statistical_summary
        z.writestr('results_summary.csv', statistical_summary(paths).to_csv(index=False))
        z.writestr('masked_validation.csv',validation.to_csv(index=False))
        z.writestr('blocked_validation_predictions.csv',blocked_predictions.to_csv(index=False))
        z.writestr('blocked_validation_scores.csv',blocked_scores.to_csv(index=False))
        z.writestr('protocol.json',json.dumps(dict(method=method,requested_method=requested_method,
            blocked_validation=blocked_protocol,masked_mae=scores,
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
                   'Exploratory sensitivity analysis. Statistical attendance uses 40 draws. '
                   'Read blocked_validation_scores.csv for separate method-selection and held-out test errors. '
                   'Attendance draw ranges are descriptive, not calibrated confidence intervals. '
                   'AI counts are unreviewed and treated as fixed; extraction uncertainty is not calibrated. '
                   'Interpolation uses meetings before and after the gap and is retrospective. '
                   'Six menu-mediator paths are reported together. Complete causal H3 is unresolved.')
    return paths, audit, buf.getvalue()
