"""H3 exploratory path models and multiple-imputation sensitivity analysis."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, root
from scipy.special import gammaln, logsumexp
from scipy.stats import t

SEED = 20261117
DRAWS = 40
SCENARIOS = {
    'reference': (1., 1.),
    'all_missing_25pct_lower': (.75, .75),
    'all_missing_25pct_higher': (1.25, 1.25),
    'erie_50pct_lower': (1., .5),
    'erie_50pct_higher': (1., 1.5),
}


def polish_minimum(objective, x0, **options):
    result = minimize(objective, x0, **options)
    if np.isfinite(result.x).all() and np.linalg.norm(result.jac, np.inf) > 1e-5:
        # The objective can reach machine precision before the score does.
        polished = root(lambda beta: objective(beta)[1], result.x)
        value, score = objective(polished.x)
        if value <= result.fun + 1e-9 and np.linalg.norm(score, np.inf) < np.linalg.norm(result.jac, np.inf):
            result.x, result.fun, result.jac = polished.x, value, score
    return result


def attendance_draws(panel):
    missing = panel.attendance.isna().to_numpy()
    continuous = pd.DataFrame({
        'time': panel.month_index, 'oend': panel.OEND_L1, 'moud': panel.MOUD_L1,
        'population': np.log(panel.population), 'mortality_lower': np.log1p(panel.y_lo),
        'mortality_upper': np.log1p(panel.y_hi),
    })
    continuous = (continuous - continuous.mean()) / continuous.std().replace(0, 1)
    county = pd.get_dummies(panel.county, dtype=float)
    design = np.column_stack([np.ones(len(panel)), continuous, county])
    assert np.isfinite(design).all()
    observed_design = design[~missing]
    y = np.log1p(panel.attendance.to_numpy()[~missing])
    # Conjugate Gaussian model for log1p attendance. The common prior for county
    # intercepts supplies a distribution for Erie, which has no observed values.
    penalties = np.r_[0.001, np.repeat(1., continuous.shape[1]), np.repeat(4., county.shape[1])]
    covariance = np.linalg.inv(observed_design.T @ observed_design + np.diag(penalties))
    posterior_mean = covariance @ observed_design.T @ y
    residual_scale = 1. + .5 * (y @ y - posterior_mean @ (observed_design.T @ y))
    shape = 2. + len(y) / 2
    rng = np.random.default_rng(SEED)
    results = []
    for _ in range(DRAWS):
        variance = residual_scale / rng.gamma(shape)
        coeffs = rng.multivariate_normal(posterior_mean, variance * covariance)
        predicted = design[missing] @ coeffs + rng.normal(0, np.sqrt(variance), missing.sum())
        results.append(np.maximum(0., np.expm1(predicted)))
    return missing, results


def fit_with_variance(frame, menu):
    counties = frame.county.unique()
    point = fit(frame, menu)
    replicates = np.array([fit(frame[frame.county != c], menu) for c in counties])
    variance = (len(counties) - 1) / len(counties) * ((replicates - replicates.mean(axis=0)) ** 2).sum(axis=0)
    return point, variance


def pool(fits, scenario, menu, n, g):
    estimates, variances = (np.array(data) for data in zip(*fits, strict=True))
    center = estimates.mean(axis=0)
    within = variances.mean(axis=0)
    between = estimates.var(axis=0, ddof=1)
    total = within + (1 + 1 / len(fits)) * between
    rows = []
    for j, term in enumerate(['attendance_to_strategy_count', 'strategy_count_to_death_rate', 'path_product']):
        # Limit reference degrees of freedom to the number of counties minus one.
        relative = (1 + 1 / len(fits)) * between[j] / total[j] if total[j] else 0.
        df = min(g - 1, (len(fits) - 1) / relative**2) if relative else g - 1
        se = np.sqrt(total[j])
        half_width = t.ppf(.975, df) * se
        transform = np.exp if j else float
        rows.append(dict(scenario=scenario, menu=menu, term=term, county_months=n, counties=g,
            estimate=float(transform(center[j])), low=float(transform(center[j] - half_width)),
            high=float(transform(center[j] + half_width)),
            p_approx=float(2 * t.sf(abs(center[j] / se), df)) if se else None,
            within_imputation_variance=float(within[j]), between_imputation_variance=float(between[j])))
    return rows


def fit(data, menu='OEND'):
    controls = pd.get_dummies(data.county, drop_first=True, dtype=float).to_numpy()
    base = np.column_stack([np.ones(len(data)), (data.month_index - data.month_index.mean()) / 12, controls])
    exposure = data.attendance.to_numpy() / 10
    mediator = data[f'{menu}_L1'].to_numpy()
    first = np.column_stack([base, exposure])
    second = np.column_stack([first, mediator])
    if any(np.linalg.matrix_rank(x) != x.shape[1] or len(data) <= x.shape[1] for x in [first, second]):
        raise ValueError('Insufficient within-county/time variation.')
    a = np.linalg.lstsq(first, mediator, rcond=None)[0][-1]
    lo = data.y_lo.to_numpy(dtype=int)
    hi = data.y_hi.to_numpy(dtype=int)
    counts = np.arange(hi.max() + 1)
    valid = (counts >= lo[:, None]) & (counts <= hi[:, None])
    offset = np.log(data.population.to_numpy() / 100_000)

    def objective(beta):
        eta = second @ beta + offset
        mu = np.exp(eta)
        scores = np.where(valid, eta[:, None] * counts - mu[:, None] - gammaln(counts + 1), -np.inf)
        logprob = logsumexp(scores, axis=1)
        expected = np.exp(scores - logprob[:, None]) @ counts
        return -logprob.sum(), second.T @ (mu - expected)

    opt = polish_minimum(objective, np.zeros(second.shape[1]), jac=True, method='BFGS', options={'gtol': 1e-7})
    if not np.isfinite(opt.x).all() or np.max(np.abs(opt.jac)) > 1e-5 or np.max(np.abs(opt.x)) > 20:
        raise ValueError('Count model did not converge to a stable finite solution.')
    b = opt.x[-1]
    return np.array([a, b, a * b])


def analyse(panel, draws=DRAWS):
    missing, replacements = attendance_draws(panel)
    rows = []
    for scenario, (other_factor, erie_factor) in SCENARIOS.items():
        factor = np.where(panel.loc[missing, 'county'].str.casefold().eq('erie'), erie_factor, other_factor)
        fits = {menu: [] for menu in ('OEND', 'MOUD')}
        for replacement in replacements[:draws]:
            completed = panel.copy()
            completed.loc[missing, 'attendance'] = replacement * factor
            for menu in fits:
                fits[menu].append(fit_with_variance(completed, menu))
        for menu, values in fits.items():
            rows.extend(pool(values, scenario, menu, len(panel), panel.county.nunique()))
    return pd.DataFrame(rows)


