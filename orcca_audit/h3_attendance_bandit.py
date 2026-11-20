"""Experimental one-step RL on masked, recorded attendance; no outcome reward.

Disjoint LinUCB with explicit epsilon exploration. Each training county-month
provides only the chosen action's feedback, once. This is a retrospective
historical-data environment, not online clinical learning or LLM fine-tuning.
"""
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from .h3_attendance_validation import METHODS, neighbours, predictions

VERSION = 'h3-attendance-bandit-1'
FEATURES = ('intercept', 'reference_count', 'left_distance', 'right_distance',
            'relative_variation', 'relative_change')
ALPHA = .25
EPSILON = .10
REWARD_CAP = 50.
SEED = 20261120


def context(observed, month):
    ref, distance = neighbours(observed, month)
    delta = pd.PeriodIndex(ref.index, freq='M').asi8 - pd.Period(month, freq='M').ordinal
    left, right = ref[delta < 0], ref[delta > 0]
    scale = 1 + abs(float(ref.mean())) if len(ref) else 1.
    return np.array([1., min(len(ref), 12) / 12,
        min(abs(delta[delta < 0]).min(), 6) / 6 if len(left) else 1.,
        min(delta[delta > 0].min(), 6) / 6 if len(right) else 1.,
        min(float(ref.std(ddof=0)) / scale, 2) / 2 if len(ref) else 0.,
        np.clip((right.mean() - left.mean()) / scale, -2, 2) / 2
        if len(left) and len(right) else 0.], dtype=float)


def reward(predicted, observed):
    return float(1 - min(abs(predicted - observed), REWARD_CAP) / REWARD_CAP)


def _scores(policy, x, exploration=False):
    result = []
    for a, b in zip(policy['A'], policy['b']):
        solved = np.linalg.solve(np.asarray(a), x)
        mean = float(np.asarray(b) @ solved)
        bonus = policy['alpha'] * np.sqrt(max(float(x @ solved), 0.)) if exploration else 0.
        result.append(mean + bonus)
    return np.asarray(result)


def action_probabilities(policy, x):
    scores = _scores(policy, x, exploration=True)
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    probability = np.full(len(METHODS), policy['epsilon'] / len(METHODS))
    probability[best] += (1 - policy['epsilon']) / len(best)
    return probability


def choose_estimator(policy, observed, month, fallback):
    available = predictions(observed, month)
    complete = all(pd.notna(available[m]) for m in METHODS)
    if policy['status'] == 'trained_experimental' and complete:
        index = int(np.argmax(_scores(policy, context(observed, month))))
        return METHODS[index], 'frozen_bandit_policy'
    # Keep the benchmark's common-support constraint for learned decisions.
    # Named statistical fallback may still operate with fewer available arms.
    if pd.notna(available.get(fallback)):
        return fallback, 'declared_statistical_fallback'
    return None, 'insufficient_context_retained_unknown'


def fit_policy(observed, blocked, fallback):
    """Fit on inner training masks, freeze, then score outer test masks.

    The original blocked rows supply the reference-month whitelist. No test
    attendance can enter a training feature, prediction, action or reward.
    """
    observed = observed[['county','attendance_month','attendance']].copy()
    lookup = {county: group.set_index('attendance_month').attendance.sort_index()
              for county, group in observed.groupby('county', sort=True)}
    training = blocked[blocked.split.eq('method_selection')].dropna(subset=list(METHODS))
    training = training.sort_values(['month','county']).reset_index(drop=True)
    policy = dict(version=VERSION, arms=list(METHODS), features=list(FEATURES),
        A=[np.eye(len(FEATURES)).tolist() for _ in METHODS],
        b=[np.zeros(len(FEATURES)).tolist() for _ in METHODS], counts=[0]*len(METHODS),
        alpha=ALPHA, epsilon=EPSILON, reward_cap=REWARD_CAP, seed=SEED,
        training_rows=len(training), training_counties=int(training.county.nunique()),
        status='untrained_fallback', fallback=fallback,
        reward_definition='1 - min(absolute attendance error in people, 50) / 50',
        deployment='Explicit experimental option; default estimator unchanged',
        evaluation='Direct held-out prediction errors; not IPS or doubly-robust policy evaluation')
    rng = np.random.default_rng(SEED)
    logs = []
    for step, i in enumerate(rng.permutation(len(training))):
        row = training.iloc[i]
        months = row.donor_months.split(' | ') if row.donor_months else []
        ref = lookup[row.county].reindex(months).dropna()
        x = context(ref, row.month)
        probs = action_probabilities(policy, x)
        action = int(rng.choice(len(METHODS), p=probs))
        predicted = float(row[METHODS[action]])
        value = reward(predicted, float(row.observed))
        policy['A'][action] = (np.asarray(policy['A'][action]) + np.outer(x, x)).tolist()
        policy['b'][action] = (np.asarray(policy['b'][action]) + value * x).tolist()
        policy['counts'][action] += 1
        logs.append(dict(step=step, county=row.county, month=row.month, action=METHODS[action],
            propensity=float(probs[action]), predicted=predicted, observed=float(row.observed),
            reward=value, reference_months=row.donor_months,
            **{name:float(value) for name,value in zip(FEATURES,x)}))
    training_log = pd.DataFrame(logs, columns=['step','county','month','action','propensity',
        'predicted','observed','reward','reference_months',*FEATURES])
    policy['training_log_sha256'] = sha256(training_log.to_csv(index=False).encode()).hexdigest()
    if len(training) >= 30 and training.county.nunique() >= 3 and min(policy['counts']) >= 3:
        policy['status'] = 'trained_experimental'
    frozen_hash = sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
    test_rows = []
    eligible = blocked[blocked.split.eq('held_out_test')].dropna(subset=list(METHODS))
    for row in eligible.itertuples():
        months = row.donor_months.split(' | ') if row.donor_months else []
        ref = lookup[row.county].reindex(months).dropna()
        method, decision = choose_estimator(policy, ref, row.month, fallback)
        predicted = getattr(row,method) if method else np.nan
        test_rows.append(dict(county=row.county,month=row.month,observed=row.observed,
            chosen_estimator=method,decision=decision,bandit_prediction=predicted,
            **{m:getattr(row,m) for m in METHODS}))
    tests = pd.DataFrame(test_rows,columns=['county','month','observed','chosen_estimator',
        'decision','bandit_prediction',*METHODS])
    metrics = []
    for method in (*METHODS,'bandit_prediction'):
        errors = tests[method] - tests.observed
        metrics.append(dict(method=method,evaluated_rows=len(tests),counties=tests.county.nunique(),
            mae=float(errors.abs().mean()),rmse=float(np.sqrt((errors**2).mean())),
            county_balanced_mae=float(tests.assign(error=errors.abs()).groupby('county').error.mean().mean())))
    assert frozen_hash == sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
    return policy, training_log, tests, pd.DataFrame(metrics)
