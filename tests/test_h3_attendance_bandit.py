import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from orcca_audit.h3_attendance_bandit import (
    action_probabilities, choose_estimator, context, fit_policy, reward,
)
from orcca_audit.h3_attendance_validation import blocked_validation


def fixture():
    # Enough synthetic feedback to exercise the trained-policy branch. Small
    # samples with fewer than three feedback events per arm must fall back.
    months = pd.period_range('2018-01','2024-12',freq='M').astype(str)
    observed = pd.DataFrame([dict(county=c,attendance_month=m,attendance=10+i%7)
        for c in ['Demo A','Demo B','Demo C'] for i,m in enumerate(months)])
    blocked,_,protocol = blocked_validation(observed)
    return observed,blocked,protocol['selected_method']


def test_test_answer_changes_cannot_update_policy_or_training_rewards():
    observed,blocked,fallback = fixture()
    policy,log,tests,scores = fit_policy(observed,blocked,fallback)
    keys = set(blocked.loc[blocked.split.eq('held_out_test'),['county','month']].itertuples(index=False,name=None))
    changed = observed.copy()
    mask = [key in keys for key in zip(changed.county,changed.attendance_month)]
    changed.loc[mask,'attendance'] += 1000
    altered,_,protocol = blocked_validation(changed)
    other,other_log,other_tests,other_scores = fit_policy(changed,altered,protocol['selected_method'])
    assert json.dumps(policy,sort_keys=True) == json.dumps(other,sort_keys=True)
    pd.testing.assert_frame_equal(log,other_log)
    pd.testing.assert_frame_equal(tests.drop(columns='observed'),other_tests.drop(columns='observed'))
    assert not scores.equals(other_scores)


def test_single_feedback_per_training_target_with_valid_exploration_probabilities():
    observed,blocked,fallback = fixture()
    policy,log,tests,_ = fit_policy(observed,blocked,fallback)
    assert policy['status']=='trained_experimental'
    assert sum(policy['counts'])==len(log)==policy['training_rows']
    assert not log.duplicated(['county','month']).any()
    assert set(zip(log.county,log.month)).isdisjoint(set(zip(tests.county,tests.month)))
    assert log.propensity.between(policy['epsilon']/3,1).all()
    probabilities=action_probabilities(policy,np.ones(6))
    assert np.isclose(probabilities.sum(),1) and (probabilities>0).all()
    assert log.reward.between(0,1).all()


def test_frozen_prediction_does_not_learn_from_unknown_gaps():
    observed,blocked,fallback=fixture()
    policy,_,_,_=fit_policy(observed,blocked,fallback)
    frozen=json.dumps(policy,sort_keys=True)
    obs=observed[observed.county.eq('Demo A')].set_index('attendance_month').attendance
    first=choose_estimator(policy,obs,'2021-06',fallback)
    assert first==choose_estimator(policy,obs,'2021-06',fallback)
    assert first[1]=='frozen_bandit_policy'
    assert frozen==json.dumps(policy,sort_keys=True)
    assert choose_estimator(policy,obs.iloc[:0],'2021-06',fallback)==(
        None,'insufficient_context_retained_unknown')


def test_insufficient_training_has_explicit_fallback_and_no_reward_fabrication():
    observed,_,_=fixture()
    small=observed.iloc[:2]
    blocked,_,protocol=blocked_validation(small)
    policy,log,_,_=fit_policy(small,blocked,protocol['selected_method'])
    assert policy['status']=='untrained_fallback' and len(log)==0
    assert sum(policy['counts'])==0
    assert reward(12,12)==1 and reward(62,12)==0


def test_context_and_policy_ignore_mortality_and_p_values():
    observed,blocked,fallback=fixture()
    original=fit_policy(observed,blocked,fallback)[0]
    observed['mortality']=1e9
    observed['p_value']=.00001
    assert original==fit_policy(observed,blocked,fallback)[0]
    series=pd.Series([12,15,14],index=['2021-04','2021-05','2021-07'])
    assert np.isfinite(context(series,'2021-06')).all()


def test_ai_completed_archive_cannot_supply_rl_reward_labels():
    from orcca_audit.h3_hybrid_attendance import run_hybrid
    raw=BytesIO()
    with ZipFile(raw,'w') as z:
        z.writestr('run_metadata.json',json.dumps({'ai_unreviewed_attendance_county_months':1}))
        z.writestr('county_month.csv','county,attendance_month,attendance\nDemo A,2021-06,999\n')
    with pytest.raises(ValueError,match='recorded-attendance baseline'):
        run_hybrid(raw.getvalue(),[],method='experimental_contextual_bandit')
