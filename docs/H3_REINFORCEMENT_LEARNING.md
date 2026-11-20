# Experimental reinforcement learning for H3 data preparation

## What it learns

`experimental_contextual_bandit` is an optional one-step reinforcement-learning
experiment. A disjoint LinUCB policy with epsilon exploration learns which of
three attendance estimators to use for a missing county-month: interpolation,
uniform local donors or time-distance-weighted donors. It does not train an LLM,
change a study hypothesis, optimize p-values, or establish missing headcounts.
The existing blocked-validation estimator remains the default.

RAG still identifies eligible, same-county/date meeting evidence. The frozen
bandit only chooses the statistical estimator after that evidence gate. Verified
attendance is unchanged; existing source-linked unreviewed AI candidates remain
separate fixed estimates. The H3 statistical specification is unchanged.

## Context, action, feedback and memory

Six bounded features describe only known, same-county reference attendance:
intercept, reference count, nearest earlier/later month distance, relative
variation and earlier/later mean change. Predictions use a six-month window.
This is retrospective reconstruction; future meetings may inform a past gap.

Each eligible training target is visited once in a seeded permutation. The
learner samples one action from epsilon-mixed LinUCB scores, observes only that
action's reward, and updates only that action's linear model. Reward is
`1 - min(abs(prediction - recorded attendance), 50) / 50`. The 50-person cap is
an engineering reward scale, not a clinical or measurement threshold. Alpha
0.25, epsilon 0.10, unit ridge penalty and seed 20261120 are fixed before this
experiment. No parameter is selected using H3 results.

Policy memory stores the source-bound training-log hash, feature schema,
version, model matrices, action counts and settings. Logs preserve each context,
chosen action, its actual sampling probability and the recorded-data reward.
No rewards are invented for truly missing attendance. A missing month's
prediction cannot update the policy, and test evaluation leaves policy memory
unchanged. No user-global or external learning service is used.

## Evaluation and limits

The existing fixed two-month block split is reused. Outer test attendance is
excluded from all training reference records. Each inner training target's
entire block is also excluded from its references. Only common-support rows
with all three available estimators enter training and test comparisons. At
least 30 training rows, three counties and three feedback events per action are
required; otherwise the named statistical fallback is reported explicitly.

After training, evaluation uses greedy predicted reward with exploration off.
All four policies are evaluated directly against the same held-out recorded
attendance. This is possible because each candidate estimator can be computed
on masked historical records. It is not off-policy evaluation of production
logs; no IPS or doubly-robust claim is made. Training propensities alone do not
justify evaluating a policy on its own training rewards.

This test split has already been used in earlier research comparisons. It is
held out from this policy's training, but is not an untouched external dataset.
Small sample size, county/time dependence and missing-not-at-random records
limit generalization. Better reward or prediction error would not prove H3 or
guarantee statistical significance. Attendance draw ranges remain descriptive,
not calibrated confidence intervals; policy uncertainty is not fully propagated.

The optional experiment does not automatically replace the default, even if one
benchmark looks better. A future independent validation set should precede
promotion. More verified attendance and independently checked retrieval labels
are needed before attempting RL-based query rewriting or retrieval routing.

## Website and exports

In **H3 source reconstruction**, rebuild the original sources, run the desired
documented no-change baseline, and retrieve meeting evidence. Select
`experimental_contextual_bandit` in **Evidence-gated attendance estimator**, then
run **Run evidence-gated H3 sensitivity**. The same option is available in the
separate unreviewed-AI sensitivity route.

The page reports training support, all fixed-method and bandit test errors, and
all six H3 p-values. The report ZIP additionally includes `bandit_policy.json`,
`bandit_training_log.csv`, `bandit_heldout_predictions.csv` and
`bandit_evaluation.csv`. Each missing-value ledger row records its actual chosen
estimator and any fallback. Downloads contain private research data when private
sources are used; none of those research artifacts belong in public GitHub.

## Method references

- Li, Chu, Langford & Schapire (2010), [A Contextual-Bandit Approach to Personalized
  News Article Recommendation](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/p661.pdf),
  disjoint LinUCB algorithm. This implementation adds explicit epsilon exploration.
- Ma et al. (2023), [Query Rewriting in Retrieval-Augmented Large Language
  Models](https://aclanthology.org/2023.emnlp-main.322/), illustrates a distinct
  future RL direction requiring an appropriate reader/retrieval reward. That
  query-rewriter training is not implemented or claimed here.
