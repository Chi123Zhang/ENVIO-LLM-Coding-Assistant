# H3 attendance reconstruction validation

The default `select_by_blocked_validation` compares interpolation, uniform local
donors and inverse-month-distance weighted local donors. All use only recorded
attendance in the same county within six months. This is retrospective
reconstruction; later meetings may inform an earlier missing month.

## Frozen evaluation protocol

- Calendar months are grouped into fixed consecutive two-month blocks using the
  monthly period ordinal divided by two (integer division).
- Every fifth block (`block % 5 == 0`) is held out. All its attendance values are
  excluded from method-selection targets and donor records.
- For each remaining method-selection target, its entire two-month block is
  also masked before prediction. Thus adjacent held-out answers cannot act as
  each other's donors.
- Methods are compared on the same eligible records. The estimator is selected
  by mean county-specific MAE, giving counties equal weight. Mortality,
  implementation measures and model p-values are not inputs to selection.
- The separate held-out test scores are descriptive evaluation and never select
  the estimator. The final reconstruction uses all observed attendance after
  selection. If no training comparison is possible, a declared uniform-donor
  fallback is reported as unvalidated.

The earlier `select_by_masked_error` remains available for reproducibility. Its
single-month masking scores serve as the legacy comparison, not independent test
performance. The H3 specification and inference are unchanged by this release.

## Limits and exports

The evidence gate requires a supplied dated meeting PDF and retrieved evidence;
it does not establish a headcount. Statistical estimates remain distinct from
observed attendance and source-linked, unreviewed AI counts. Donor draw ranges
are descriptive and are not calibrated confidence intervals. Known-record
masking does not establish accuracy for truly missing records or missing-not-at-
random data. No independent empirical retrieval labels are introduced.

The sensitivity ZIP includes `blocked_validation_predictions.csv`,
`blocked_validation_scores.csv`, the frozen protocol, donor months, attendance
draw ranges, all six H3 paths, p-values and fit diagnostics. The website exposes
validation scores and a direct CSV download as well as the complete ZIP.

Retrieval memory now keys the complete source provenance, including county,
date and locators. Caller edits cannot mutate stored results. Failed requests
are retryable, empty scopes do not displace useful results, and successful query
memory is bounded to 128 entries per session.
