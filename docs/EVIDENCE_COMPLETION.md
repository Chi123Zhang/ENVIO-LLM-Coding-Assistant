# Evidence Completion: bounded local research workflow

New page: `pages/03_ORCCA_Evidence_Completion.py`. Existing ENVIO core files,
configuration, findings and audit code are unchanged. No new dependency is needed.

## Inputs and privacy

Public mode exposes synthetic practice only. Local uploads require both
`ORCCA_DEPLOYMENT_MODE=local_private` and `ORCCA_ALLOW_PRIVATE_UPLOADS=1`.
The module has no external LLM call. Files stay in session memory. No source data,
research findings, reviewer names or outputs are committed or sent to a service.

Run the existing application with `streamlit run app.py`, then select
**ORCCA Evidence Completion** from native Streamlit navigation. Download the
synthetic input ZIP for a complete format example and practice only.

The first version supports OEND/MOUD `started`, `discontinued`, `start_month`.
It does not infer questionnaire meanings, impute mortality, or fit H2a/H3–H5.
It does not yet resolve all kinds of uncertainty in already populated cells.

Use a normalized CSV/TSV with:

`county,month,menu,slot,version,show,started,discontinued,start_month`

`month` is the applicable event month, `menu` is OEND/MOUD, `slot` identifies a
strategy under an explicitly reviewed questionnaire version. Binary statuses
must be mapped against the supplied dictionary to 0/1/blank first. Blank does
not mean zero. Duplicate keys fail visibly. The original Tracker + REDCap
dictionary workbook pair can instead use the existing version-aware adapter;
its semantic mappings still require researcher review. Original workbook
coordinates and dictionary references remain in the normalized output.

No-update/skip records and dependent fields whose started status is unknown
are held for review, not statistically filled. Start dates later than the event
month block source candidates. Arbitrary unsupported status codes fail visibly.

Evidence CSV/TSV requires `county,month,menu,slot,text` and optionally `version`
and `source_ref`. County, applicable month, menu, slot and version must match
exactly. Missing version matches only missing version; no cross-version guess.
Do not substitute interview/report dates for applicable event dates. A source
reference is user-supplied until verified; the actually loaded CSV filename,
physical ending line and excerpt hash are tracked separately. Reviewer answers,
model outputs and formal evaluation labels must not enter this source index.

## Retrieval and proposed values

The page reuses ENVIO BM25 and the existing fail-visible hybrid adapter. It
retrieves only exact-context excerpts. The prototype extracts standalone literal
lines such as `started=1`, `discontinued: 0`, `start_month=2024-01`. A literal
assertion must really exist in the source: never rewrite ambiguous narrative
as a definite assertion merely to obtain a candidate. Narrative can be retrieved
without an automatic candidate. Date, scope and semantic meaning require human
review. Conflicting assertions block automatic completion. Hybrid unavailability
never silently falls back. This is retrieval plus conservative rule extraction,
not an external LLM generation experiment or validated accuracy improvement.

## Statistical sensitivity scenarios

For eligible binary unknowns only, an optional illustrative Beta–Bernoulli
model uses observed **earlier** snapshots from the same county/menu/slot/version,
with `show=1`; discontinuation also requires `started=1`. At least three
comparable observations are required. With `s` ones in `n` observations and
Beta(1,1) prior, the working posterior is Beta(1+s, 1+n-s). Display its mean
and 2.5/97.5 percentiles. Generate 20 posterior-predictive binary draws with
an explicit seed and per-issue random stream. No future records or completed
candidate values become statistical support. Source conflicts cannot be bypassed
by the statistical button. These requirements do not establish model adequacy.

This assumes exchangeability, Bernoulli observations and ignorable missingness
within a strategy. Repeated monthly status records are dependent, so nominal
posterior intervals are **conditional on an unvalidated working model** and
may substantially understate uncertainty. This is an inspectable sensitivity
baseline, not formal multiple imputation, empirical validation or recovered
truth. It is not an analysis-ready recommendation for the real study. A proper
longitudinal imputation model needs agreed missingness/branching rules, strategy
identity over versions, temporal dependencies, model diagnostics and MNAR
sensitivity work before inferential use. No outcome or causal model is run here.

Method references: [SciPy Beta distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html),
[SciPy beta-binomial distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html),
[van Buuren: reporting missing-data analyses](https://stefvanbuuren.name/fimd/sec-reporting.html),
[van Buuren: multilevel imputation](https://stefvanbuuren.name/fimd/sec-multioutcome.html).

## Human decisions and exports

Reviewer, explicit accept/reject and rationale are mandatory. No answer is
preselected as accepted. Each decision records UTC time, exact input fingerprint,
candidate ID and settings. A field can have at most one active acceptance.
Replacing files or removing required inputs invalidates candidates and decisions.
Nothing is applied silently; restarting the server loses unsaved session work.

The ZIP includes JSON/Markdown, derived records, issue queue, all candidates,
review ledger and approved statistical scenario values. Original normalized
fields are preserved. Only accepted source candidates appear in `__reviewed`
columns, linked by `__candidate_id`; unresolved values stay missing. Statistical
values appear only in a separate long-form sensitivity overlay keyed by record,
field and scenario, never as observed values or extra participants. An accepted
statistical candidate records permission to explore a scenario, not confirmation
of the value's truth. CSV exports use existing spreadsheet formula protection.

## Evaluation before research claims

Software regression uses explicitly synthetic cases, not performance labels.
Next select a local researcher-reviewed pilot, adjudicate applicable event month
and missingness types, then compare rules, No-RAG LLM and RAG LLM under appropriate
data permission. Split by county/document, keep human answers outside the index,
measure supported recovery, false completion, abstention, correction effort and
actual time. Artificial masking of observed cells tests only that masking scheme;
it cannot establish performance on genuinely missing or MNAR fields. No accuracy,
F1, kappa, time saving or effect improvement is claimed by this release.
