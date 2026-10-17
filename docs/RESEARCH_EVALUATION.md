# Planned research evaluation — not run

The existing private review case set is not bundled or labelled by this feature.
No human reference labels, accuracy, kappa, time saving or empirical RAG gains
are reported. Synthetic unit tests verify software behavior only.

## One bounded experiment

Use source-linked research-review questions as the primary task: identify an
exposure definition, find its locator, compare windows, or abstain when a model
implementation is absent. Compare (1) deterministic dictionary/rule lookup,
(2) No-RAG LLM with the task/schema only, and (3) RAG LLM with the identical model,
prompt/output schema and decoding settings plus retrieved evidence. Record
model/prompt/retriever versions, hashes, output, latency, and external-service
permission. Run private evidence locally; current external demo gate does not
authorize sending study evidence. Build a separate approved runner before that
experiment. A No-RAG answer based on outside knowledge cannot count as evidence.

Two researchers independently label correct answer, acceptable source locations,
answerability and required unknowns; lock references before scoring. Record their
disagreements before adjudication, then resolve through discussion/third reviewer.
Keep reference labels in a separate private directory. Group by document family
and county/source interview when applicable; duplicate/adjacent passages stay in
the same development/test group. Freeze prompts and retrieval on development
only. Audit and test answers/model outputs must not be indexed as source evidence.

Report per-task mapping accuracy, state errors, exact locator correctness,
reviewer-rated evidence support, abstention/unknown rate, actual human edits and
measured elapsed review time. For genuine multi-class coding, report per-class
precision/recall/F1, macro-F1, and a kappa suitable for the label structure;
binary multi-label codes need per-code agreement, not forced mutually exclusive
classes. Disclose case/group counts, imbalance, confidence limitations, erroneous
citations, unsupported claims and disagreements. No scores until real labels
exist. A small convenience set measures feasibility, not generalization.

## Interfaces for later implementation research

First establish county, observation time, event-applicability time and source
certainty. Untimed text remains document-level. No mention, explicit absence and
insufficient information are separate labels. Literature can support mechanisms
but cannot create county events. ENVIO interview evidence is not ORCCA evidence.

Only after approved variable definitions and valid linkage, consider a small
prespecified descriptive question: whether prior coalition participation or
leadership engagement is associated with subsequent implementation/coverage.
Specify the exposure, outcome and time window first; distinguish potential
common causes from mediators such as implementation intensity. Account for
repeated counties and limited independent clusters. Do not treat county-month
rows as independent, use significance to select models, or label associations
causal. No H2a refit, DML, causal forest or automated hypothesis search is part
of this feature. Missing original scripts still limit verification of likelihood,
annual constraints, offset, time terms and uncertainty estimation.
