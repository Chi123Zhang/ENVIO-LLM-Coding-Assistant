# Source-linked research findings

`pages/02_ORCCA_Research_Findings.py` adds a research-review view alongside the
existing audit demonstration. No original ENVIO file or model setting changes.
The page separates verified source checks, candidate-rule compatibility, method
review and insufficient material. It does not fit a model, use an external LLM,
or claim that retrieval improves research accuracy.

## Local research review

Prepare a UTF-8 JSON review bundle **outside the public repository**. Configure
its absolute path on a loopback-only local process:

```sh
ORCCA_DEPLOYMENT_MODE=local_private ORCCA_ALLOW_PRIVATE_UPLOADS=1 \
ORCCA_FINDINGS_BUNDLE=/absolute/private/review.json \
streamlit run app.py --server.address 127.0.0.1 --server.fileWatcherType none
```

The new page has five views: source-linked findings, evidence retrieval and
explicitly authored explanations, hypothesis material readiness, retrieval
capabilities/evaluation limits, and report export. Original evidence and Python
calculation receipts are labelled separately. Evidence hashes verify excerpt
content, not semantic truth. Document-level hashes must state whether they
identify original file bytes or a browser extraction. In local private mode, a missing configured bundle is reported explicitly. In
public mode, the five approved aggregate summaries bundled with the release
are the default; they are labelled as secondary summaries, not original data.

Use only raw source excerpts and reproducible calculation receipts in retrieval.
Human labels, adjudication and model outputs are rejected by the existing ENVIO
retrieval validator. Authored finding statements and question explanations are
not automatically added to the index. Preset questions and editable queries use English or field IDs. Queries are
not silently translated through an external service. Private hybrid uses cached
weights; public mode may download the fixed public assets documented in
`PUBLIC_RELEASE.md`. Failure remains explicit, without fallback.

## Required schema

| Field | Contract |
|---|---|
| schema_version | `1.0` |
| classification | `private_research`, `publication_draft`, `approved_public_summary`, or synthetic software-test classification |
| sources | Unique source_id, source_file and honest hash/version metadata |
| documents | Existing retrieval fields plus source_id, locator and evidence_kind |
| evidence_kind | original_excerpt, deterministic_receipt, or approved_summary |
| findings | Unique finding_id, title, statement, level, verification_method, limitations, evidence_ids |
| questions | Unique question_id, question, query, review_answer, answerability status and optional finding_ids |
| hypotheses | Hypothesis definition, available materials, missing materials and next step |
| evaluation | Actual human-label status; improvement_measured must be false |

Evidence IDs must resolve to documents, and documents must resolve to actual
source records. Every excerpt needs a matching SHA-256 and a valid location
range. Claiming reproduced statistical models is rejected by this review schema.
The export includes the current findings, all supporting excerpts and sources,
the current retrieval result when valid, and the review-bundle fingerprint.

## Public release gate

A local bundle is never made public merely by configuring a path. Outside local
private mode, loading requires all of the following:

- classification `approved_public_summary`;
- publication.status `approved` and a nonempty approval_reference;
- only approved_summary evidence with classification approved_public_excerpt;
- a SHA-256 pin for the exact approved bundle bytes: `public_review.py` pins the
  bundled default, while an explicit operator path requires
  `ORCCA_PUBLIC_FINDINGS_SHA256`.

A publication_draft or private_research bundle is denied. Public retrieval only
searches approved secondary summaries; it must not be represented as retrieval
over the undisclosed original documents. This technical gate does not replace
human approval of the actual release content. Keep private bundles, cases,
reference labels, paths, raw workbooks and private source excerpts out of Git.

Before any approved release, review the exact public bundle, record permission
for its content, pin its final hash, run local/browser checks and then deploy
through a separately approved merge. Keep original ORCCA uploads and external
generation disabled in Cloud. This page has no external generation function or
upload widget.

## Evaluation

Demonstrated software capabilities are source traceability, explicit failure,
dataset/query invalidation and report reproducibility. Accuracy improvement,
evidence support and time savings require real independent human annotation.
Compare rules/dictionary, No-RAG LLM and RAG LLM on unused county/document groups;
keep development pilot material out of claims about a formal held-out test set.
No performance metrics are supplied by this module.

## English interface and presentation

ORCCA controls, review labels and workflow descriptions are in English.
Findings and research questions come from the operator's bundle; use an English
presentation copy to change their language. Preserve original evidence text,
source IDs, reference links and checksums. An English presentation must not be
represented as a translated source quotation. Local presentation bundles remain
outside Git and still require explicit approval before any public release.

The findings view states which results come from Python checks and which come
from document review. Search locates supporting records; it does not establish
the findings. Review notes were prepared in advance with AI assistance and need
researcher confirmation. The notes are not generated during retrieval and are
excluded from the search index. Technical records sit behind expanders while
the main view shows findings, their basis and limits.

The existing `orcca_audit/ui.py` has only presentation changes: links to the
additional pages and a compact exposure summary, with raw diagnostics retained
under expanders. No audit calculation, privacy rule or original ENVIO core file
is changed by the English-language update.
