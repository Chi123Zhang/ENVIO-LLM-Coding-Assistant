# ORCCA Research Audit

This is an additive, source-linked review module. Start the existing `app.py`;
choose **ORCCA Research Audit** in the sidebar. The existing ENVIO workflow and
entry point remain in place. No statistical model is fitted by this module.

## Run

Python 3.12 is the tested runtime. From the repository:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt pytest
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.fileWatcherType none --browser.gatherUsageStats false
```

Default mode is a public synthetic demo, with no ORCCA upload controls. Synthetic
inputs can be downloaded from the page. For a **loopback-only local process**, use:

```sh
ORCCA_DEPLOYMENT_MODE=local_private ORCCA_ALLOW_PRIVATE_UPLOADS=1 .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.fileWatcherType none --browser.gatherUsageStats false
```

Never set these two flags on a public server. They describe operator-controlled
deployment trust, not user consent. The unchanged ENVIO page retains its existing
interview uploads; its added notice prohibits using them for ORCCA research data.

## Page functions and contracts

| Tab | Inputs | Result and limits |
|---|---|---|
| Data Consistency Audit | Panel + monthly + annual tables | Unique county-month keys, source intervals, suppression, population, full-year bounds with outside-window months. Compatibility does not verify a likelihood implementation. |
| EBP Exposure Verification | Panel + normalized tracker, or original local Tracker/dictionary pair | Candidate L1 and L13 replay, unknown-status bounds, source/field locators. Matching is not proof of original coding. |
| Model Specification Comparison | Two JSON specification records | Field differences and unknowns with user-supplied source locations; no automatic claims about a paper or model result. |
| Evidence-Grounded RAG | Bundled synthetic document or local PDF/text | Extractive retrieval with actual chunk text, file, page/lines and hash. No answer when evidence/retrieval fails. |
| Audit Report Export | Current results | In-memory ZIP with JSON, Markdown, CSV checks/anomalies; raw files excluded. Private reports may contain identifiers. |

Table formats are detected from bytes, not the filename extension. Text tables
support UTF-8 CSV/TSV (including CDC text exports named `.xls`). OOXML XLSX is
read-only; old binary XLS, macros, oversized archives and ambiguous workbook
sheets are rejected. Upload limit: 20 MB per file; expanded workbook limit:
100 MB / 2 million populated cells. No raw input is persisted by this module.

Required normalized columns:

* Panel: `county,date,Deaths,y_lo,y_hi,suppressed,population`. Exposure checks use
  `OEND_L1,OEND_L13,MOUD_L1,MOUD_L13` when present.
* Monthly: `county,month,Deaths`. Annual: `county,year,Deaths,population`.
  `normalize_mortality()` recognizes CDC county/month/year headers. County name
  normalization only strips the explicit ` County, NY` suffix; no fuzzy match.
* Tidy tracker: `county,month,menu,slot,show,started,discontinued`; optional
  `start_month`. Only OEND/MOUD, binary status or blanks are accepted. A skipped
  snapshot (`show=0`) needs a placeholder row, distinguishing it from missing
  events. An absent month breaks carry-forward.
* Original local workbooks: Tracker `orcca_tracker_v2` plus `Sheet1` county lookup, dictionary
  `ORCCAT_DD`. The adapter uses exact versioned REDCap identifiers; it does not
  guess field labels. Other workbook layouts require an explicit adapter.
* Specs JSON: array of exactly two objects with `spec_id` and `source`, plus
  `exposure_window,outcome_window,exposure_scale,missing_rule,distribution,county_effects,time_terms,prior_mortality,population_offset,annual_constraint,standard_errors`
  as available. Missing fields stay unknown. See `compare_specifications()`.

Every table reader records source hash/name and physical row number (for a quoted
multiline CSV record, its ending physical line). Workbook references retain sheet,
row, versioned field and dictionary row. Text retrieval retains physical text
lines; PDF lines refer to extracted page text, not the visual PDF grid. Scanned
pages with no extracted text fail visibly; OCR is not silently invented.

`Suppressed` is represented using an explicitly named **assumed 1–9 interval**.
Zero, missing and unavailable values remain distinct. Rules must be approved
against the actual extract's suppression notes before substantive analysis.
Unknown discontinuation produces lower/upper candidate intensities, not a filled
fact. Unknown started status is also flagged and widens the upper candidate
bound. The permissive replay value still counts only explicitly started entries;
it is exported separately from uncertainty bounds. Candidate carry-forward and
lag replay do not confirm restart semantics. The explicitly positional `Sheet1`
county lookup may contain repeated labels; only its first two columns are used,
with unique IDs required. Other duplicate workbook headers fail visibly.

## Reuse and permission boundaries

`retrieval.py` reuses ENVIO's tokenizer and BM25. `hybrid.py` reuses `Chunk`,
`TechMPowerRAG.retrieve()`, FAISS and the existing model identifiers. It constructs
a fresh in-memory index, bounds FAISS k for small corpora, and never reads or
writes the shared ENVIO index. Only model weights may be process-cached. Weights
must already exist locally; missing weights return `hybrid_unavailable_no_fallback`.
BM25 is fully runnable without model credentials/weights. These are software
paths, not evidence that one method is more accurate.

External explanation is disabled by default. Enabling it requires server flag
`ORCCA_ENABLE_EXTERNAL_LLM=1`, an exact `ORCCA_LLM_MODEL`, and the existing API key
configured as a server secret. Even then, only the exact bundled synthetic
document and preset questions qualify. Arbitrary questions or uploaded documents
cannot be sent. The generation output is separate and explicitly unreviewed;
exact quote/ID checks verify strings, not logical entailment. No external model
was called during local validation. Existing ENVIO model settings are unchanged.

Future interfaces are declared in `extensions.py` for Implementation Evidence
Analysis and Hypothesis Evaluation. They are visibly unimplemented. Human labels,
reference answers and generated outputs must never enter the retrieval index.
No private research evidence, results, labels or API keys are included here.
