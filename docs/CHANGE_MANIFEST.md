# Incremental change manifest

Baseline main: `f716bb4f93cc5854adb004d4b4e239be84cd1184`.
The 28 pre-existing tracked files are enumerated with original SHA-256 hashes in
`baseline_main.json`. No existing path is renamed or deleted.

| Existing path | Exact change | Reason |
|---|---|---|
| `app.py` | Three inserted sidebar lines, zero deleted lines | Link to the two workspaces and display the public-data boundary. All original functions, settings and controls remain. |
| `requirements.txt` | Three appended lines, zero deleted lines | Blank separator, explanatory comment, `openpyxl>=3.1.5,<4` for read-only local workbook parsing. Existing dependency constraints are unchanged. |

`rag_system.py`, `expression_layer.py`, `llm_batch_coding.py`,
`query_orchestrator.py`, `background_memory.py`, the existing README, research
documents, evaluation files and all other original paths remain byte-identical.
The baseline test removes only the two exact documented additions and verifies
every original file's SHA-256. It fails on any other change or missing file.

New paths:

* `orcca_audit/`: deterministic audit engine, bounded in-memory I/O, local workbook
  adapter, synthetic fixtures/document, permission policy, BM25/hybrid retrieval,
  exports, isolated UI and future research interfaces.
* `pages/01_ORCCA_Research_Audit.py`: new native Streamlit page. Existing `app.py`
  remains the launch target; no router migration or module move.
* `tests/`: original-function regressions, page navigation/uploads, source/parser
  contracts, small-corpus hybrid adapter and baseline integrity.
* `docs/`: this manifest, baseline hashes, operation/schema guide, deployment
  runbook, research evaluation design and actual testing record. Existing docs
  are not overwritten.
* `.github/workflows/orcca-audit.yml`: PR/feature-branch tests only, no deployment.
* `.gitignore`: new cache/secret/private-output exclusions; no tracked file removal.

Local private validation code/results stay outside this clone. No study rows,
questionnaire text, human labels, private findings or API keys are added to Git.
