# Browser acceptance, 2026-10-17

Scope: feature/orcca-research-audit, local full Streamlit application. No merge,
production deployment, model fit, or research-data publication. All browser
uploads used fictional DEMO_A/B/C files. Private research pilot artifacts are
outside this repository and are not included here.

| Check | Result and evidence |
|---|---|
| Actual file chooser uploads | Passed for panel, monthly, annual, tidy tracker, JSON specifications, text evidence, and the alternative Tracker/dictionary XLSX pair. These were browser uploads, not AppTest injection. |
| Five audit tabs | Passed. Consistency: 18 rows / 3 synthetic counties / 18 source matches. Tidy exposure: 72 candidate matches, 1 uncertain check. Workbook exposure: 72 matches, 0 uncertain checks. Specification differences and unknown fields displayed; source-linked retrieval and export worked. |
| Replacement / missing inputs | Passed. Replacing one outcome with 999 changed source matches from 18 to 17. Removing required panel removed results/download. Replacing evidence removed previous citations before retrieval. Malformed specification JSON displayed a contract error and no tabs/report. |
| Session clear | Initially failed: selected filenames remained visible while backend returned to demo. Fixed by rotating uploader widget identity on clear. Browser recheck: 0 old file chips, 8 empty upload controls, explicit demo notice. |
| Switch ENVIO / ORCCA | Passed with expanded sidebar navigation. Returning after uploaded inputs resets uploads and explicitly identifies the demo; earlier private results do not appear as current. |
| Actual report ZIP downloads | Passed for both input paths. Opened browser downloads with Python zipfile/JSON; every deterministic section and all input SHA-256 hashes matched independent backend recomputation. Tidy ZIP retrieval section also exactly matched BM25 recomputation. Workbook ZIP recorded actual hybrid retrieval. |
| BM25 | Actually executed in browser and Python using original ENVIO tokenizer/BM25. Extractive evidence is not a generated answer. |
| Hybrid | Actually loaded existing local weights with local_files_only=True, then used ENVIO retrieval with FAISS/BM25/reranking. Models: sentence-transformers/all-MiniLM-L6-v2 and cross-encoder/ms-marco-MiniLM-L-6-v2. Browser retrieval succeeded. This smoke test is not a retrieval-quality evaluation. |
| Real LLM | Not tested. No enabled operator model ID, external-generation flag, or provider key in this process. Generation stayed disabled. Regression model replies are mocks, explicitly separate from real-weight retrieval. |
| Legacy ENVIO | Browser landing, synthetic text upload, coding/qa/summary mode selection, and explicit missing-key error passed. Original coding/RAG/synthesis functions pass mocked-model regression; evaluation and PDF/CSV/ZIP functions pass synthetic software tests. Full provider-backed legacy workflow is not claimed tested. |
| Public ORCCA mode | Browser demo renders and has no file upload widgets. BM25 works and generation is disabled. Production Cloud behavior remains untested. |

`python -m pytest -q`: **32 passed**. `python -m pip check`: no broken
requirements. Existing structure/hash checks pass. No original ENVIO core file,
dependency, data file, model setting or saved result changed in this acceptance
patch. Only ORCCA uploader state was fixed; additional edits are tests and docs.

The actual input element was hidden and the first chooser attempt timed out.
Using its visible Upload button succeeded; complete uploads and downloads then
finished normally. At a narrow browser width, expand the sidebar before page
navigation and collapse it before using main-area controls.

## Reproduce locally with synthetic data only

1. Run `python tests/build_browser_fixtures.py /tmp/orcca-fixtures` from the repo.
2. Start the existing full application:
   `ORCCA_DEPLOYMENT_MODE=local_private ORCCA_ALLOW_PRIVATE_UPLOADS=1 streamlit run app.py --server.address 127.0.0.1 --server.fileWatcherType none`.
3. Open ORCCA Research Audit. Upload panel.csv, monthly.csv, annual.csv,
   tracker.csv, specs.json and evidence.txt in their corresponding controls.
   Confirm the values above. Inspect all five tabs.
4. In Evidence-Grounded RAG, retrieve with BM25, then hybrid if both weights are
   already cached. A missing cache must report explicit failure, not success.
   Do not enable external generation for uploaded evidence.
5. Download the report. Inspect audit_report.json and CSV sections; compare
   provenance.uploaded_sources hashes with the actual uploaded bytes.
6. Remove panel.csv. Results and report download must disappear. Upload
   panel_changed.csv; source matches must become 17. Replace evidence.txt with
   evidence_changed.txt; old citations must disappear. Search a unique absent
   token to check the explicit no-match state.
7. Replace specs.json with bad_specs.json; no audit tabs/report should appear.
   Restore valid specs. Remove tidy tracker before selecting synthetic_tracker.xlsx
   and synthetic_dictionary.xlsx as a pair. Expect 72 matches and 0 uncertain
   checks for this separate, deliberately simpler workbook fixture.
8. Clear the session: all selected filenames must disappear. With uploads
   present, switch to ENVIO and back: the current dataset must be identified
   correctly, without stale private results. Streamlit resets page-owned inputs.
9. Restart without both private flags for public-mode acceptance. No ORCCA
   uploader should exist. Never deploy the private flags to public hosting.

Public synthetic-demo code is ready for owner review. Deployment still requires
explicit merge/deployment approval and a Cloud smoke test. Missing LLM credentials
do not block a generation-disabled demo, but no real-generation claim is made.
