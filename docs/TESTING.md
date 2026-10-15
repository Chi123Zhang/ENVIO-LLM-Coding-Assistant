# Actual validation and limits

Local macOS / Python 3.12.14; Streamlit 1.66.0, pandas 3.0.6, numpy 2.5.4,
FAISS 1.15.1, openpyxl 3.1.5, sentence-transformers 6.1.0.
The complete existing requirements installed in an isolated environment.
`python -m pip check`: no broken requirements.

`python -m pytest -q`: **29 passed**, including:

| Area | Actual checks |
|---|---|
| Existing structure | All 28 original tracked paths preserved; only exact documented additions accepted |
| Interview Coding / RAG | Original TXT loader, coding and RAG refinement functions with synthetic text and mocked LLM replies |
| Evaluation | Original coding comparison and paired summary, synthetic software fixtures only |
| Synthesis | Original role-adaptive interface plus actual expression-layer function with mocked model |
| Export | Original PDF/CSV/ZIP generation; new in-memory audit ZIP and CSV formula escaping |
| Full page | Existing app entry renders, all original modes remain, native page switch and return work; new page has five tabs |
| Privacy / uploads | Public ORCCA constructs no upload widget; both server flags needed locally; injected uploaded-file payloads run all new tabs; incomplete inputs and invalid specs visibly stop |
| Deterministic audit | Unique keys, annual outside-window bounds, incomplete year, zero/suppression/missing states, lag history/unknowns, exact dictionary mapping, duplicate lookup rejection |
| Retrieval | Actual BM25 source text/locators/hashes; empty/no-match cases, reference-label exclusion, quote mismatch, external-model permission and explicit hybrid failure |
| Hybrid adapter | Actual original retrieve algorithm plus real FAISS, fake embeddings/reranker; small corpus has no padded negative/duplicate indices |

Browser checks on the running full local app: original landing page, ORCCA native
navigation, synthetic checks, source evidence with exact file/lines/hash, disabled
external generation, actual ZIP download (opened and inspected). The local private
file chooser transferred a synthetic panel and displayed the required-input
error; a complete multi-file browser flow was not completed because chooser
transfers were unusually slow. Full payload processing is covered by AppTest
injection and parser/engine tests, not claimed as a full browser upload test.

The initial local server used macOS FSEvents and exited during development file
changes. Restart with `--server.fileWatcherType none` as documented; no core code
change was needed. Local font caches can be directed to a writable
`MPLCONFIGDIR`. New-page dataframe width deprecation notices remain compatible
with the existing minimum Streamlit constraint.

The existing production app was inspected read-only and displayed its original
ENVIO landing page. No merge, production setting/secret change or reboot was
performed. Cloud build, deployed ORCCA behavior and production model calls remain
pending owner approval. No real external model, downloaded weight performance or
human reference labels were tested. Cached-model hybrid integration has a software
adapter test; semantic retrieval quality and provider availability are unverified.
Private source validation ran locally with no model fitting or external model
calls; its research findings are deliberately excluded from this public file.
