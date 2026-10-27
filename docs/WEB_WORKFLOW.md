# Browser-operated ORCCA project workflow

Open **ORCCA Workflow**, or **Start / manage project** from any ORCCA page.
The existing ENVIO interview coding page and six core modules are unchanged.

## Operator and data boundary

The public app supports the entire workflow with bundled synthetic practice
inputs. It does not construct research upload widgets. External generation
remains disabled for research evidence. A browser checkbox cannot change that.

A private server configured with the existing `ORCCA_DEPLOYMENT_MODE=local_private`
and `ORCCA_ALLOW_PRIVATE_UPLOADS=1` flags supports uploads. The loopback-only
local server remains the authorized research environment. This change does not
make the public Streamlit app a private research host. Deploying a research
upload service remotely requires a separately authorized private deployment
with appropriate authentication, access and data-use arrangements.

Once the server is configured, users manage files, reruns, evidence applicability,
review decisions and downloads in the browser; no bundle-path edits are needed
for each dataset. Original files are never overwritten or committed. No new
storage service, hosted model call, database or shared evidence index is added.

## Steps

1. **Inputs.** Upload panel, monthly and annual sources together. Optionally add
   a tidy tracker or the original Tracker/dictionary workbook pair, results CSV,
   two source-linked specification records, evidence documents and a scoped
   completion-evidence CSV. Parse and run the audit. Each file is bounded to
   20 MB; retained project inputs are bounded to 100 MB combined.
2. **Audit.** Inspect the actual current-version counts and open the detailed
   Research Audit page. Results/errors are computed from these inputs, never
   borrowed from the saved five-finding study review.
3. **Completion.** Open Evidence Completion using **Current web project**.
   Evidence CSV needs county, event-applicable month, menu, strategy slot,
   version, text and source reference. Alternatively, link an exact uploaded
   passage to an eligible field in the browser and record who checked its
   applicability. An interview date alone does not establish event time.
   Free narrative is searchable but does not automatically become a field value.
   Automatic candidates still require explicit source `field=value` assertions.
   Review candidates with a reviewer and reason. Accepted source candidates can
   create a derived tracker version and rerun the audit. Original tracker values,
   field-level change markers and decision lineage are retained. Statistical
   scenarios remain separate and never change source-based audit values.
4. **Findings and review.** Generate draft statements from current calculation
   receipts. They are deterministic summaries, not LLM discoveries. Review each
   finding or record a separate researcher note. **Current web project** and
   **Saved research review** are explicitly different report sources. The saved
   review remains available without becoming evidence for new inputs. No H3–H5
   conclusions or model results are inferred from absent materials.
5. **Export.** Accepted-only is the default. A draft export requires explicitly
   clearing that option and preserves pending/rejected statuses. The ZIP includes
   the readable English report, current normalized tables, actual calculation
   receipt files under their cited names, evidence index, manifest, version
   lineage and finding/completion decisions. No independent human-reference
   evaluation labels or empirical performance metrics are manufactured.

PDF pages, DOCX extracted main-document paragraphs and PPTX slide text can be
indexed. DOCX is not assigned invented page numbers; slide references are not
mislabelled as PDF pages. Images, notes and image-only pages require review.

## Version and session behavior

A changed or removed input immediately invalidates the current project,
retrieval results, completion candidates and reviews. Audit, Findings and
Completion report that no current version exists until rerun; they do not
silently fall back to saved conclusions or sample data. Other uploaded inputs
are retained across page navigation, allowing a single file to be replaced.
Explicit retained-file removal also works after the original upload widget has
been cleaned up by Streamlit. Clearing the web project removes its input buffers.

Every derived version records its parent. Previously accepted findings expire
when inputs or source-backed tracker values change. A new version requires new
finding review. Review decisions are workflow decisions, not independent labels
for measuring research accuracy.

State is session-only. Download work before closing the session, reloading into
a new session, or restarting the server. Downloads are not an automatic restore
mechanism for past approvals. No study data are written to a global cache.

## Validation scope

Synthetic unit/integration tests cover cross-page state, stale-result blocking,
source applicability, exact quotations, source hashes, approved-only exports,
original/derived tracker separation, private-mode boundaries and document locators.
Actual browser acceptance separately exercises file choosers, input replacement,
required-source removal, retrieval, review, derived rerun and ZIP downloads.
Successful software checks do not establish RAG accuracy gains, missing-value
truth recovery, H2a reproduction or H3–H5 effects.
