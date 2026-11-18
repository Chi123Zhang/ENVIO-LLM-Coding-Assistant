# H3 source comparison and evidence retrieval

This release adds an incremental H3 modelling and source-review workflow. Original
ENVIO core files, configurations and data remain unchanged. Public mode uses only
fictional bundled inputs. Research uploads and external ORCCA generation remain
closed on Streamlit Cloud.

## Public demonstration

1. Open **ORCCA Workflow**, then **H3 source reconstruction**.
2. Select **Full H3 comparison demo** under **Synthetic source scenario**.
3. Choose **Rebuild H3 from source files**. The fixture has six fictional counties
   and 72 county-months. It contains a deliberately missing attendance month.
4. **Run H3 tests on current source data** shows the observed-source analysis.
5. **Run documented no-change comparison** reconstructs implementation snapshots
   only where an explicit fictional no-change response and known adjacent state
   support the rule. The same six exploratory models are fitted before and after.
6. **Retrieve synthetic attendance evidence** runs actual BM25 against the bundled
   fictional PDF. It shows source text and page references. This step makes no
   LLM call, creates no human label and fills no attendance value.
7. Download the comparison ZIP, evidence-review ZIP or full synthetic input ZIP.
   Open START_HERE.html in the comparison ZIP. Read nested analysis ZIPs for model
   rows, source hashes, diagnostics and the change ledger.
8. **Clear source reconstruction** removes the current comparison and retrieval.

Synthetic numerical results demonstrate computation only. They are not findings
from ORCCA, an empirical accuracy benchmark or evidence for causal mediation.

## Private research route

In a local-private deployment, original workbook, dictionary, CDC and attendance
uploads remain available. Source hashes bind the rule sources to the current
reconstruction. Existing observed values are never overwritten. Explicit no-change
reconstruction is separate from attendance evidence review and human-approved
additions. The private PDF evidence route can export blank independent review
forms. AI outputs are never treated as human approvals.

No real research results, Tracker, dictionary, attendance records, reviewer forms,
local model responses or private output ZIPs are included in this release. No new
cloud secret is needed. The local Qwen experiment is not a cloud generation service.

## Validation

The full local regression suite passed: 111 tests. An actual browser session
rebuilt the full fictional fixture (72 county-months in six counties), fitted
all six observed-source specifications, ran the documented no-change comparison
(24 to 65 complete county-months per path), retrieved the fictional PDF with
BM25 and downloaded the comparison ZIP. The demonstration reconstructed 252
mediator cells without changing observed values or filling attendance. These
are synthetic software checks, not research findings or accuracy estimates.
The six existing ENVIO core Python files are unchanged. No dependencies,
production secrets or research-upload permissions are changed.

## Rollback procedure

Before this release, main was
`9338ce87e84d64b95473327fed3dcc4392fc73eb`. If deployment fails, revert only the
new release merge and preserve unrelated commits. The Cloud entry remains app.py
on main in the existing repository. Verify both the original app and Workflow
after any rollback.
