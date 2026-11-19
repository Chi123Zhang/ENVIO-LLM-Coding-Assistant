# H3 input guide

Open **Workflow → H3 source reconstruction** for the current original-source
attendance pathway. **Project inputs** remains the separate Tracker review route.

## 1. Three primary tables

- Original Tracker workbook: implementation records and intermediate measures.
- Original monthly CDC table: monthly death counts and suppressed intervals.
- Original annual CDC table: annual population and total-consistency information.

No previously derived H2a panel is required. Counts are not mortality rates.

## 2. Definitions and coalition-process data

The REDCap dictionary defines version-specific fields and status rules. County
attendance CSVs supply the recorded coalition-process exposure for the current
attendance → implementation → death-count pathway. They are structured inputs,
not merely searchable documents. The reconstruction requires both the dictionary
and attendance data in addition to the three primary tables.

Missing required files keep reconstruction disabled. A message lists what is
missing. Existing upload widget keys, accepted formats and input-fingerprint
invalidation remain unchanged.

## 3. Targeted RAG evidence

After reconstruction, **Find evidence for missing attendance** accepts meeting
PDFs. Retrieval targets county-month gaps in the current structured inputs and
preserves original passages and page references. Interviews and other documents
remain available through the project evidence workflow; this targeted PDF route
does not automatically extract attendance measures from them.

BM25 and source memory assist evidence review. Source memory stores original
excerpts, locators and retrieval logs. It is not Transformer training or automatic
numeric completion. A matching passage is not a verified headcount. Accepted
corrections require an explicit review; unknown values remain unknown.

## Release scope and validation

Baseline main: `f60f01ec0dec8406d5a1d1bedb6b4b748d152d9a`.

Changes are confined to four ORCCA UI modules and this guide:

- `h3_sources_ui.py`: input overview, primary-table grouping, separate dictionary
  and attendance controls, and a missing-input explanation.
- `h3_attendance_evidence_ui.py`: identify the PDF uploader as supporting evidence
  and explain the limits of retrieval and source memory.
- `study_ui.py`: distinguish the statistical source route from Tracker review;
  clarify that defined attendance measures can be structured model inputs.
- `research_plan_ui.py`: show private-session planning guidance in a private
  session and retain fictional-question guidance in public practice.

No calculation, model specification, upload permission, external-generation
policy, original core module, raw research file or model configuration changed.
The existing 115-test regression suite passed locally. Local browser inspection
confirmed the group order and the disabled reconstruction button when inputs are
absent. GitHub CI and hosted browser checks are recorded in the release PR.

Research data, downloaded private reports and reviewer materials are excluded
from this release.
