# H3 study preparation demo

Start at `/ORCCA_Workflow`, not the original ENVIO coding page. Choose **H3 study preparation**. The original ENVIO page and the historical audit remain available. H4 and H5 are deferred.

Use only synthetic practice data for a shareable recording. Do not expose study filenames, raw records, private quotes, attendance identities or local research review exports in the recording. The private-upload option is available only on the authorized local server; the public app has no research upload controls.

## A four-minute route

1. Explain the research question: coalition process X may act through EBP implementation/reach M on a later outcome Y. The three cards describe the path, not estimated effects.
2. In **Project inputs**, leave **Synthetic practice** selected and click **Load synthetic H3 project**. No old H2a panel is required.
3. Open **Data review**. Show counties, event months and repeated records. Point out that the coalition exposure and later outcome are not supplied in the practice case. A missing requirement remains missing; the demo does not fabricate it.
4. Open **Evidence and completion**. Enter `coalition` and click **Find source evidence**. Show the synthetic source text, locator and retrieval status. Explain that BM25 retrieves; it does not establish mediation. A real dictionary field ID uses exact version-field lookup. External generation is disabled. Hybrid must report its actual availability, without a BM25 fallback.
5. Follow **Review source-backed completion candidates**. With **Current web project** selected, review the one eligible synthetic missing discontinuation status. Retrieve its source candidate; accept it with a reviewer and reason. In the completion page's export/handoff section, click **Create derived version and rerun audit**. This demonstrates human-reviewed source recovery, not statistical truth recovery or a new H3 result.
6. Return to Workflow. The project revision changes and prior findings/approvals expire. Generate current study findings, inspect the source and limitations, and accept one finding as a preparation check. This approval is not an independent evaluation label.
7. In **Export**, download the accepted-only ZIP. Open START_HERE.html and show the version/source records. The ZIP is synthetic and includes the original/derived tracker lineage and completion decision.

## Suggested narration

> I have narrowed the prototype to preparing H3: whether coalition processes relate to overdose outcomes through implementation and reach. The Tracker is the primary implementation source. Interviews, meeting records and attendance can help check ambiguous records and interpret mechanisms, without replacing the structured data or duplicating prior qualitative work. This demonstration uses synthetic inputs. It shows data review, source retrieval, a human-reviewed completion and versioned findings. It does not estimate a mediation effect or demonstrate an accuracy gain from RAG.

## What can be claimed

- The workflow accepts a Tracker without the old H2a panel in private mode.
- Available inputs and unresolved H3 requirements are visible.
- Exact dictionary lookup and BM25 retrieval are distinct.
- Source-based candidates need explicit review before a derived version is created.
- Replacing/removing inputs or changing project type invalidates current results and approvals.
- Export contains current-version records, review status and source references.

Do not claim H3 is supported, missing values are all recoverable, CTH phases are reconstructed, hybrid passed without running it, or RAG improved accuracy/time without independent evaluation. Generic ENVIO codebook labels are not automatically validated ORCCA measures. External LLM generation was not added by this change.

## Implementation boundary

New H3 logic lives in orcca_audit/study.py and study_ui.py. The existing project/review/export functions are reused. Small dispatch additions join the existing Audit, Findings, Completion and ENVIO handoff to the H3 project. No original ENVIO core file, model setting or dependency is changed by this release. Historical audit arithmetic remains separately tested. The first field-review table focuses on OEND/MOUD slot 1; it is not an inventory of every strategy or a computed mediator. All original raw columns are counted.
