# H3 source reconstruction

Open Workflow and select **H3 source reconstruction**. Run **Rebuild H3 from source files** to inspect coverage and download a ZIP containing an HTML report, the county-month table, Tracker cell references, coverage, and input hashes.

The public deployment bundles synthetic source workbooks and CDC-shaped text tables. Research file uploads require the existing operator settings `ORCCA_DEPLOYMENT_MODE=local_private` and `ORCCA_ALLOW_PRIVATE_UPLOADS=1`.

In a private session, provide the original Tracker workbook (orcca_tracker_v2 and Sheet1), its version-specific REDCap dictionary (ORCCAT_DD), the original monthly and annual CDC tables, and the original attendance CSVs. The original CDC tables may be tab-delimited text with an .xls extension. Parsing uses file contents.

Choose the outcome window. The current ordering uses attendance in t−2, a Tracker snapshot in t−1, and the outcome in t. This is a proposed temporal specification for review. County names are checked against CDC FIPS codes. Counts published as Suppressed retain the research interval of 1–9. Exact zeros remain zero. Confirm CDC query filters before interpreting the endpoint.

Active snapshots count explicitly listed strategies with applicable, defined start and stop answers and consistent recorded dates. The current branching evaluator supports flat binary equalities joined by one type of AND or OR. Unsupported expressions remain unresolved. Months without a report remain unknown. One unresolved listed strategy makes the complete menu snapshot unavailable. Venue breadth counts distinct decoded venue categories among clearly active entries. The report does not establish beneficiary reach.

Attendance counts distinct normalized names explicitly marked present at a meeting, averaged across observed meetings in the month. Empty marks remain unrecorded. Meetings with no explicit present marks remain unknown, including cases where an older provisional table counted them as zero. Staff roles, spelling variants, proxy attendance and historical membership require review.

This route creates source-linked inputs and reports coverage. It performs no imputation or model fitting. The earlier panel-based sensitivity analysis remains available in a collapsed section, with its separate assumptions and output. The source reconstruction ZIP contains no p-values. Real source data and review outputs must remain within the approved private environment.

The three menus each produce one row per county-month. Coverage is reported by menu. The full H3 research question also requires confirmed process definitions, service reach, nonfatal outcomes and an appropriate confounder strategy.

The source reconstruction inputs are independent of the project loaded in Workflow steps 1–5. File replacement, removal, route changes and window changes invalidate the previous reconstruction output. Use the explicit source report for this run. No previous Findings or completion decision is automatically treated as a source correction.

## Browser-based exploratory tests

After reconstruction, select **Run H3 tests on current source data**. The private
route can use the entire supplied CDC period. Each menu and intermediate measure
uses its own complete records; unknown inputs remain unknown. Uploading all
records does not make every row eligible for estimation.

The exploratory specification uses attendance at t−2, implementation at t−1 and
CDC counts at t. OLS and interval Poisson links include county fixed effects,
a linear trend and a population offset. Delete-county jackknife uncertainty is
approximate; failed fits have no invented p-values. This is not validated causal
mediation or the complete H3 study.

**Download H3 statistical report** includes input hashes, source reconstruction,
model rows, results and stability diagnostics. Changing inputs or rebuilding
invalidates the statistical result. Public practice remains synthetic; research
uploads require private deployment. No external generation or imputation runs.
