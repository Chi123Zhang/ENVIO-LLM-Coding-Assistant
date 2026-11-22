# Observed count sensitivity

The private H3 source reconstruction includes **Test H3 with observed organization and delivery-site counts**, after rebuilding original inputs. **Run six observed-count H3 paths** tests three menus against two alternative intermediate measures. The page and downloads expose all six attempts, approximate p-values, intervals, significance labels, Holm-adjusted path p-values and fit diagnostics.

- Implementing organization–strategy pairs sum the recorded number of implementing partner organizations/practices over active reported strategy slots.
- Delivering fixed-site–strategy pairs sum recorded fixed physical locations currently delivering services over those slots.

These are not unique organization/site counts or numbers of people reached. A repeated entity can occur in more than one strategy. Planned sites and REACH survey identifiers are not substitutes for delivered sites or service volume. Safer Prescribing has no assumed delivery-site field.

Each version-specific field must have a matching source label and applicable dictionary condition. Every listed strategy state must resolve, and every active count must be a recorded nonnegative integer. An explicitly inactive slot contributes zero. A missing active count or unresolved state makes the entire aggregate unknown. Counts are never carried forward using implementation-status no-change rules. Original source cells are not modified.

Timing and inference match the existing observed-path analysis: attendance at t−2, mediator at t−1, CDC counts at t; county effects, linear calendar trend, population offset, interval likelihood for suppressed deaths, and delete-county jackknife. Unavailable fits remain visible. This is exploratory measurement sensitivity, with six planned indirect-path attempts and 18 link/path tests; additional prior analyses remain exploratory too. It is not causal validation of all H3 constructs.

`h3_observed_counts.zip` includes a readable HTML report, `paths.csv`, `all_link_and_path_tests.csv`, `fit_diagnostics.csv`, `count_aggregates.csv`, source-cell and dictionary references in `count_cell_basis.csv`, included model rows, delete-county fits and a protocol with input hashes. Exports remain private; original research data and results are not committed to GitHub.

The bounded dictionary evaluator now supports explicitly parenthesized groups of binary equalities joined by AND/OR. Missing fields use three-valued logic. Unsupported syntax, malformed expressions, ambiguous unparenthesized mixed operators and excessive nesting remain unresolved. No Python expression evaluation is used. Evaluation parses the entire expression, including logically dominated branches. See [REDCap's institutional branching-logic guide](https://www.ctsi.ufl.edu/wordpress/files/2017/06/Branching-Logic-in-REDCap-%E2%80%93-How.pdf) for source syntax context; this implementation deliberately supports a narrow subset.

Validation uses synthetic rules, missing/invalid/grouped-expression cases, aggregation completeness, zero versus unknown, source-hash binding and all-attempt exports. Real-source private acceptance compares the reusable application report with a separately constructed field-applicability ledger. No estimator is selected by a p-value.
