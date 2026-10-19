# ORCCA research review: public summary

Approved for aggregate publication by the project owner on 2026-10-20.
This is a secondary review of supplied research materials. Original files, private excerpts and review cases are not published.
The checks do not reproduce H2a or test H3–H5 effects.

Panel structure and source checks agree
The panel contains 136 unique county–month records: eight counties over 17 months. Death intervals, suppression flags and population match the supplied sources for all 136 rows.
Basis: Python checks compared each panel row with the monthly and annual CDC files, then checked key uniqueness and coverage.
Limits: This establishes agreement among the supplied files. It does not validate every outcome definition or the fitted model.

Exposure values match a candidate rule; status questions remain
All 544 OEND/MOUD lag values fall within the permissive candidate-rule matches. Of these checks, 64 still involve uncertain status or require review.
Basis: Start and discontinuation fields were mapped by questionnaire version. Snapshot and L1/L13 calculations retained definite values, bounds and permissive candidates.
Limits: This is not 100% accuracy or 64 confirmed errors. Blanks may reflect branching. The original panel-building code and status rules are still needed.

Partial-year counts cannot be equated with annual totals
For five county–years, the counts within the analysis window cannot equal the full annual total. Including months outside the window leaves all 16 county–years compatible with the annual sources.
Basis: Bounds for all 12 months were divided into inside- and outside-window counts, then checked jointly against the annual totals.
Limits: This rules out a particular partial-year equality. It does not show that the original model used that equality; the joint likelihood and handling of outside-window months need code review.

The two H2a summaries use different stated designs
The slides describe three months of exposure followed by three months of deaths, with standardized log intensity. The panel stores monthly deaths and candidate raw intensities. The reported estimates should not be treated as the same specification.
Basis: The review compared slides 6–8, panel fields and records, the main results rows and the supplied email description. No model was refitted.
Limits: The stored columns do not reveal every transformation before fitting. Original code is needed to establish which specifications were used and why the estimates differ.

H3–H5 need process data beyond the supplied files
The abstract outlines mediation, moderation and CTH sequencing questions. Table 4 lists meeting, interview and fidelity sources, but the eight supplied materials do not contain the complete county–time process data needed to analyse these questions.
Basis: The inventory compared the supplied files and local fields with abstract pages 1, 3 and 4 and the text-analysis summaries on slides 2–3.
Limits: The team may hold data that were not supplied. Aggregate topic or sentiment proportions from 7,850 sentences cannot be assigned to county–months. Strategy start-up dates are not CTH phase timelines.
