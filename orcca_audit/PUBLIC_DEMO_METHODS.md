# Synthetic audit protocol

This is a fictional software demonstration, not ORCCA research evidence.
DEMO_A, DEMO_B and DEMO_C are invented identifiers. Every count is synthetic.

Protocol A uses the previous three months of active-entry counts as exposure.
Protocol A uses a single outcome month and reports raw exposure units.
Protocol B uses the same previous three-month exposure window.
Protocol B uses a subsequent three-month outcome sum; log/standardization details are unspecified.

Candidate active entries have started=1 and are not marked discontinued=1.
An unknown discontinuation is not explicit No; report a review flag and a count bound.
An explicitly skipped snapshot may carry the preceding snapshot, but an absent event is not no-change.
Annual compatibility includes all twelve source months, including those outside the analysis window.
Matching candidate lag arithmetic does not verify the original model implementation.
No H2a model is fitted or reproduced by this demonstration.
No human reference labels or measured RAG performance are supplied.
