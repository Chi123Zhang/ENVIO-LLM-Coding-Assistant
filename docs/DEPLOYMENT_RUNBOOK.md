# Approval-gated deployment and rollback

Recorded main baseline: `f716bb4f93cc5854adb004d4b4e239be84cd1184`.
Per-file hashes are in `baseline_main.json`. Development branch:
`feature/orcca-research-audit`. Existing entry point remains `app.py`.

No production deployment or merge is authorized until the owner approves the
tested PR. Local review is equivalent to a test deployment for this delivery;
there is no new public test service or private-data deployment.

## Before approval

1. Run `python -m pytest -q` using Python 3.12 and the full existing requirements.
2. Run loopback Streamlit as documented in `ORCCA_AUDIT.md`; inspect both pages,
   synthetic retrieval, unknowns, failures and ZIP downloads. Keep private-mode
   processes loopback-only and upload only locally authorized sources.
3. Review the PR file list, baseline preservation test and CI outcome. Do not
   copy local study reports, workbooks, labels, keys or research results into it.
4. Obtain explicit approval to merge this PR and update the existing app.

## After explicit approval

1. Recheck main and the tested head; resolve changes on the feature branch if
   needed and rerun tests. Merge through the PR; record the merge SHA.
2. In Streamlit Community Cloud, inspect the existing app's repository, branch
   and entry point. Keep the current repository, `main`, and `app.py`. A tracked
   main deployment may rebuild automatically after merge. Reboot/redeploy only
   if required. Never set the two local-private upload flags in Cloud.
3. Default ORCCA needs no new secrets. Leave external generation disabled for
   the initial release. Optional synthetic-only generation requires the three
   server settings documented in `ORCCA_AUDIT.md`; configure secrets through
   Cloud, never Git. Do not ask for secrets in chat or copy values into logs.
4. Verify the existing public URL: ENVIO landing page and modes, ORCCA's five
   tabs, synthetic 18-row demo, source retrieval/citations, explicit unavailable
   hybrid state if weights are absent, and ZIP report download. Confirm no ORCCA
   upload controls. A live external-model smoke test requires authorized model
   settings and synthetic input; report separately from mocked regressions.
5. Record UTC time, deployed commit and actual checks in the release record.

## Failure / rollback

If verification fails, preserve logs without secrets and restore the known-good
app by reverting the merge via a new Git commit/PR, then rebuild the tracked main
app. Preserve existing secret settings. Verify the original Interview Coding
page again. Avoid reset/force-push or deleting original data. Baseline SHA and
hashes identify the pre-integration state; inspect intervening commits so rollback
does not discard unrelated work. A proposed rollback to production also requires
the owner's deployment authorization. Do not claim a successful release until
the deployed page is actually checked.

Official guidance checked for this run: [managing an app and GitHub updates](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app),
[rebooting an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/reboot-your-app).
The current app's Cloud settings were not changed or independently confirmed;
inspect its actual tracked branch before any approved release.
