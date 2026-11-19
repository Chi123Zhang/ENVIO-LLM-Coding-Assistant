# Real H3 research on the existing website

## Streamlit-managed team-private deployment

An alternative to a separate OIDC client is Streamlit Cloud's existing viewer
access control. This makes the entire app private, including the demonstration.
The operator must FIRST set Sharing to **Only specific people can view this app**
and verify that both the normal URL and the internal application path reject an
unauthenticated browser. Do not invite additional viewers without authorization.

Only after that verification, add the following noncredential configuration in
Cloud Secrets, preserving all existing values:

```toml
[orcca_platform_private]
enabled = true
visibility_verified = "only_specific_people"
app_host = "expression-4wfrtnmolvi6ksdsst2h3c.streamlit.app"
```

This is an operator-controlled deployment mode: the app does not query the Cloud
sharing API or independently attest the ACL at runtime. It trusts the verified
upstream platform boundary and matches the hosted HTTPS URL. Do not enable it on
a public deployment. Before making the app public again, disable this mode,
restart and clear research sessions, then verify the upload controls are closed.
Research upload contents and generation restrictions are unchanged. No Google
Cloud client registration is required for this route.

## Optional app-level OIDC route

The public URL can retain its synthetic demonstration while authenticated,
allowlisted researchers use the original-source upload controls in their own
server sessions. This code is disabled by default. Enabling it does not establish
institutional approval of the hosting environment.

## Operator setup (not research data)

Use Streamlit's native OIDC login with a team-controlled identity provider.
Google's verified-email claims are supported by the strict issuer/email gate.
Register the existing production redirect URI with the provider:

`https://expression-4wfrtnmolvi6ksdsst2h3c.streamlit.app/oauth2callback`

Keep the client credentials and a strong random cookie secret only in Streamlit
Cloud Secrets. Do not commit them or paste them into chat. Configure the existing
Secrets without replacing unrelated ENVIO settings:

```toml
[auth]
redirect_uri = "https://expression-4wfrtnmolvi6ksdsst2h3c.streamlit.app/oauth2callback"
cookie_secret = "OPERATOR_SUPPLIES_A_STRONG_RANDOM_VALUE"
client_id = "OPERATOR_CONFIGURES_PROVIDER_CLIENT_ID"
client_secret = "OPERATOR_CONFIGURES_PROVIDER_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
expose_tokens = false

[orcca_research]
enabled = true
issuer = "https://accounts.google.com"
allowed_emails = ["OPERATOR_CONFIGURES_EXACT_APPROVED_RESEARCHER_EMAIL"]
```

These are placeholders, not usable credentials. Require Streamlit with native
st.login/st.user support and Authlib. For other identity providers, verify their
claims before adapting the gate; do not bypass verified-email or expiry checks.
Provider sign-in may require opening the app directly rather than embedded.

Authentication setup requires the operator to complete provider registration,
credential configuration and any consent prompts. A mock login or AppTest is not
an actual cloud identity test. Keep research uploads closed until an allowlisted
identity succeeds and an unauthorized identity is rejected.

## Researcher flow after deployment and authentication

1. Workflow → Real research access → Sign in for research access.
2. H3 source reconstruction → Private research files.
3. Upload the original Tracker, its REDCap dictionary, monthly and annual CDC
   tables, and the seven available county attendance CSVs. Do not upload panel.csv.
4. Keep Use the entire supplied CDC period selected and rebuild. Check coverage;
   a full source grid does not imply complete analysis records.
5. Run H3 tests on current source data. These are explicitly exploratory models.
6. Run documented no-change comparison as a separate dictionary-based sensitivity
   analysis. Its reconstructed mediator states are not human-validated AI fills.
7. Upload county/date-labelled meeting PDFs; retrieve evidence for attendance gaps.
   Original text, page references and source/run hashes go into source memory.
8. Download the independent reviewer form. Retain unknowns when evidence does not
   establish an exact comparable attendance measure. AI responses are not human labels.
9. Upload independently reviewed decisions; compare approved additions against
   the same original model specification and download both reports and the ledger.
10. Sign out and clear this session. Downloads remain under the researcher's control.

Uploaded source files and reports are held in the current session, not committed
to GitHub or published as demonstration assets. Processing still occurs on the
Streamlit hosting server; this is not local-only or a claim of institutional
compliance. Research evidence is not eligible for external LLM generation.
The original ENVIO text-coding page is not the real H3 upload entry.

## Scientific scope

RAG retrieves evidence; deterministic rules and human decisions have separate
provenance. Source memory stores original evidence rather than training a model.
No recovery accuracy improvement is assumed. H3 results require endpoint and
design review, effect estimates, intervals and diagnostics; significance is not
a success condition. The new source-analysis reports are not yet automatically
integrated into the older Findings approval/version workflow.

References: https://docs.streamlit.io/develop/api-reference/user/st.login and
https://docs.streamlit.io/develop/api-reference/user/st.user
