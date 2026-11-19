# Experimental RAG attendance completion

This is a separate AI-unreviewed research experiment. It does not create human
approvals, edit an original workbook, or establish a validated H3 effect.

## Website workflow

1. Use the private **Workflow → H3 source reconstruction** route. Upload the
   original Tracker, monthly CDC and annual CDC tables, dictionary and attendance
   CSVs. Rebuild the source inputs.
2. Run the documented no-change comparison if that is the chosen fixed baseline.
3. Upload meeting PDFs and retrieve attendance evidence for the current gaps.
4. Open **Experimental RAG + local LLM completion**. Select the fixed baseline,
   opt in to a separate unreviewed dataset and run the experiment.
5. Inspect candidate and unknown cases, raw model outputs, mechanical check
   failures, model settings and the unchanged-model comparison. Download the
   **AI-unreviewed RAG experiment** ZIP.

This feature is absent from public/synthetic-only sessions. Existing human-review
controls remain separate. Research external-API generation remains disabled.

## Actual methods

ENVIO's existing BM25 tokenizer and ranking retrieve three passages per query
within the target county, month and filename-derived meeting date. Two fixed
queries are fused by reciprocal rank; at most three original passages enter each
meeting extraction. The complete original-source memory and query log remain
separate from generated outputs. Cases without matching evidence remain unknown
without an LLM call. This is not a learned or self-updating neural memory.

The fixed model is `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The weights hash is checked before
loading. Generation runs on the application server CPU with bfloat16 weights,
greedy decoding, seed 20261119, a 2,048-token input limit and a 220-token output
limit. Model assets may download from the public Hugging Face repository. Only
fixed public artifact identifiers are sent for download; research text goes to
the in-process model, not an external inference API. No model training occurs.

The model must distinguish coalition headcounts from speakers, membership, poll
responses, patients, kits and other-event counts. Automatic checks require a
complete typed JSON response, a nonnegative integer count, same-date/county
source scope, an exact source quotation and an explicit count token. Approximate
wording is rejected. These checks do not prove semantic correctness or complete
meeting coverage. Both false candidates and abstention are possible.

A monthly candidate requires an eligible AI count for every meeting represented
in that case's supplied PDFs. Their mean uses the existing attendance unit.
Only previously missing attendance is filled in a separate derived copy. The
result is labelled `AI_unreviewed_candidate`, human confirmation remains false,
and confirmatory eligibility remains false. Original observed attendance and
all outcome values are preserved.

## Comparison and interpretation

The selected baseline is fixed before generation. Using the documented no-change
baseline holds the documented implementation reconstruction constant;
the new comparison isolates AI attendance additions. Both fits use the existing
OLS and interval-Poisson pathway, lags, county effects, trend, offset and
delete-county inference. The current model specification is not searched for a
significant result.

The ZIP contains original-source memory, retrieval log, raw model outputs,
candidate-level provenance, a runtime manifest, separately labelled derived data,
both statistical reports and their comparison. The manifest distinguishes model
calls, eligible candidates, rejected outputs, unresolved cases and experimental
additions. Zero eligible candidates means zero additions and unchanged inputs;
it is not a successful missing-value recovery claim. No accuracy, error-rate or
time-saving metric can be established without independent reference labels.

## Software and model checks

Synthetic software tests cover known-value preservation, unknown retention,
date/source isolation, invalid count/quote rejection and stale/tampered results.
Test fakes are explicitly labelled as software tests and are not empirical LLM
runs. A separate real-model synthetic smoke test checks live inference before
deployment. Real research results are generated through the private website and
remain excluded from the public repository.

Model reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
