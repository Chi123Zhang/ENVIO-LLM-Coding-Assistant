# Local Qwen review demonstration

This extends the H3 workflow released in PR #4. All existing ENVIO and ORCCA pages remain available. The only existing source change is two lines in `orcca_audit/study_ui.py` to render an isolated demonstration in step 3.

## What to show

1. Open **ORCCA Workflow** and keep **H3 study preparation** selected.
2. In **1 · Project inputs**, click **Load synthetic H3 project**.
3. Open **3 · Evidence and completion** and expand **Local Qwen review · recorded synthetic demonstration**.
4. Click **Replay synthetic Qwen review**. ENVIO BM25 actually runs for three fictional questions. The model responses are recordings of a genuine local Qwen run, not fresh cloud generation.
5. Select a case. Read the question, proposed month or unknown, the source lines and model limitations. All responses remain unconfirmed. No project value or Finding changes.
6. Download **synthetic Qwen run** to inspect the model's raw JSON, input hashes, exact source lines and runtime settings.

The examples demonstrate a documented start, a plan without a confirmed start, and an unresolved strategy identity. They are handwritten fictional scenarios, not transformed private records. They are separate from the older 24-case masking pilot and are not a new evaluation dataset.

## What runs where

| Component | Execution |
| --- | --- |
| ENVIO BM25 | Runs in the Streamlit process on replay |
| Qwen2.5-7B-Instruct Q4_K_M generation | Ran on a local CPU; saved recording is bundled |
| Cloud Qwen generation | Not enabled or claimed |
| Hybrid retrieval | Existing independent evidence-search option, unchanged by this feature |
| Human labels or accepted corrections | None created by the demo |
| Source memory | Six fictional source lines; model outputs are excluded |

Matching citation strings establish where a quote came from, not whether the interpretation is correct. No scientific accuracy or time-saving claim follows from three examples. Actual study reviews stay outside the repository and public website.

## Reproduce the public recording locally

Use an environment containing the existing repository requirements and the optional `llama-cpp-python==0.3.36` package. The latter is deliberately **not** added to Streamlit's requirements: replay needs no GGUF weights, GPU or extra inference runtime.

```
python scripts/run_synthetic_model_review.py --model-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf --output /new/path/synthetic_qwen_review.json
```

The script accepts no research data, prompts or URLs. It uses only the hardcoded fictional fixture, existing ENVIO BM25 and local model weights. It refuses to overwrite an existing recording. It blocks Python socket connection/name-resolution attempts during generation. This is not an independently audited air-gap guarantee. CPU settings avoid Metal/GPU allocation. Model hash, prompt hash, source/input hashes, seed, tokens, runtime and raw responses are preserved.

Public mode keeps research uploads and ORCCA external generation disabled. This change does not alter ENVIO's original model configuration, upload controls or exports. No private inputs, annotations, model weights or secrets are included.

## Recording provenance clarification

The shipped three-case recording is a completed local run. Its first generator stored the hash of the final case-specific response grammar rather than the base grammar. The executed script is preserved byte-for-byte in `docs/provenance/recording_generator_v1.py.txt`, and the record explicitly labels that fingerprint scope. Raw replies, usage and settings are unchanged. The current reproduction runner fixes that metadata field. This is a provenance correction, not a model rerun or a change to the answers.
