# Public summaries and hybrid deployment

The project owner authorized publication of the five aggregate review summaries
and repair of cloud hybrid on 2026-10-20. `PUBLIC_RESEARCH_SUMMARY.md` is the
public secondary source. The matching JSON contains five summary chunks with
exact Markdown line references and excerpt hashes. Original tables, source
excerpts, data hashes, county-specific cases and human annotation cases are
excluded. The summaries retain their caveats; no effect or RAG performance is
claimed. Hypothesis-readiness notes describe general missing material only.

`orcca_audit/public_review.py` pins the exact approved JSON hash. Public Findings
loads this bundle by default without Secrets changes. An explicit operator path
still needs its own hash and approval. Private-mode inputs do not silently fall
back to the public bundle. The existing public loader rejects private excerpts,
pending approval and mismatched hashes. Generation stays disabled.

## Hybrid weights

The prior adapter required cached weights. A fresh Cloud machine has no such
cache, causing explicit `hybrid_unavailable_no_fallback`. Public mode now allows
asset downloads for only the existing ENVIO embedding and reranking models,
pinned to these revisions:

| Model | Revision |
|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | c9745ed1d9f207416be6d2e6f8de32d1f16199bf |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | c5ee24cb16019beea0893ab7796b1df96625c6b8 |

Model cards: [embedding](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
The legacy reranker identifier redirects to the canonical model name; ENVIO's
original model constants are unchanged.

Only JSON/tokenizer files and safetensors weights are fetched, without a Hub
token or remote model code. Model constructors then use local paths with
`local_files_only=True` and `trust_remote_code=False`. Embedding and reranking
execute in the application CPU; no query or evidence is sent to a hosted
inference endpoint. FAISS, BM25 and ENVIO's retrieval/reranking algorithm are
reused, with a fresh in-memory evidence index for each call. Loaded model objects
are process-cached. Reports record model revisions and inference location.

Private mode never downloads model assets. Operators may also disable public
downloads using `ORCCA_ALLOW_MODEL_DOWNLOADS=0`. An unsuccessful download, load
or retrieval remains an explicit failure, with no BM25 fallback or success
claim. First-use loading has a visible spinner; no weight files enter Git.

Software tests use mock weights only where labelled as unit tests. Real-model
local and Cloud runs must be recorded separately from unit tests and from
research accuracy evaluation. A successful hybrid run establishes execution,
not improved validity, precision, recall or time saving.
