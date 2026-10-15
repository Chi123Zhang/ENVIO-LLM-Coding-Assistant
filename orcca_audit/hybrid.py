"""In-memory adapter to ENVIO's FAISS/BM25/reranker implementation.

Weights must already exist locally; this module never downloads them at runtime.
The shared ENVIO index is never loaded, written, or mixed with audit documents.
"""
from functools import lru_cache


class _BoundedIndex:
    """Avoid FAISS -1 padding when the audit corpus is smaller than ENVIO's k."""

    def __init__(self, index):
        self.index = index

    def search(self, vectors, k):
        return self.index.search(vectors, min(k, self.index.ntotal))


@lru_cache(maxsize=1)
def local_models():
    from rag_system import EMBED_MODEL_NAME, RERANKER_MODEL_NAME
    from sentence_transformers import SentenceTransformer, CrossEncoder
    return (SentenceTransformer(EMBED_MODEL_NAME, device="cpu", local_files_only=True),
            CrossEncoder(RERANKER_MODEL_NAME, device="cpu", local_files_only=True))


def local_hybrid_retrieve(query, documents, top_k):
    import faiss
    from rag_system import TechMPowerRAG, Chunk, BM25Okapi, tokenize_for_bm25
    embedder, reranker = local_models()
    # Bypass only the constructor's eager weight download; reuse its retrieve()
    # algorithm with fresh, request-scoped evidence and no persistence methods.
    rag = TechMPowerRAG.__new__(TechMPowerRAG)
    rag.embed_model, rag.reranker = embedder, reranker
    rag.chunks = [Chunk(d["chunk_id"], d["text"], d["source_file"], "Audit evidence", d.get("page"),
        "Source evidence", "Research audit", "Document", "Local", True) for d in documents]
    rag.chunk_texts = [d["text"] for d in documents]
    rag.bm25_tokens = [tokenize_for_bm25(t) for t in rag.chunk_texts]
    rag.bm25 = BM25Okapi(rag.bm25_tokens)
    embeddings = embedder.encode(rag.chunk_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    rag.index = _BoundedIndex(index)
    by_id = {d["chunk_id"]: d for d in documents}
    return [(by_id[c.chunk_id], score) for c, score in rag.retrieve(query, top_k=top_k)]
