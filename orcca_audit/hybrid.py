"""In-memory adapter to ENVIO's FAISS/BM25/reranker implementation.

Public mode may fetch only fixed public model assets, never send evidence.
Private mode remains offline. Inference always runs in the local process.
The shared ENVIO index is never loaded, written, or mixed with audit documents.
"""
from functools import lru_cache
import os

MODEL_REVISIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
}


def model_downloads_allowed():
    return (os.environ.get("ORCCA_DEPLOYMENT_MODE", "public_demo") == "public_demo"
        and os.environ.get("ORCCA_ALLOW_MODEL_DOWNLOADS", "1") == "1")


class _BoundedIndex:
    """Avoid FAISS -1 padding when the audit corpus is smaller than ENVIO's k."""

    def __init__(self, index):
        self.index = index

    def search(self, vectors, k):
        return self.index.search(vectors, min(k, self.index.ntotal))


def local_models():
    return _models(model_downloads_allowed())


@lru_cache(maxsize=2)
def _models(allow_downloads):
    from rag_system import EMBED_MODEL_NAME, RERANKER_MODEL_NAME
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from huggingface_hub import snapshot_download
    def assets(name):
        return snapshot_download(repo_id=name, revision=MODEL_REVISIONS[name],
            local_files_only=not allow_downloads, token=False, max_workers=2,
            allow_patterns=["*.json", "model.safetensors", "vocab.txt", "1_Pooling/config.json"])
    embed_path, rerank_path = assets(EMBED_MODEL_NAME), assets(RERANKER_MODEL_NAME)
    return (SentenceTransformer(embed_path, device="cpu", local_files_only=True, trust_remote_code=False),
            CrossEncoder(rerank_path, device="cpu", local_files_only=True, trust_remote_code=False))


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
    embeddings = embedder.encode(rag.chunk_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=16).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    rag.index = _BoundedIndex(index)
    by_id = {d["chunk_id"]: d for d in documents}
    return [(by_id[c.chunk_id], score) for c, score in rag.retrieve(query, top_k=top_k)]


def rrf_retrieve(query, documents, top_k, index_cache):
    """Fuse scoped lexical and dense rankings, then cross-encode 16 candidates."""
    import hashlib
    import json
    import numpy as np
    from rag_system import BM25Okapi, tokenize_for_bm25
    embedder, reranker = local_models()
    key = hashlib.sha256(json.dumps([MODEL_REVISIONS,
        [(d['chunk_id'],d['sha256']) for d in documents]],sort_keys=True).encode()).hexdigest()
    reused = key in index_cache
    if not reused:
        texts = [d['text'] for d in documents]
        matrix = embedder.encode(texts,convert_to_numpy=True,normalize_embeddings=True,batch_size=16).astype('float32')
        bm25 = BM25Okapi([tokenize_for_bm25(t) for t in texts])
        # Session-owned evidence index. Bound storage by corpus, never by user-global cache.
        if len(index_cache)>=24:
            index_cache.pop(next(iter(index_cache)))
        index_cache[key]=(matrix,bm25)
    matrix,bm25 = index_cache[key]
    vector = embedder.encode([query],convert_to_numpy=True,normalize_embeddings=True).astype('float32')[0]
    lexical = bm25.get_scores(tokenize_for_bm25(query))
    dense = matrix @ vector
    rankings = [np.argsort(-lexical,kind='stable')[:20],np.argsort(-dense,kind='stable')[:20]]
    scores = {}
    for ranking in rankings:
        for rank,index in enumerate(ranking,1):
            scores[int(index)] = scores.get(int(index),0) + 1/(60+rank)
    selected = sorted(scores,key=lambda i:(-scores[i],documents[i]['chunk_id']))[:16]
    reranked = reranker.predict([(query,documents[i]['text']) for i in selected],batch_size=16)
    order = sorted(zip(selected,reranked),key=lambda pair:(-float(pair[1]),documents[pair[0]]['chunk_id']))[:top_k]
    return [(documents[i],float(score)) for i,score in order],reused
