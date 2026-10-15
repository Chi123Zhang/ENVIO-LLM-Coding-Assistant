"""Real FAISS/original retrieve algorithm; fake weights, no quality claims."""
import hashlib
import numpy as np

from orcca_audit import hybrid


def test_small_corpus_has_no_faiss_padding_duplicates(monkeypatch):
    class Embedder:
        def encode(self, texts, **kwargs):
            return np.array([[1., 0.] if "window" in t else [0., 1.] for t in texts], dtype="float32")

    class Reranker:
        pairs = []

        def predict(self, pairs):
            self.pairs = pairs
            return np.array([2. if "window" in text else 1. for _, text in pairs])

    reranker = Reranker()
    monkeypatch.setattr(hybrid, "local_models", lambda: (Embedder(), reranker))
    docs = [{"chunk_id": str(i), "source_file": "synthetic.txt", "page": None,
             "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
            for i, text in enumerate(["Synthetic window protocol", "Synthetic source definition"])]
    hits = hybrid.local_hybrid_retrieve("window", docs, 3)
    assert [d["chunk_id"] for d, _ in hits] == ["0", "1"]
    assert len(reranker.pairs) == 2
