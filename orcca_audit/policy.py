"""Server-side controls. A browser checkbox cannot approve private transmission."""
import hashlib
import os


def private_uploads_allowed():
    return os.environ.get("ORCCA_DEPLOYMENT_MODE", "public_demo") == "local_private" and os.environ.get("ORCCA_ALLOW_PRIVATE_UPLOADS") == "1"


def external_generation_allowed(documents):
    # Only exact, bundled, public synthetic evidence can leave this module.
    from .demo import public_documents
    approved = {d["sha256"] for d in public_documents()}
    return (os.environ.get("ORCCA_ENABLE_EXTERNAL_LLM") == "1" and bool(documents)
        and all(d.get("classification") == "synthetic_public" and d.get("sha256") in approved
            and hashlib.sha256(d["text"].encode()).hexdigest() == d.get("sha256") for d in documents))
