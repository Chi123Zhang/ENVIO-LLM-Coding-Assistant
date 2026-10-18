"""Validate source-linked research reviews without bundling study materials.

Private bundles are local operator inputs. Public bundles additionally require
an approved classification and an operator-pinned hash of their exact bytes.
No generation, model fitting, or network client is used here.
"""
import hashlib
import json
from pathlib import Path

from .engine import AuditError
from .retrieval import validate_documents

LEVELS = {
    "verified_source_check": "Checked against supplied sources",
    "candidate_rule_match": "Candidate rule match — review needed",
    "method_review": "Methods review — confirmation needed",
    "insufficient_material": "Materials incomplete",
}
CLASSIFICATIONS = {"private_research", "publication_draft", "approved_public_summary", "synthetic_test"}


def validate_bundle(bundle):
    if not isinstance(bundle, dict) or bundle.get("schema_version") != "1.0":
        raise AuditError("Unsupported research findings schema.")
    if bundle.get("classification") not in CLASSIFICATIONS:
        raise AuditError("Research findings need an explicit data classification.")
    for field in ["findings", "documents", "sources", "questions", "hypotheses"]:
        if not isinstance(bundle.get(field), list):
            raise AuditError(f"Research bundle needs a {field} list.")
    if not bundle["findings"] or not bundle["documents"]:
        raise AuditError("A research review needs findings and source evidence.")
    try:
        validate_documents(bundle["documents"])
    except AuditError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError("Invalid research evidence schema.") from exc
    docs = {d["chunk_id"]: d for d in bundle["documents"]}
    sources = {s.get("source_id"): s for s in bundle["sources"] if isinstance(s, dict)}
    if None in sources or len(sources) != len(bundle["sources"]):
        raise AuditError("Research source IDs must be unique and present.")
    for doc in docs.values():
        if doc.get("source_id") not in sources:
            raise AuditError("Every evidence excerpt needs an actual source record.")
        if not isinstance(doc.get("line_end"), int) or doc["line_end"] < doc["line_start"]:
            raise AuditError("Invalid evidence source range.")
        if doc.get("evidence_kind") not in {"original_excerpt", "deterministic_receipt", "approved_summary"}:
            raise AuditError("Evidence must distinguish original text, calculations and summaries.")
    ids = set()
    for finding in bundle["findings"]:
        if not isinstance(finding, dict) or not finding.get("finding_id") or finding["finding_id"] in ids:
            raise AuditError("Finding IDs must be unique and present.")
        ids.add(finding["finding_id"])
        if finding.get("level") not in LEVELS:
            raise AuditError("A finding needs an explicit evidence level.")
        for field in ["title", "statement", "verification_method", "limitations"]:
            if not isinstance(finding.get(field), str) or not finding[field].strip():
                raise AuditError(f"Finding missing {field}.")
        refs = finding.get("evidence_ids")
        if not isinstance(refs, list) or not refs or any(ref not in docs for ref in refs):
            raise AuditError("Findings cannot cite absent evidence.")
        if finding.get("model_reproduced") is True:
            raise AuditError("This review cannot assert model reproduction.")
    question_ids = set()
    for q in bundle["questions"]:
        if not isinstance(q, dict) or not q.get("question_id") or q["question_id"] in question_ids:
            raise AuditError("Question IDs must be unique and present.")
        question_ids.add(q["question_id"])
        if q.get("status") not in {"answerable", "partial", "insufficient_evidence"}:
            raise AuditError("Unknown question answerability.")
        if any(ref not in ids for ref in q.get("finding_ids", [])):
            raise AuditError("Question refers to an absent finding.")
        if not q.get("question") or not q.get("query") or not q.get("review_answer"):
            raise AuditError("Question needs a query and a clearly authored review answer.")
    if bundle.get("evaluation", {}).get("improvement_measured") is not False:
        raise AuditError("No measured RAG improvement is established by this module.")
    return bundle


def load_bundle(path, *, private_mode=False, approved_sha256=None):
    source = Path(path)
    if source.stat().st_size > 20 * 1024 * 1024:
        raise AuditError("Research findings bundle exceeds 20 MB.")
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        bundle = validate_bundle(json.loads(payload))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError("Research findings must be valid UTF-8 JSON.") from exc
    if not private_mode:
        publication = bundle.get("publication", {})
        if (bundle["classification"] != "approved_public_summary" or
                not approved_sha256 or digest != approved_sha256 or
                publication.get("status") != "approved" or not publication.get("approval_reference")):
            raise AuditError("Public research findings require approval and the exact approved file hash.")
        if any(d.get("classification") != "approved_public_excerpt" or d.get("evidence_kind") != "approved_summary"
               for d in bundle["documents"]):
            raise AuditError("Public findings cannot contain private original-source excerpts.")
    return bundle, digest


def retrieval_fingerprint(bundle_hash, query, backend):
    return hashlib.sha256(json.dumps([bundle_hash, query, backend], ensure_ascii=False).encode()).hexdigest()


def evidence_for(finding, bundle):
    docs = {d["chunk_id"]: d for d in bundle["documents"]}
    return [docs[key] for key in finding["evidence_ids"]]
