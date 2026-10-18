"""Synthetic schema/UI fixtures only. No study findings or source data."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from orcca_audit.engine import AuditError
from orcca_audit.findings import evidence_for, load_bundle, retrieval_fingerprint, validate_bundle

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    text = "SYNTHETIC protocol. An unknown status requires source review."
    return {
        "schema_version": "1.0", "classification": "synthetic_test", "review_date": "synthetic",
        "sources": [{"source_id": "s", "source_file": "synthetic_protocol.txt"}],
        "documents": [{"chunk_id": "d", "source_id": "s", "source_file": "synthetic_protocol.txt",
            "page": None, "line_start": 1, "line_end": 1, "locator": "synthetic line 1",
            "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "evidence_kind": "original_excerpt", "classification": "private_local"}],
        "findings": [{"finding_id": "F", "title": "SYNTHETIC source review", "statement": "SYNTHETIC status remains unknown.",
            "verification_method": "Read a synthetic software fixture.", "limitations": "Not research evidence.",
            "level": "candidate_rule_match", "evidence_ids": ["d"]}],
        "questions": [{"question_id": "Q", "question": "Synthetic unknown status?", "query": "unknown status",
            "review_answer": "SYNTHETIC fixture needs review.", "status": "partial", "finding_ids": ["F"]}],
        "hypotheses": [], "evaluation": {"human_labelled_cases": 0, "improvement_measured": False},
    }


def write_fixture(tmp_path, bundle):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def test_private_bundle_requires_full_source_chain(tmp_path):
    bundle = fixture()
    loaded, digest = load_bundle(write_fixture(tmp_path, bundle), private_mode=True)
    assert evidence_for(loaded["findings"][0], loaded)[0]["text"] == bundle["documents"][0]["text"]
    assert len(digest) == 64
    broken = copy.deepcopy(bundle)
    broken["findings"][0]["evidence_ids"] = ["absent"]
    with pytest.raises(AuditError, match="absent evidence"):
        validate_bundle(broken)
    broken = copy.deepcopy(bundle)
    broken["documents"][0]["source_id"] = "absent"
    with pytest.raises(AuditError, match="actual source"):
        validate_bundle(broken)


def test_no_fabricated_citations_or_label_indexing():
    bundle = fixture()
    bundle["documents"][0]["text"] += " unapproved change"
    with pytest.raises(AuditError, match="hash mismatch"):
        validate_bundle(bundle)
    bundle = fixture()
    bundle["documents"][0]["role"] = "human_label"
    with pytest.raises(AuditError, match="cannot be indexed"):
        validate_bundle(bundle)
    bundle = fixture()
    bundle["findings"][0]["model_reproduced"] = True
    with pytest.raises(AuditError, match="model reproduction"):
        validate_bundle(bundle)


def test_public_requires_approved_exact_file_and_no_private_excerpts(tmp_path):
    bundle = fixture()
    with pytest.raises(AuditError, match="approval"):
        load_bundle(write_fixture(tmp_path, bundle))
    bundle["classification"] = "approved_public_summary"
    bundle["publication"] = {"status": "approved", "approval_reference": "synthetic software fixture only"}
    path = write_fixture(tmp_path, bundle)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(AuditError, match="private original"):
        load_bundle(path, approved_sha256=digest)
    bundle["documents"][0].update(classification="approved_public_excerpt", evidence_kind="approved_summary")
    path = write_fixture(tmp_path, bundle)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_bundle(path, approved_sha256=digest)[0]["classification"] == "approved_public_summary"
    with pytest.raises(AuditError, match="approval"):
        load_bundle(path, approved_sha256="wrong")
    bundle["publication"]["status"] = "pending_owner_approval"
    path = write_fixture(tmp_path, bundle)
    with pytest.raises(AuditError, match="approval"):
        load_bundle(path, approved_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_fingerprints_change_with_input_query_or_backend():
    original = retrieval_fingerprint("a", "query", "bm25")
    assert len({original, retrieval_fingerprint("b", "query", "bm25"),
        retrieval_fingerprint("a", "changed", "bm25"), retrieval_fingerprint("a", "query", "hybrid")}) == 4


def test_public_page_does_not_substitute_demo_or_show_private_bundle(monkeypatch, tmp_path):
    monkeypatch.delenv("ORCCA_DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("ORCCA_ALLOW_PRIVATE_UPLOADS", raising=False)
    monkeypatch.delenv("ORCCA_FINDINGS_BUNDLE", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.switch_page("pages/02_ORCCA_Research_Findings.py").run()
    assert not at.exception
    assert any("Demo results are not used" in x.value for x in at.info)
    assert not at.get("file_uploader") and not at.tabs
    monkeypatch.setenv("ORCCA_FINDINGS_BUNDLE", str(write_fixture(tmp_path, fixture())))
    at.run()
    assert not at.exception
    assert any("approval" in x.value for x in at.error)
    assert not at.tabs


def test_findings_page_retrieves_raw_evidence_and_invalidates_stale_results(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCCA_DEPLOYMENT_MODE", "local_private")
    monkeypatch.setenv("ORCCA_ALLOW_PRIVATE_UPLOADS", "1")
    bundle = fixture()
    path = write_fixture(tmp_path, bundle)
    monkeypatch.setenv("ORCCA_FINDINGS_BUNDLE", str(path))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.switch_page("pages/02_ORCCA_Research_Findings.py").run()
    assert not at.exception and len(at.tabs) == 5
    at.button(key="findings_retrieve").click().run()
    assert any("Supporting records found" in str(x.value) for x in at.markdown)
    assert any("SYNTHETIC protocol" in x.value for x in at.code)
    assert not any("Generate" in b.label for b in at.button)
    bundle["documents"][0]["text"] = "SYNTHETIC replacement protocol."
    bundle["documents"][0]["sha256"] = hashlib.sha256(bundle["documents"][0]["text"].encode()).hexdigest()
    write_fixture(tmp_path, bundle)
    at.run()
    assert not any("Search status" in str(x.value) for x in at.markdown)
    at.text_input(key="findings_query_Q").set_value("replacement").run()
    at.button(key="findings_retrieve").click().run()
    assert "findings_result" in at.session_state
    at.selectbox(key="findings_backend").select("hybrid").run()
    at.session_state["envio_reset_sentinel"] = "preserve unrelated state"
    at.button(key="findings_reset").click().run()
    assert not at.exception
    assert "findings_result" not in at.session_state
    assert "findings_fingerprint" not in at.session_state
    assert at.text_input(key="findings_query_Q").value == "unknown status"
    assert at.selectbox(key="findings_backend").value == "bm25"
    assert at.session_state["envio_reset_sentinel"] == "preserve unrelated state"
    assert any("Search reset" in x.value for x in at.success)
    assert any("SYNTHETIC status remains unknown" in x.value for x in at.markdown)
    assert not any("Search status" in str(x.value) for x in at.markdown)
    at.switch_page("app.py").run()
    assert not at.exception and at.title[0].value == "ENVIO LLM Coding Assistant"
