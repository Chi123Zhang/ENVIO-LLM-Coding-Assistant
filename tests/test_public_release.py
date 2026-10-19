"""Public-content integrity and model-asset policy, not research performance."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from orcca_audit.findings import load_bundle
from orcca_audit.public_review import bundled_public_review
from orcca_audit import hybrid

ROOT = Path(__file__).resolve().parents[1]


def test_approved_public_summary_has_real_secondary_source_lines():
    path, sha = bundled_public_review()
    bundle, actual = load_bundle(path, approved_sha256=sha)
    assert actual == sha and len(bundle["findings"]) == len(bundle["documents"]) == 5
    lines = (ROOT / "docs/PUBLIC_RESEARCH_SUMMARY.md").read_text().splitlines()
    for doc in bundle["documents"]:
        assert doc["classification"] == "approved_public_excerpt"
        assert doc["evidence_kind"] == "approved_summary"
        assert doc["text"] == "\n".join(lines[doc["line_start"]-1:doc["line_end"]])
        assert doc["source_file"] == "docs/PUBLIC_RESEARCH_SUMMARY.md"
    assert bundle["evaluation"]["improvement_measured"] is False
    assert "24" not in str(bundle["evaluation"])


def test_public_findings_load_without_cloud_secrets(monkeypatch):
    for key in ("ORCCA_DEPLOYMENT_MODE", "ORCCA_ALLOW_PRIVATE_UPLOADS", "ORCCA_FINDINGS_BUNDLE", "ORCCA_PUBLIC_FINDINGS_SHA256"):
        monkeypatch.delenv(key, raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.switch_page("pages/02_ORCCA_Research_Findings.py").run()
    assert not at.exception and len(at.tabs) == 5
    assert any("Public summary" in x.value for x in at.info)
    assert not at.get("file_uploader")
    assert sum(x.value.startswith("F") for x in at.subheader) == 5
    assert not any("Generate" in x.label for x in at.button)


def test_only_public_fixed_assets_can_be_downloaded(monkeypatch):
    monkeypatch.delenv("ORCCA_DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("ORCCA_ALLOW_MODEL_DOWNLOADS", raising=False)
    assert hybrid.model_downloads_allowed()
    monkeypatch.setenv("ORCCA_DEPLOYMENT_MODE", "local_private")
    assert not hybrid.model_downloads_allowed()
    monkeypatch.setenv("ORCCA_DEPLOYMENT_MODE", "public_demo")
    monkeypatch.setenv("ORCCA_ALLOW_MODEL_DOWNLOADS", "0")
    assert not hybrid.model_downloads_allowed()


def test_model_loader_fetches_assets_then_uses_offline_inference(monkeypatch):
    import huggingface_hub
    import sentence_transformers
    calls, constructors = [], []
    def download(**kwargs):
        calls.append(kwargs)
        return "/synthetic/model/path"
    def model(path, **kwargs):
        constructors.append((path, kwargs))
        return object()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", model)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", model)
    hybrid._models.cache_clear()
    try:
        hybrid._models(True)
        assert len(calls) == 2
        for call in calls:
            assert call["revision"] == hybrid.MODEL_REVISIONS[call["repo_id"]]
            assert call["token"] is False and call["local_files_only"] is False
            assert "model.safetensors" in call["allow_patterns"]
        assert all(c[1]["local_files_only"] and c[1]["trust_remote_code"] is False for c in constructors)
        calls.clear()
        hybrid._models(False)
        assert all(call["local_files_only"] for call in calls)
    finally:
        hybrid._models.cache_clear()
