from io import BytesIO
import json
import zipfile
from unittest.mock import Mock

import pytest

from orcca_audit.demo import demo_data, public_documents
from orcca_audit.engine import AuditError
from orcca_audit.exports import report_bundle, safe_csv
from orcca_audit.io import read_table, read_workbook, extract_document
from orcca_audit.policy import private_uploads_allowed, external_generation_allowed
from orcca_audit.retrieval import PUBLIC_QUESTIONS, retrieve, explain_with_external_model, verify_model_response


def test_tsv_named_xls_and_physical_line_provenance():
    frame = read_table(b'county\tmonth\tDeaths\nDEMO_A\t2024/01\tSuppressed\n', 'demo.xls')
    assert frame.attrs["actual_format"] == "TSV text"
    assert frame.iloc[0]["_source_row"] == 2
    frame = read_table(b'a,b\n"two\nlines",3\n', 'demo.csv')
    assert frame.iloc[0]["_source_row"] == 3


def test_malformed_and_duplicate_header_inputs_fail_visibly():
    for data in [b'\x00binary', b'a,a\n1,2\n', b'a,b\n1,2,3\n']:
        with pytest.raises(AuditError):
            read_table(data, 'input.csv')


def test_xlsx_is_read_without_extension_assumptions():
    import openpyxl
    b = BytesIO(); book = openpyxl.Workbook(); ws = book.active
    ws.append(["county", "month"]); ws.append(["DEMO_A", "2024-01"]); book.save(b)
    frame = read_table(b.getvalue(), 'unknown.bin')
    assert frame.attrs["actual_format"] == "OOXML workbook"
    assert frame.iloc[0]["county"] == "DEMO_A"


def test_duplicate_workbook_labels_require_explicit_positional_sheet():
    import openpyxl
    b = BytesIO(); book = openpyxl.Workbook(); ws = book.active; ws.title = "Lookup"
    ws.append(["label", "label"]); ws.append(["SYNTHETIC_ID", "DEMO_A"]); book.save(b)
    with pytest.raises(AuditError, match="Duplicate"):
        read_workbook(b.getvalue())
    frames = read_workbook(b.getvalue(), positional_sheets=("Lookup",))
    assert frames["Lookup"].iloc[0, 1] == "DEMO_A"


def test_public_upload_and_external_controls_are_server_side(monkeypatch):
    monkeypatch.delenv("ORCCA_DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("ORCCA_ALLOW_PRIVATE_UPLOADS", raising=False)
    monkeypatch.delenv("ORCCA_ENABLE_EXTERNAL_LLM", raising=False)
    assert not private_uploads_allowed()
    assert not external_generation_allowed(public_documents())
    monkeypatch.setenv("ORCCA_DEPLOYMENT_MODE", "local_private")
    assert not private_uploads_allowed()
    monkeypatch.setenv("ORCCA_ALLOW_PRIVATE_UPLOADS", "1")
    assert private_uploads_allowed()
    monkeypatch.setenv("ORCCA_ENABLE_EXTERNAL_LLM", "1")
    assert external_generation_allowed(public_documents())
    private = extract_document(b'Private local test fixture, not study data.', 'demo.txt')
    assert not external_generation_allowed(private)
    client = Mock()
    with pytest.raises(PermissionError):
        explain_with_external_model("private free-text query", {"status": "retrieved_evidence_unreviewed", "citations": public_documents()}, "test-model", client)
    client.chat.completions.create.assert_not_called()


def test_rag_uses_real_demo_source_and_rejects_wrong_quotes(monkeypatch):
    docs = public_documents()
    result = retrieve(PUBLIC_QUESTIONS[0], docs)
    assert result["status"] == "retrieved_evidence_unreviewed"
    cite = result["citations"][0]
    assert cite["text"] == docs[0]["text"] and cite["source_file"].endswith("PUBLIC_DEMO_METHODS.md")
    answer = {"status": "answered", "answer": "Synthetic explanation.", "citations": [{"chunk_id": cite["chunk_id"], "quote": "invented quotation"}]}
    assert verify_model_response(answer, result["citations"]) == "quote_mismatch"
    assert retrieve("zzzz_absent_token", docs)["status"] == "retrieval_failed"
    assert retrieve("", docs)["status"] == "unsupported_query_language_or_empty"
    assert retrieve("window", [])["status"] == "empty_index"
    from orcca_audit import hybrid
    monkeypatch.setattr(hybrid, "local_hybrid_retrieve", Mock(side_effect=RuntimeError("weights absent")))
    failed = retrieve("window", docs, "hybrid")
    assert failed["status"] == "hybrid_unavailable_no_fallback"
    assert not failed["citations"]


def test_reference_labels_cannot_be_indexed():
    docs = public_documents(); docs[0]["role"] = "human_label"
    with pytest.raises(AuditError, match="cannot be indexed"):
        retrieve("window", docs)


def test_external_demo_never_forwards_caller_metadata(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("ORCCA_ENABLE_EXTERNAL_LLM", "1")
    docs = public_documents()
    docs[0]["caller_private_metadata"] = "SYNTHETIC_PRIVATE_SENTINEL"
    client = Mock()
    response = {"status": "answered", "answer": "Synthetic explanation.", "citations": [
        {"chunk_id": docs[0]["chunk_id"], "quote": "This is a fictional software demonstration"}]}
    client.chat.completions.create.return_value = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))])
    result = explain_with_external_model(PUBLIC_QUESTIONS[0],
        {"status": "retrieved_evidence_unreviewed", "citations": docs}, "test-model", client)
    prompt = client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in prompt
    assert "caller_private_metadata" not in prompt
    assert result["status"] == "model_generated_unreviewed_citation_strings_verified"


def test_exports_are_memory_only_and_escape_spreadsheet_formulas():
    blob = report_bundle({"test": [{"status": "needs_review", "value": "=HYPERLINK(1)"}]}, "synthetic_public_demo")
    with zipfile.ZipFile(BytesIO(blob)) as z:
        assert "audit_report.md" in z.namelist()
        report = json.loads(z.read("audit_report.json"))
        assert not report["model_reproduced"] and not report["rag_accuracy_measured"]
        assert report["human_evaluation"] == "not_run"
        assert b"'=HYPERLINK" in z.read("test.csv")
