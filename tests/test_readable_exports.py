"""Export layout/integrity checks, not empirical research evaluation."""
import csv
from io import BytesIO, StringIO
import json
import zipfile

from orcca_audit.exports import report_bundle
from orcca_audit.findings import load_bundle
from orcca_audit.public_review import bundled_public_review


def test_readable_export_preserves_records_and_separates_full_text():
    path, digest = bundled_public_review()
    bundle, _ = load_bundle(path, approved_sha256=digest)
    results = {"research_findings": bundle["findings"], "hypotheses": bundle["hypotheses"],
        "evidence_excerpts": bundle["documents"], "sources": bundle["sources"]}
    with zipfile.ZipFile(BytesIO(report_bundle(results, bundle["classification"]))) as z:
        assert json.loads(z.read("audit_report.json"))["results"] == results
        md = z.read("audit_report.md").decode()
        html = z.read("START_HERE.html").decode()
        assert "```json" not in md and "H2a has not been reproduced" in md
        assert '<table>' in html and 'lang="en"' in html
        rows = list(csv.DictReader(StringIO(z.read("evidence_index.csv").decode("utf-8-sig"))))
        assert len(rows) == 5 and "text" not in rows[0]
        for row, doc in zip(rows, bundle["documents"]):
            assert z.read(row["Full text file"]).decode() == doc["text"]
            assert row["Evidence ID"] == doc["chunk_id"]
            assert doc["source_file"] in md
        assert "evidence_excerpts.csv" in z.namelist()


def test_html_escapes_content_and_preserves_source_language():
    from orcca_audit.presentation import readable_report
    f = {"finding_id": "TEST", "title": "<script>test</script>", "level": "method_review",
        "statement": "Synthetic finding", "verification_method": "Unit test", "limitations": "Not research", "evidence_ids": ["d"]}
    doc = {"chunk_id": "d", "source_file": "synthetic.csv", "text": "合成原文", "locator": "原文定位", "line_start": 1, "line_end": 2}
    md, html, index = readable_report({"research_findings": [f], "evidence_excerpts": [doc]}, "private_research", {})
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "原文定位" not in html and "合成原文" not in md
    assert index[0]["First line or row"] == 1
