"""Readable research-review exports. Source quotations remain unmodified."""
from html import escape

from .findings import LEVELS


def evidence_index(documents):
    """Keep long quotations out of the presentation table; retain exact IDs."""
    return [{"Evidence ID": d["chunk_id"], "Source file": d["source_file"],
        "Page": d.get("page"), "First line or row": d.get("line_start"),
        "Last line or row": d.get("line_end"),
        "Evidence type": {"original_excerpt": "Original source excerpt",
            "deterministic_receipt": "Calculation record", "approved_summary": "Approved secondary summary"}.get(d.get("evidence_kind"), "Source record"),
        "Text version": d.get("text_version", "Untranslated source text"),
        "Full text file": f"evidence/{i:03d}.txt"}
        for i, d in enumerate(documents, 1)]


def readable_report(results, classification, provenance):
    findings = results["research_findings"]
    docs = results.get("evidence_excerpts", [])
    index = evidence_index(docs)
    by_id = {d["chunk_id"]: row for d, row in zip(docs, index)}
    scope = ("Approved aggregate summaries only. Original study documents are not published."
        if classification == "approved_public_summary" else
        "Synthetic software practice only. These are not study findings."
        if classification.startswith("synthetic") else
        "Private research review. This package contains source excerpts; share only with authorized researchers.")
    intro = ["ORCCA research review", scope,
        "These are data and methods checks. H2a has not been reproduced; H3–H5 have not been tested. Retrieval accuracy and time savings have not been measured."]
    date_label = "Generated" if provenance.get("run_id") else "Reviewed"
    md = ["# " + intro[0], "", intro[1], "", intro[2], "",
        date_label + ": " + str(provenance.get("review_date") or "Not specified"), ""]
    html = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>ORCCA research review</title>',
        '<style>body{font:16px/1.6 Arial,sans-serif;color:#243247;max-width:1050px;margin:40px auto;padding:0 24px}h1,h2{line-height:1.3}h2{margin-top:36px}p{max-width:90ch}.scope{background:#eef4fa;padding:16px}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #dce3eb;overflow-wrap:anywhere}th{background:#eef4fa}th:first-child{width:9%}.status{color:#52667c;font-size:14px}.source{font-size:14px}a{color:#245b91}@media print{body{margin:0;max-width:none}article{break-inside:avoid}}</style>',
        '<h1>ORCCA research review</h1>', '<p class="scope">' + escape(scope) + '</p>',
        '<p>' + escape(intro[2]) + '</p>',
        '<p class="status">' + date_label + ': ' + escape(str(provenance.get("review_date") or "Not specified")) + '</p>',
        '<h2>Findings at a glance</h2><table><thead><tr><th>ID</th><th>Topic</th><th>Evidence status</th></tr></thead><tbody>']
    for f in findings:
        level = LEVELS.get(f["level"], f["level"].replace("_", " "))
        html.append('<tr><td>' + escape(f["finding_id"]) + '</td><td>' + escape(f["title"]) + '</td><td>' + escape(level) + '</td></tr>')
    html.append('</tbody></table>')
    for f in findings:
        title = f["finding_id"] + " · " + f["title"]
        level = LEVELS.get(f["level"], f["level"].replace("_", " "))
        md += ["## " + title, "", "Evidence status: " + level, ""]
        html += ['<article><h2>' + escape(title) + '</h2><p class="status">' + escape(level) + '</p>']
        if f.get("review_status"):
            status = "Review status: " + f["review_status"]
            md += [status, ""]
            html.append('<p class="status">' + escape(status) + '</p>')
        for label, key in (("Finding", "statement"), ("Basis", "verification_method"), ("Limits", "limitations")):
            md += ["**" + label + ":** " + str(f[key]), ""]
            html.append('<p><strong>' + label + ':</strong> ' + escape(str(f[key])) + '</p>')
        if f.get("human_review"):
            review = f["human_review"]
            for label, key in (("Reviewer", "reviewer"), ("Review reason", "reason"), ("Researcher note", "researcher_note")):
                if review.get(key):
                    md += ["**" + label + ":** " + review[key], ""]
                    html.append('<p><strong>' + label + ':</strong> ' + escape(review[key]) + '</p>')
        refs = []
        for eid in f.get("evidence_ids", []):
            row = by_id.get(eid)
            if not row:
                continue
            loc = (f"page {row['Page']}; " if row["Page"] is not None else "") + f"lines/rows {row['First line or row']}–{row['Last line or row']}"
            text = f"{row['Source file']} — {loc} — {eid}"
            md.append(f"- [{text}]({row['Full text file']})")
            refs.append('<li><a href="' + row['Full text file'] + '">' + escape(text) + '</a></li>')
        html.append('<p class="source"><strong>Supporting records</strong></p><ul class="source">' + ''.join(refs) + '</ul></article>')
        md.append("")
    md += ["## H3–H5: readiness and next steps", ""]
    html.append('<h2>H3–H5: readiness and next steps</h2>')
    for h in results.get("hypotheses", []):
        title = h["hypothesis"] + " · " + h["definition"]
        md += ["### " + title, ""]
        html.append('<h3>' + escape(title) + '</h3>')
        for label, key in (("Available", "available"), ("Missing", "missing"), ("Next step", "next_step")):
            md += ["**" + label + ":** " + str(h[key]), ""]
            html.append('<p><strong>' + label + ':</strong> ' + escape(str(h[key])) + '</p>')
    note = "Full English evidence text is in the evidence folder. Translated records are explicitly marked in the JSON and CSV, with separate original and translated text hashes. Original research files remain unchanged and local. The evidence index lists locations without embedding long passages in spreadsheet cells. JSON and CSV exports are included for technical review."
    md += ["## Reading the evidence", "", note, ""]
    html += ['<h2>Reading the evidence</h2><p>' + escape(note) + '</p></html>']
    return "\n".join(md), "\n".join(html), index
