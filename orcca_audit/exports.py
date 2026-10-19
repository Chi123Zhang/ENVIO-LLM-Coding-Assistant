"""In-memory, session-scoped exports; raw uploads are never included by default."""
from io import BytesIO
import json
import math
import zipfile

import pandas as pd


def safe_csv(rows):
    frame = pd.DataFrame(rows)
    # Stop spreadsheet formula execution from source-controlled strings.
    def safe(value):
        return "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
    for name in frame.columns:
        frame[name] = frame[name].map(safe)
    frame.columns = [safe(str(c)) for c in frame.columns]
    return frame.to_csv(index=False).encode("utf-8-sig")


def report_bundle(results, classification, provenance=None):
    if "research_findings" in results:
        from .english_export import english_results
        results = english_results(results)
    report = {"schema_version": "1.0", "data_classification": classification,
        "evidence_levels": ["verified_against_supplied_sources", "candidate_replay_only", "model_generated_unreviewed", "unknown"],
        "model_reproduced": False, "rag_accuracy_measured": False, "human_evaluation": "not_run",
        "provenance": provenance or {}, "results": results}
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if hasattr(value, "item"):
            return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    payload = json.dumps(clean(report), ensure_ascii=False, indent=2, default=str, allow_nan=False)
    target = BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("audit_report.json", payload)
        if "research_findings" in results:
            from .presentation import readable_report
            markdown, html, index = readable_report(results, classification, provenance or {})
            archive.writestr("audit_report.md", markdown)
            archive.writestr("START_HERE.html", html)
            archive.writestr("evidence_index.csv", safe_csv(index))
            archive.writestr("README.txt", "START_HERE.html: formatted English reading copy; open in a browser.\n"
                "audit_report.md: English text report.\nevidence_index.csv: compact source-location table.\n"
                "evidence/: English evidence text. Translated records are explicitly marked in the JSON/CSV, with original and translated text hashes.\n"
                "Other CSV and JSON files: technical exports; long source text may not display well in spreadsheets.\n"
                "Original study files are unchanged. Private exports remain restricted to authorized researchers.\n")
            for i, doc in enumerate(results.get("evidence_excerpts", []), 1):
                text = doc["text"]
                if doc.get("text_version"):
                    text = (doc["text_version"] + "\nOriginal text SHA-256: " + doc["original_text_sha256"] +
                        "\nEnglish text SHA-256: " + doc["sha256"] + "\n\n" + text)
                archive.writestr(f"evidence/{i:03d}.txt", text)
        else:
            archive.writestr("audit_report.md", "# Research audit\n\nData: " + classification +
                "\n\nChecks compare supplied sources. Candidate matches do not reproduce a model. "
                "No empirical RAG performance has been measured.\n\n```json\n" + payload + "\n```\n")
        for section, result in results.items():
            if isinstance(result, dict):
                for name, rows in result.items():
                    if isinstance(rows, list) and rows:
                        archive.writestr(f"{section}_{name}.csv", safe_csv(rows))
            elif isinstance(result, list) and result:
                archive.writestr(f"{section}.csv", safe_csv(result))
    return target.getvalue()
