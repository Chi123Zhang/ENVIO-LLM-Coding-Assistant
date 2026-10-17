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
