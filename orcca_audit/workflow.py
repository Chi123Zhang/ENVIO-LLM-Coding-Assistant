"""Session project versions joining audit, evidence completion and findings.

No global state, persistence, external generation, effect fitting or raw overwrites.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from zoneinfo import ZoneInfo

import pandas as pd

from .completion import digest, identify_issues, prepare_evidence, reviewed_exports
from .engine import AuditError, audit_consistency, verify_exposures, compare_specifications
from .findings import validate_bundle


def invalidate(state, reason="Inputs changed. Run the audit again."):
    state["workflow_active"] = True
    state["workflow_invalid_reason"] = reason
    state.pop("workflow_project", None)
    for key in list(state):
        if key.startswith(("findings_", "completion_", "orcca_retrieval", "orcca_explanation")):
            del state[key]


def current_project(state, private):
    project = state.get("workflow_project")
    if project and project["classification"] == "private_research" and not private:
        invalidate(state, "Private project access is disabled on this server.")
        return None
    return project


def start_project(state, data, documents, completion_evidence, manifest, private=False, parent=None, ledger=None):
    revision = state.get("workflow_revision", 0) + 1
    snapshot = {k: (v.copy(deep=True) if isinstance(v, pd.DataFrame) else deepcopy(v)) for k, v in data.items()}
    inputs = digest([manifest, {k: v.to_json(orient="split", date_format="iso") if isinstance(v, pd.DataFrame) else v
        for k, v in snapshot.items()}, documents,
        completion_evidence.to_json(orient="split") if completion_evidence is not None else None])
    run_id = digest([revision, inputs, parent])
    results = {}
    try:
        results["consistency"] = audit_consistency(snapshot["panel"], snapshot["monthly"], snapshot["annual"])
    except AuditError as exc:
        results["consistency"] = {"status": "input_error", "message": str(exc)}
    if "tracker" in snapshot:
        try:
            results["exposures"] = verify_exposures(snapshot["panel"], snapshot["tracker"])
        except AuditError as exc:
            results["exposures"] = {"status": "input_error", "message": str(exc)}
    if snapshot.get("specifications"):
        results["specifications"] = compare_specifications(snapshot["specifications"])
    invalidate(state, "")
    project = {"run_id": run_id, "revision": revision, "input_fingerprint": inputs,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "private_research" if private else "synthetic_test",
        "data": snapshot, "documents": deepcopy(documents), "completion_evidence": completion_evidence,
        "manifest": deepcopy(manifest), "results": results, "parent_run_id": parent,
        "completion_ledger": deepcopy(ledger or []), "finding_decisions": {}, "bundle": None}
    state["workflow_revision"] = revision
    state["workflow_project"] = project
    state["workflow_findings_source"] = "Current web project"
    return project


def _receipt(project, section, value):
    payload = {"run_id": project["run_id"], "method": "Deterministic Python audit; not original-source prose or a fitted model", "result": value}
    payload = json.loads(json.dumps(payload, default=str), parse_constant=lambda _: None)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    name = f"workflow_{section}_{project['run_id'][:12]}.json"
    return {"chunk_id": "receipt-" + section, "source_id": name, "source_file": name,
        "page": None, "line_start": 1, "line_end": len(text.splitlines()),
        "locator": "Generated calculation receipt; trace inputs through the version manifest.",
        "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "classification": "private_local" if project["classification"] == "private_research" else "synthetic_public",
        "evidence_kind": "deterministic_receipt"}


def build_findings(project):
    docs, findings = [], []
    def add(section, title, statement, limits, level, value):
        doc = _receipt(project, section, value); docs.append(doc)
        findings.append({"finding_id": "C" + str(len(findings)+1), "title": title, "statement": statement,
            "verification_method": "Calculated from this project version; see the calculation receipt and input manifest.",
            "limitations": limits, "level": level, "evidence_ids": [doc["chunk_id"]], "review_status": "pending"})
    r = project["results"]; consistency = r["consistency"]
    if "summary" in consistency:
        s = consistency["summary"]
        add("consistency", "Panel structure and source agreement",
            f"This input has {s['panel_rows']} county-month rows, {s['counties']} counties and {s['months']} distinct months. "
            f"The county-month key is unique; complete grid: {s['complete_grid']}. "
            f"{s['verified_rows']} rows agree with all supplied death, suppression and population checks; "
            f"{len(consistency['anomalies'])} rows need review.",
            "Agreement among supplied files does not establish outcome-definition validity or reproduce H2a.",
            "verified_source_check", consistency)
        windows = consistency["annual_windows"]
        ready = [w for w in windows if "full_year_compatible" in w]
        add("annual", "Annual and partial-window compatibility",
            f"{len(ready)} county-years have complete annual checks; {len(windows)-len(ready)} lack sufficient source data. "
            f"{sum(not w['incorrect_window_equals_annual_feasible'] for w in ready)} cannot satisfy partial-window equals annual total. "
            f"{sum(w['full_year_compatible'] for w in ready)} are compatible when outside-window months are included.",
            "These are interval compatibility checks, not evidence about how the original likelihood was implemented.",
            "verified_source_check", windows)
    else:
        add("consistency", "Audit could not verify this input", consistency["message"],
            "Correct the inputs and rerun. No successful source check is asserted.", "insufficient_material", consistency)
    exposures = r.get("exposures")
    if exposures and "summary" in exposures:
        s = exposures["summary"]
        add("exposures", "Candidate exposure rules and unresolved statuses",
            f"{s['candidate_matches']} of {s['checks']} lag checks match the permissive candidate rule. "
            f"{s['uncertain_or_mismatched']} lag checks require review; {len(exposures['anomalies'])} tracker status/date flags were recorded.",
            "Candidate matching does not verify the original algorithm. Blanks, branching, stopping and restart rules still require source review.",
            "candidate_rule_match", exposures)
    else:
        add("exposures", "Exposure verification remains incomplete",
            exposures.get("message", "Tracker data were not supplied.") if exposures else "Tracker data were not supplied.",
            "No exposure verification result is substituted from another project.", "insufficient_material", exposures or {})
    if "specifications" in r:
        rows = r["specifications"]
        known = [x for x in rows if x['specification_A'] != 'unknown' and x['specification_B'] != 'unknown']
        add("specifications", "Supplied model specification comparison",
            f"Of {len(rows)} specification fields, {len(known)} are specified in both records and "
            f"{sum(x['specification_A'] != x['specification_B'] for x in known)} differ. "
            f"{len(rows)-len(known)} fields are unknown in at least one record.",
            "The comparison uses supplied source-linked specifications. No original model code or fitted result is verified.",
            "method_review", rows)
    # Only raw excerpts or calculation receipts enter the search index, never review decisions.
    for raw in project["documents"]:
        doc = deepcopy(raw); doc.setdefault("line_end", doc["line_start"])
        doc.setdefault("evidence_kind", "original_excerpt"); doc.setdefault("source_id", doc["source_file"])
        if doc["chunk_id"] not in {d["chunk_id"] for d in docs}:docs.append(doc)
    sources = [{"source_id": name, "source_file": name} for name in dict.fromkeys(d["source_id"] for d in docs)]
    bundle = {"schema_version": "1.0", "classification": project["classification"],
        "review_date": datetime.fromisoformat(project["created_at_utc"]).astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat(), "findings": findings, "documents": docs,
        "sources": sources, "questions": [], "hypotheses": [],
        "evaluation": {"human_labelled_cases": 0, "improvement_measured": False},
        "audit_run": {"run_id": project["run_id"], "revision": project["revision"],
            "input_fingerprint": project["input_fingerprint"], "inputs": project["manifest"],
            "parent_run_id": project["parent_run_id"], "external_llm_calls": 0, "effect_models_fitted": 0}}
    validate_bundle(bundle)
    project["bundle"] = bundle
    project["finding_decisions"] = {}
    return bundle


def decide_finding(project, finding_id, decision, reviewer, reason, note=""):
    from .english_export import english_text
    if not project.get("bundle") or finding_id not in {f["finding_id"] for f in project["bundle"]["findings"]}:
        raise AuditError("Generate findings on the current input version first.")
    if decision not in {"approve", "reject"} or not reviewer.strip() or not reason.strip():
        raise AuditError("An explicit decision, reviewer and reason are required.")
    for value in (reviewer, reason, note):english_text(value)
    project["finding_decisions"][finding_id] = {"finding_id": finding_id, "run_id": project["run_id"],
        "decision": decision, "reviewer": reviewer.strip(), "reason": reason.strip(), "researcher_note": note.strip(),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat()}


def findings_for_export(project, approved_only=False):
    if not project.get("bundle"):raise AuditError("Generate current findings before export.")
    bundle = deepcopy(project["bundle"])
    output = []
    for f in bundle["findings"]:
        decision = project["finding_decisions"].get(f["finding_id"])
        if decision and decision["run_id"] != project["run_id"]:raise AuditError("Stale finding decision.")
        if approved_only and (not decision or decision["decision"] != "approve"):continue
        f["review_status"] = decision["decision"] if decision else "pending"
        f["human_review"] = decision
        output.append(f)
    if not output:raise AuditError("No findings have been approved for this version.")
    bundle["findings"] = output
    return bundle


def apply_completion(state, project, data, issues, candidates, decisions):
    exported = reviewed_exports(data, issues, candidates, decisions, project["run_id"])
    accepted = [x for x in exported["review_ledger"] if x["kind"] == "source_evidence" and x["human_decision"]["decision"] == "approve"]
    if not accepted:raise AuditError("Accept at least one source-based candidate first. Statistical scenarios cannot update the audit.")
    tracker = data.copy(deep=True)
    for c in accepted:
        field=c['field'];mask=tracker.record_id == c['record_id']
        if field+'__original_value' not in tracker:tracker[field+'__original_value']=tracker[field]
        if field+'__completion_kind' not in tracker:tracker[field+'__completion_kind']='original_or_missing'
        if field+'__candidate_id' not in tracker:tracker[field+'__candidate_id']=''
        # CSV intake may use pandas' strict string dtype; a reviewed binary
        # candidate is numeric. Preserve original strings in the parallel column.
        tracker[field]=tracker[field].astype(object)
        tracker.loc[mask,field]=c['value']
        tracker.loc[mask,field+'__completion_kind']='human_approved_source_evidence'
        tracker.loc[mask,field+'__candidate_id']=c['candidate_id']
    tracker = tracker.drop(columns=["record_id"], errors="ignore")
    new_data = dict(project["data"])
    new_data.setdefault("original_tracker", project["data"]["tracker"].copy(deep=True))
    new_data["tracker"] = tracker
    return start_project(state, new_data, project["documents"], project["completion_evidence"], project["manifest"],
        private=project["classification"] == "private_research", parent=project["run_id"],
        ledger=project["completion_ledger"] + exported["review_ledger"])


def add_scoped_evidence(state, project, issue_id, chunk_id, quote, reviewer, reason):
    """Human-confirmed applicability, never automatically inferred from a date."""
    from .completion import KEYS
    from .exports import safe_csv
    from .io import read_table
    from .english_export import english_text
    _, issues = identify_issues(project['data']['tracker'])
    issue = next((x for x in issues if x['issue_id'] == issue_id and x['eligible']), None)
    doc = next((x for x in project['documents'] if x['chunk_id'] == chunk_id), None)
    if not issue or not doc or not quote.strip() or quote not in doc['text']:
        raise AuditError('Select a current eligible field and an exact passage from the selected document.')
    if not reviewer.strip() or not reason.strip():raise AuditError('An applicability reviewer and reason are required.')
    for value in (reviewer, reason):english_text(value)
    if hashlib.sha256(doc['text'].encode()).hexdigest() != doc['sha256']:raise AuditError('Source excerpt hash mismatch.')
    ref = {k: doc.get(k) for k in ('source_file','page','slide','line_start','line_end','chunk_id','sha256')}
    ref.update(applicability_reviewer=reviewer.strip(), applicability_reason=reason.strip(),
        applicability='Human-supplied county/time/strategy scope, not independently verified')
    row = {**{k: issue[k] for k in KEYS}, 'text': quote, 'source_ref': json.dumps(ref, ensure_ascii=False)}
    prior = project['completion_evidence']
    rows = prior.drop(columns=['source_file','_source_row'],errors='ignore').to_dict('records') if prior is not None else []
    rows.append(row)
    frame = read_table(safe_csv(rows), 'current_completion_evidence.csv')
    manifest = deepcopy(project['manifest'])
    manifest['scoped_evidence'] = {'source_file':'current_completion_evidence.csv', 'sha256':frame.attrs['sha256'],
        'scope_assignment':'Human supplied; recorded applicability is not an automatic factual verification'}
    return start_project(state, project['data'], project['documents'], frame, manifest,
        private=project['classification']=='private_research',parent=project['run_id'],ledger=project['completion_ledger'])


def export_project(project, approved_only=False):
    bundle = findings_for_export(project, approved_only)
    from .exports import report_bundle
    results = {"research_findings": bundle["findings"], "hypotheses": bundle["hypotheses"],
        "evidence_excerpts": bundle["documents"], "sources": bundle["sources"],
        "finding_review_decisions": list(project["finding_decisions"].values()),
        "completion_review_ledger": project["completion_ledger"], **project["results"]}
    for role in ("panel", "monthly", "annual", "tracker", "original_tracker", "reported_results"):
        if role in project["data"]:results["current_"+role] = project["data"][role].to_dict("records")
    if project['completion_evidence'] is not None:
        results['current_completion_evidence'] = project['completion_evidence'].to_dict('records')
    provenance = {**bundle["audit_run"], "review_date": bundle["review_date"], "date_timezone": "Asia/Shanghai"}
    from io import BytesIO
    import zipfile
    target = BytesIO(report_bundle(results, project["classification"], provenance))
    # Ship actual receipt files under their cited names as well as the evidence index.
    with zipfile.ZipFile(target, "a", zipfile.ZIP_DEFLATED) as archive:
        for doc in bundle['documents']:
            if doc['evidence_kind']=='deterministic_receipt' and doc['source_file'] in {
                f"workflow_{section}_{project['run_id'][:12]}.json" for section in ('consistency','annual','exposures','specifications')}:
                archive.writestr(doc['source_file'],doc['text'])
    return target.getvalue()
