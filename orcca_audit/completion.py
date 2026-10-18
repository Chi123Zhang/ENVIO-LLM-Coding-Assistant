"""Evidence recovery and explicitly assumed binary sensitivity scenarios.

No model/API calls, original-data overwrites, or recovery of suppressed counts.
Evidence extraction deliberately accepts only explicit field=value lines.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import beta

from .engine import AuditError, as_month, empty, require
from .retrieval import retrieve

FIELDS = ("started", "discontinued", "start_month")
KEYS = ("county", "month", "menu", "slot", "version")
UNKNOWN = {"", "unknown", "uncertain", "na", "n/a", "null", "?", "missing", "not reported", "未知", "不确定", "未报告"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        default=str).encode()).hexdigest()


def missing(value):
    return empty(value) or str(value).strip().lower() in UNKNOWN


def canonical(value, field):
    if missing(value):
        return None
    if field == "start_month":
        return as_month(value)
    try:
        if float(value) in (0, 1):
            return int(float(value))
    except (ValueError, TypeError):
        pass
    raise AuditError("Status codes must be 0/1 or an explicit unknown; verify the dictionary mapping.")


def prepare_records(frame):
    require(frame, ["county", "month", "menu", "slot", "show", *FIELDS], "Completion Tracker")
    data = frame.copy().reset_index(drop=True)
    if "version" not in data:
        data["version"] = ""
    for col in ("county", "menu", "slot", "version"):
        data[col] = data[col].map(lambda x: "" if empty(x) else str(x).strip())
    data["slot"] = data["slot"].str.replace(r"\.0$", "", regex=True)
    data["month"] = data["month"].map(as_month)
    if (data[["county", "menu", "slot"]] == "").any().any() or not set(data.menu) <= {"OEND", "MOUD"}:
        raise AuditError("County, menu and strategy slot must be explicit.")
    if data.duplicated(list(KEYS)).any():
        raise AuditError("Duplicate county/month/menu/slot/version; resolve before completion.")
    for field in (*FIELDS, "show"):
        for value in data[field]:
            canonical(value, field)
    data["record_id"] = [digest({key: r[key] for key in KEYS})[:20] for r in data.to_dict("records")]
    return data


def identify_issues(frame):
    data = prepare_records(frame)
    issues = []
    for row in data.to_dict("records"):
        started, show = canonical(row["started"], "started"), canonical(row["show"], "show")
        for field in FIELDS:
            if not missing(row[field]):
                continue
            reason = "missing_or_explicit_unknown"
            eligible = True
            if row["slot"] == "0" or show != 1:
                reason, eligible = "skip_or_update_status_requires_review", False
            elif field in {"discontinued", "start_month"} and started != 1:
                reason, eligible = "branching_or_started_status_requires_review", False
            issues.append({"issue_id": row["record_id"] + ":" + field, "record_id": row["record_id"],
                **{key: row[key] for key in KEYS}, "field": field,
                "original_value": None if empty(row[field]) else str(row[field]),
                "eligible": eligible, "reason": reason,
                "source_file": row.get("source_file", "supplied_tracker"),
                "source_sheet": row.get("source_sheet", ""), "source_row": row.get("_source_row"),
                "source_field": row.get(field + "_field", field),
                "dictionary_row": row.get("dictionary_" + field + "_row")})
    return data, issues


def prepare_evidence(frame):
    require(frame, ["county", "month", "menu", "slot", "text"], "Context-linked evidence CSV")
    docs = []
    for n, raw in enumerate(frame.to_dict("records")):
        scope = {key: "" if empty(raw.get(key)) else str(raw[key]).strip() for key in KEYS}
        scope["month"] = as_month(raw["month"])
        scope["slot"] = scope["slot"].removesuffix(".0")
        if any(not scope[key] for key in ("county", "menu", "slot")) or scope["menu"] not in {"OEND", "MOUD"}:
            raise AuditError("Evidence needs explicit county, applicable month, menu and slot.")
        text = str(raw["text"]) if not empty(raw["text"]) else ""
        if not text.strip():
            raise AuditError("Evidence text cannot be empty.")
        # These are uploaded source excerpts, never reviewer answers or outputs.
        if str(raw.get("role", "source_excerpt")) != "source_excerpt":
            raise AuditError("Only source excerpts can be indexed; no labels or model outputs.")
        source = str(raw.get("source_file", "evidence.csv"))
        line = int(raw.get("_source_row", n + 2))
        docs.append({**scope, "chunk_id": digest([scope, source, line, text]),
            "source_file": source, "line_start": line, "line_end": line,
            "page": None, "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "classification": "private_local", "role": "source_excerpt",
            "upstream_source_as_supplied": str(raw.get("source_ref", "")),
            "locator_kind": "uploaded_csv_record_end_line; upstream reference is user-supplied"})
    return docs


def _candidate(issue, run_id, kind, **values):
    result = {"issue_id": issue["issue_id"], "record_id": issue["record_id"],
        **{key: issue[key] for key in KEYS},
        "field": issue["field"], "run_id": run_id, "kind": kind,
        "human_review_required": True, **values}
    result["candidate_id"] = digest(result)
    return result


def evidence_candidate(issue, documents, run_id, backend="bm25"):
    if not issue["eligible"]:
        return {"status": "held_for_branching_review", "candidates": [], "citations": []}
    scoped = [d for d in documents if all(d.get(key, "") == issue[key] for key in KEYS)]
    if not scoped:
        return {"status": "no_exact_county_time_strategy_version_evidence", "candidates": [], "citations": []}
    result = retrieve(issue["field"], scoped, backend, top_k=len(scoped))
    if result["status"] != "retrieved_evidence_unreviewed":
        return {**result, "candidates": []}
    field = issue["field"]
    pattern = re.compile(r"^\s*" + re.escape(field) + r"\s*[:=]\s*(\d{4}-\d{2}|[01])\s*$", re.M)
    claims = []
    for doc in result["citations"]:
        for match in pattern.finditer(doc["text"]):
            try:
                value = canonical(match.group(1), field)
            except AuditError:
                continue
            if field == "start_month" and value > issue["month"]:
                return {**result, "status": "start_after_applicable_month_requires_review", "candidates": []}
            claims.append({"value": value, "chunk_id": doc["chunk_id"], "quote": match.group(0).strip(),
                "source_file": doc["source_file"], "line": doc["line_start"], "sha256": doc["sha256"]})
    if not claims:
        return {**result, "status": "retrieved_but_no_explicit_field_assertion", "candidates": []}
    if len({str(c["value"]) for c in claims}) != 1:
        return {**result, "status": "conflicting_source_assertions", "claims": claims, "candidates": []}
    return {**result, "status": "source_candidate_unreviewed", "candidates": [
        _candidate(issue, run_id, "source_evidence", value=claims[0]["value"], evidence=claims,
            retrieval_backend=backend, extraction="exact field=value lines; no LLM inference")]}


def binary_scenarios(issue, data, run_id, *, assumptions_acknowledged=False, seed=2026, draws=20):
    if not assumptions_acknowledged:
        raise AuditError("Statistical sensitivity scenarios require explicit assumption acknowledgement.")
    if not issue["eligible"] or issue["field"] not in {"started", "discontinued"}:
        raise AuditError("Only eligible binary unknowns support this sensitivity scenario.")
    if not 2 <= draws <= 100 or not 0 <= seed < 2**32:
        raise AuditError("Use 2–100 draws and a nonnegative 32-bit seed.")
    # Same strategy and version; no future information used to impute the past.
    donors = data[(data.county == issue["county"]) & (data.menu == issue["menu"]) &
        (data.slot == issue["slot"]) & (data.version == issue["version"]) & (data.month < issue["month"])]
    support = []
    for row in donors.to_dict("records"):
        if canonical(row["show"], "show") != 1 or (issue["field"] == "discontinued" and
                canonical(row["started"], "started") != 1):
            continue
        value = canonical(row[issue["field"]], issue["field"])
        if value is not None:
            support.append({"record_id": row["record_id"], "month": row["month"], "value": value,
                "source_file": row.get("source_file"), "source_row": row.get("_source_row")})
    if len(support) < 3:
        return {"status": "insufficient_prior_comparable_observations", "candidates": [], "n_support": len(support)}
    successes, n = sum(r["value"] for r in support), len(support)
    a, b = 1 + successes, 1 + n - successes
    # Per-field stream avoids using identical draws for different missing cells.
    stream_seed = int(digest([seed, issue["issue_id"]])[:16], 16)
    rng = np.random.default_rng(stream_seed)
    values = rng.binomial(1, rng.beta(a, b, draws)).tolist()
    candidate = _candidate(issue, run_id, "statistical_sensitivity_only", draws=values,
        p_mean=a / (a + b), p_interval_assuming_model=beta.ppf([.025, .975], a, b).tolist(),
        support=support, n_support=n, seed=seed, stream_seed=stream_seed, prior="Beta(1,1)", posterior=[a, b],
        assumption="Prior snapshots are exchangeable Bernoulli observations and missingness is ignorable within the same county/menu/slot/version. Repeated states are dependent; this assumption is unverified.",
        limitation="Exploratory sensitivity scenarios, not recovered truth, validated missingness model or formal multiple imputation. No effects fitted.")
    return {"status": "statistical_scenarios_unreviewed", "candidates": [candidate]}


def reviewed_exports(data, issues, candidates, decisions, run_id):
    by_issue, by_candidate = {x["issue_id"]: x for x in issues}, {x["candidate_id"]: x for x in candidates}
    records = data.copy()
    for field in FIELDS:
        records[field + "__reviewed"] = records[field].map(lambda v: canonical(v, field)).astype(object)
        records[field + "__completion_kind"] = "original_or_missing"
        records[field + "__candidate_id"] = ""
    approved, ledger = {}, []
    for decision in decisions:
        candidate = by_candidate.get(decision.get("candidate_id"))
        if candidate is None or candidate["run_id"] != run_id or decision.get("run_id") != run_id:
            raise AuditError("Stale or unknown candidate/decision; rerun on the current inputs.")
        if digest({k: v for k, v in candidate.items() if k != "candidate_id"}) != candidate["candidate_id"]:
            raise AuditError("Candidate content changed after creation.")
        if decision.get("decision") not in {"approve", "reject"} or not str(decision.get("reviewer", "")).strip() or not str(decision.get("reason", "")).strip():
            raise AuditError("Reviewer, reason and an explicit approve/reject decision are required.")
        issue = by_issue.get(candidate["issue_id"])
        if not issue or not issue["eligible"] or candidate["field"] != issue["field"]:
            raise AuditError("Candidate does not match an eligible issue.")
        ledger.append({**candidate, "human_decision": decision})
        if decision["decision"] == "approve":
            if issue["issue_id"] in approved:
                raise AuditError("Multiple approved candidates for the same field.")
            approved[issue["issue_id"]] = candidate
    for issue_id, candidate in approved.items():
        if candidate["kind"] != "source_evidence":
            continue
        field = candidate["field"]
        mask = records.record_id == candidate["record_id"]
        records.loc[mask, field + "__reviewed"] = candidate["value"]
        records.loc[mask, field + "__completion_kind"] = "human_approved_source_evidence"
        records.loc[mask, field + "__candidate_id"] = candidate["candidate_id"]
    scenarios = []
    for candidate in approved.values():
        if candidate["kind"] == "statistical_sensitivity_only":
            for i, value in enumerate(candidate["draws"], 1):
                scenarios.append({"scenario": i, "record_id": candidate["record_id"], "field": candidate["field"],
                    "assumed_value": value, "candidate_id": candidate["candidate_id"], "kind": candidate["kind"]})
    return {"derived_records": records.to_dict("records"), "review_ledger": ledger,
        "statistical_scenario_values": scenarios,
        "completion_metadata": {"run_id": run_id, "original_fields_preserved": True,
            "scenario_scope": "long-form sensitivity overlays; not independent participants or recovered values",
            "external_llm_calls": 0, "effect_models_fitted": 0}}
