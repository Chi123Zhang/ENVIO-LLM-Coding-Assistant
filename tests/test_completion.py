"""Synthetic software tests; never empirical research labels/performance."""
import copy
import json
from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from orcca_audit.completion import (binary_scenarios, evidence_candidate, identify_issues,
    prepare_evidence, reviewed_exports)
from orcca_audit.completion_ui import synthetic_inputs
from orcca_audit.engine import AuditError
from orcca_audit.exports import report_bundle

ROOT = Path(__file__).resolve().parents[1]


def setup():
    raw, text = synthetic_inputs()
    data, issues = identify_issues(raw)
    return raw, data, issues, prepare_evidence(text)


def issue_at(issues, month, field):
    return next(i for i in issues if i["month"] == month and i["field"] == field)


def decision(candidate, approve=True, run="r"):
    return {"candidate_id": candidate["candidate_id"], "run_id": run,
        "decision": "approve" if approve else "reject", "reviewer": "SYNTHETIC tester", "reason": "Software test only"}


def test_missing_is_not_no_and_branching_is_held():
    raw, data, issues, _ = setup()
    assert len(issues) == 4
    raw.loc[0, ["started", "discontinued", "start_month"]] = [0, None, None]
    raw.loc[1, ["show", "started"]] = [0, None]
    _, issues = identify_issues(raw)
    assert not issue_at(issues, "2024-01", "discontinued")["eligible"]
    assert not issue_at(issues, "2024-02", "started")["eligible"]
    with pytest.raises(AuditError, match="Duplicate"):
        identify_issues(pd.concat([raw, raw.iloc[:1]]))
    raw.loc[0, "started"] = 9
    with pytest.raises(AuditError, match="codes"):
        identify_issues(raw)


def test_retrieval_requires_exact_context_and_explicit_text():
    _, _, issues, docs = setup()
    target = issue_at(issues, "2024-04", "started")
    result = evidence_candidate(target, docs, "r")
    candidate = result["candidates"][0]
    assert result["status"] == "source_candidate_unreviewed" and candidate["value"] == 1
    assert candidate["evidence"][0]["quote"] in docs[0]["text"]
    for key in ("county", "month", "slot", "version", "menu"):
        changed = copy.deepcopy(docs)
        changed[0][key] = "different"
        assert not evidence_candidate(target, changed, "r")["candidates"]
    raw, ev = synthetic_inputs()
    ev.loc[0, "text"] = "Narrative: do not assume started=1."
    assert evidence_candidate(target, prepare_evidence(ev), "r")["status"] == "retrieved_but_no_explicit_field_assertion"
    ev.loc[0, "role"] = "human_label"
    with pytest.raises(AuditError, match="source excerpts"):
        prepare_evidence(ev)


def test_conflict_and_retrieval_failure_never_fill(monkeypatch):
    _, _, issues, docs = setup()
    target = issue_at(issues, "2024-06", "discontinued")
    assert evidence_candidate(target, docs, "r")["status"] == "conflicting_source_assertions"
    target = issue_at(issues, "2024-04", "started")
    monkeypatch.setattr("orcca_audit.completion.retrieve", lambda *a, **kw:
        {"status": "hybrid_unavailable_no_fallback", "citations": []})
    assert not evidence_candidate(target, docs, "r", "hybrid")["candidates"]


def test_statistics_are_prior_only_seeded_and_explicitly_assumed():
    _, data, issues, _ = setup()
    target = issue_at(issues, "2024-05", "discontinued")
    with pytest.raises(AuditError, match="acknowledgement"):
        binary_scenarios(target, data, "r")
    result = binary_scenarios(target, data, "r", assumptions_acknowledged=True)
    candidate = result["candidates"][0]
    assert candidate["n_support"] == 3 and candidate["posterior"] == [1, 4]
    assert candidate["p_mean"] == .2 and len(candidate["draws"]) == 20
    assert all(r["month"] < target["month"] for r in candidate["support"])
    assert result == binary_scenarios(target, data, "r", assumptions_acknowledged=True)
    data.loc[data.month > target["month"], "discontinued"] = 1
    assert result == binary_scenarios(target, data, "r", assumptions_acknowledged=True)
    result = binary_scenarios(target, data.iloc[:2], "r", assumptions_acknowledged=True)
    assert result["status"] == "insufficient_prior_comparable_observations"


def test_approval_required_originals_preserved_and_scenarios_separate():
    raw, data, issues, docs = setup()
    source = evidence_candidate(issue_at(issues, "2024-04", "started"), docs, "r")["candidates"][0]
    stats = binary_scenarios(issue_at(issues, "2024-05", "discontinued"), data, "r",
        assumptions_acknowledged=True)["candidates"][0]
    candidates = [source, stats]
    before = data.copy(deep=True)
    exported = reviewed_exports(data, issues, candidates, [], "r")
    assert pd.isna(exported["derived_records"][3]["started__reviewed"])
    assert not exported["statistical_scenario_values"]
    exported = reviewed_exports(data, issues, candidates, [decision(source), decision(stats)], "r")
    pd.testing.assert_frame_equal(data, before)
    row = exported["derived_records"][3]
    assert pd.isna(row["started"]) and row["started__reviewed"] == 1
    assert pd.isna(exported["derived_records"][4]["discontinued__reviewed"])
    assert len(exported["statistical_scenario_values"]) == 20
    assert len(exported["derived_records"]) == len(raw)
    with pytest.raises(AuditError, match="Stale"):
        reviewed_exports(data, issues, candidates, [decision(source, run="previous")], "r")
    bad = copy.deepcopy(source); bad["value"] = 0
    with pytest.raises(AuditError, match="changed"):
        reviewed_exports(data, issues, [bad], [decision(bad)], "r")
    with zipfile.ZipFile(BytesIO(report_bundle(exported, "synthetic_public"))) as archive:
        assert "derived_records.csv" in archive.namelist()
        assert json.loads(archive.read("audit_report.json"))["results"]["completion_metadata"]["original_fields_preserved"]


def test_completion_public_ui_end_to_end_without_uploads(monkeypatch):
    monkeypatch.delenv("ORCCA_DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("ORCCA_ALLOW_PRIVATE_UPLOADS", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.switch_page("pages/03_ORCCA_Evidence_Completion.py").run()
    assert not at.exception and not at.get("file_uploader")
    assert len(at.tabs) == 4 and at.radio(key="completion_mode").options == ["模拟练习"]
    at.button(key="completion_retrieve").click().run()
    assert len(at.session_state["completion_candidates"]) == 1
    cid = next(iter(at.session_state["completion_candidates"]))
    at.text_input(key="completion_reviewer").set_value("Synthetic tester")
    at.text_area(key="completion_reason_" + cid).set_value("Synthetic source matches the target")
    at.radio(key="completion_decision_" + cid).set_value("接受")
    next(b for b in at.button if b.label == "保存本次人工判定").click().run()
    assert not at.exception and len(at.session_state["completion_decisions"]) == 1
    assert any("人工判定已保存" in x.value for x in at.success)
    at.switch_page("app.py").run()
    assert not at.exception and at.title[0].value == "ENVIO LLM Coding Assistant"
