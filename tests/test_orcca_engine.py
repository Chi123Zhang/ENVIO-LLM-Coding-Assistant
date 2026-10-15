import copy

import pandas as pd
import pytest

from orcca_audit.demo import demo_data
from orcca_audit.engine import (AuditError, audit_consistency, interval, verify_exposures,
                                compare_specifications, shift)


def test_public_fixtures_are_synthetic_and_annual_windows_include_outside():
    data = demo_data()
    assert all(c.startswith("DEMO_") for c in data["panel"].county)
    result = audit_consistency(data["panel"], data["monthly"], data["annual"])
    assert result["summary"]["verified_rows"] == len(data["panel"])
    assert all(r["full_year_compatible"] for r in result["annual_windows"])
    assert all(r["outside_months"] == 6 for r in result["annual_windows"])
    assert all(not r["incorrect_window_equals_annual_feasible"] for r in result["annual_windows"])
    assert not result["summary"]["model_reproduced"]


def test_duplicate_panel_key_is_not_silently_deduplicated():
    d = demo_data()
    with pytest.raises(AuditError, match="duplicate keys"):
        audit_consistency(pd.concat([d["panel"], d["panel"].iloc[:1]]), d["monthly"], d["annual"])


def test_states_keep_zero_suppression_and_missing_distinct():
    assert interval(0) == (0, 0, "zero")
    assert interval("Suppressed") == (1, 9, "suppressed_assumed_1_9")
    assert interval("")[2] == "missing"
    assert interval("Not Applicable")[2] == "unavailable"
    assert interval("unreported")[2] == "unrecognized"


def test_candidate_replay_keeps_unknown_discontinuation_visible():
    d = demo_data()
    result = verify_exposures(d["panel"], d["tracker"])
    assert result["summary"]["checks"] == len(d["panel"]) * 4
    assert result["summary"]["candidate_matches"] == result["summary"]["checks"]
    assert any(r["reason"] == "unknown_discontinuation" for r in result["anomalies"])
    assert any(r["reason"] == "start_date_after_event_month" for r in result["anomalies"])
    assert any(r["needs_review"] and r["permissive_candidate_matches"] for r in result["lag_checks"])
    assert not result["summary"]["original_algorithm_verified"]


def test_absent_month_is_not_no_change():
    d = demo_data()
    tr = d["tracker"].loc[lambda x: x.month != "2024-02"].copy()
    tr.loc[tr.month == "2024-03", "show"] = 0
    result = verify_exposures(d["panel"], tr)
    assert any(r["status"] == "insufficient_history" for r in result["lag_checks"])


def test_missing_outside_month_does_not_report_annual_success():
    d = demo_data()
    result = audit_consistency(d["panel"], d["monthly"].loc[lambda x: x.month != "2024-12"], d["annual"])
    assert all(r["status"] == "insufficient_full_year_source" for r in result["annual_windows"])


def test_malformed_numeric_value_is_an_actionable_error():
    d = demo_data(); d["panel"].loc[0, "population"] = float("nan")
    with pytest.raises(AuditError, match="numeric"):
        audit_consistency(d["panel"], d["monthly"], d["annual"])


def test_comparison_preserves_unknowns_and_actual_source():
    result = compare_specifications(demo_data()["specifications"])
    assert next(r for r in result if r["item"] == "standard_errors")["specification_A"] == "unknown"
    assert all("PUBLIC_DEMO_METHODS" in r["source_A"] for r in result)
    assert shift("2024-01", -1) == "2023-12"


def test_unknown_started_is_not_explicit_no():
    d = demo_data(); d["tracker"].loc[0, "started"] = None
    result = verify_exposures(d["panel"], d["tracker"])
    assert any(r["reason"] == "unknown_started_status" for r in result["anomalies"])
    assert any(r["candidate_lo"] != r["candidate_hi"] for r in result["lag_checks"])


def test_malformed_specs_fail_as_contract_error():
    for value in [None, {}, ["a", "b"]]:
        with pytest.raises(AuditError):
            compare_specifications(value)
