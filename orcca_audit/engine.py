"""Deterministic audit logic generalized from the local ORCCAT audits.

No fitting, latent-death recovery, filesystem writes, or network requests.
Candidate replay is never labelled as reproduction of the original model.
"""
from __future__ import annotations

import calendar
import math
from datetime import date, datetime

import pandas as pd


class AuditError(ValueError):
    """Visible, actionable input/contract failure."""


def empty(value):
    return value is None or (not isinstance(value, (list, dict)) and bool(pd.isna(value))) or str(value).strip() == ""


def as_month(value):
    if empty(value):
        raise AuditError("Missing event/outcome month.")
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.strftime("%Y-%m")
    text = str(value).strip().replace("/", "-")
    for fmt in ("%Y-%m", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m")
        except ValueError:
            pass
    raise AuditError("Unrecognized month format; use YYYY-MM or a full date.")


def shift(month, amount):
    year, mon = map(int, month.split("-"))
    serial = year * 12 + mon - 1 + amount
    return f"{serial // 12:04d}-{serial % 12 + 1:02d}"


def interval(value):
    if empty(value):
        return None, None, "missing"
    text = str(value).strip()
    if text == "Suppressed":
        return 1, 9, "suppressed_assumed_1_9"
    if text in {"Not Applicable", "Not Available", "Unreliable"}:
        return None, None, "unavailable"
    try:
        number = float(text)
    except ValueError:
        return None, None, "unrecognized"
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        return None, None, "unrecognized"
    return int(number), int(number), "zero" if number == 0 else "observed"


def require(frame, fields, label):
    if not set(fields) <= set(frame.columns):
        raise AuditError(f"{label}: missing required columns: {', '.join(sorted(set(fields) - set(frame.columns)))}")


def keyed(frame, keys, label):
    require(frame, keys, label)
    if frame[list(keys)].isna().any().any() or any(frame[k].map(empty).any() for k in keys) or frame.duplicated(list(keys)).any():
        raise AuditError(f"{label}: missing or duplicate keys; no first/last-row fallback is allowed.")


def prepare(frame):
    frame = frame.copy()
    if "_source_row" not in frame:
        frame["_source_row"] = range(2, len(frame) + 2)
    if "source_file" not in frame:
        frame["source_file"] = "input_table"
    if "date" not in frame and "month" not in frame:
        raise AuditError("Input needs an explicit date or month column.")
    frame["month"] = frame["date" if "date" in frame else "month"].map(as_month)
    return frame


def number(value, label):
    try:
        parsed = float(value)
    except (ValueError, TypeError) as exc:
        raise AuditError(f"{label}: expected a finite numeric value.") from exc
    if not math.isfinite(parsed):
        raise AuditError(f"{label}: expected a finite numeric value.")
    return parsed


def audit_consistency(panel, monthly, annual):
    require(panel, ["county", "date", "Deaths", "y_lo", "y_hi", "suppressed", "population"], "Panel")
    require(monthly, ["county", "month", "Deaths"], "Monthly source")
    require(annual, ["county", "year", "Deaths", "population"], "Annual source")
    p, m, a = prepare(panel), prepare(monthly), annual.copy()
    if "_source_row" not in a:
        a["_source_row"] = range(2, len(a) + 2)
    a["year"] = a["year"].astype(str).str.replace(r"\.0$", "", regex=True)
    keyed(p, ["county", "month"], "Panel")
    keyed(m, ["county", "month"], "Monthly source")
    keyed(a, ["county", "year"], "Annual source")
    mi = {(r["county"], r["month"]): r for r in m.to_dict("records")}
    ai = {(r["county"], r["year"]): r for r in a.to_dict("records")}
    checks, anomalies = [], []
    for r in p.to_dict("records"):
        src = mi.get((r["county"], r["month"]))
        ann = ai.get((r["county"], r["month"][:4]))
        lo, hi, state = interval(src["Deaths"]) if src else (None, None, "source_missing")
        row = {"county": r["county"], "month": r["month"], "panel_row": r["_source_row"],
               "panel_file": r["source_file"], "monthly_row": src["_source_row"] if src else None,
               "source_state": state, "annual_row": ann["_source_row"] if ann else None,
               "monthly_file": src.get("source_file", "monthly_source") if src else None,
               "annual_file": ann.get("source_file", "annual_source") if ann else None}
        row["interval_matches"] = lo is not None and number(r["y_lo"], "y_lo") == lo and number(r["y_hi"], "y_hi") == hi
        row["deaths_matches"] = src is not None and interval(r["Deaths"]) == interval(src["Deaths"])
        row["suppression_matches"] = lo is not None and number(r["suppressed"], "suppressed") == int(state == "suppressed_assumed_1_9")
        row["population_matches"] = ann is not None and number(r["population"], "population") == number(ann["population"], "annual population")
        row["status"] = "verified_against_supplied_sources" if all(row[k] for k in
            ["interval_matches", "deaths_matches", "suppression_matches", "population_matches"]) else "needs_review"
        checks.append(row)
        if row["status"] == "needs_review":
            anomalies.append(dict(row, reason="Missing source or inconsistent supplied values; not repaired."))
    windows = []
    panel_keys = set(zip(p.county, p.month))
    for (county, year), ann in sorted(ai.items()):
        if not any((county, f"{year}-{j:02d}") in panel_keys for j in range(1, 13)):
            continue
        inside, outside, inside_rows, outside_rows = [], [], [], []
        for j in range(1, 13):
            key = county, f"{year}-{j:02d}"
            src = mi.get(key)
            if src is None or interval(src["Deaths"])[0] is None:
                break
            target, locations = (inside, inside_rows) if key in panel_keys else (outside, outside_rows)
            target.append(interval(src["Deaths"])[:2]); locations.append(str(src["_source_row"]))
        al, ah, state = interval(ann["Deaths"])
        if len(inside) + len(outside) != 12 or al is None:
            windows.append({"county": county, "year": year, "status": "insufficient_full_year_source",
                "annual_file": ann.get("source_file", "annual_source"), "annual_row": ann["_source_row"]})
            continue
        wl, wh = sum(x[0] for x in inside), sum(x[1] for x in inside)
        ol, oh = sum(x[0] for x in outside), sum(x[1] for x in outside)
        lower, upper = max(wl, al - oh), min(wh, ah - ol)
        windows.append({"county": county, "year": year, "window_months": len(inside),
            "outside_months": len(outside), "window_lo": wl, "window_hi": wh,
            "outside_lo": ol, "outside_hi": oh, "annual_lo": al, "annual_hi": ah,
            "feasible_window_lo": lower, "feasible_window_hi": upper,
            "full_year_compatible": lower <= upper, "incorrect_window_equals_annual_feasible": wl <= ah and al <= wh,
            "annual_row": ann["_source_row"], "monthly_inside_rows": ";".join(inside_rows),
            "monthly_outside_rows": ";".join(outside_rows),
            "annual_file": ann.get("source_file", "annual_source"),
            "monthly_files": ";".join(sorted(set(m[m.county == county].source_file.astype(str)))),
            "status": "compatibility_check_only_not_model_implementation"})
    return {"row_checks": checks, "annual_windows": windows, "anomalies": anomalies,
        "summary": {"panel_rows": len(p), "counties": p.county.nunique(), "months": p.month.nunique(),
            "unique_key": True, "complete_grid": len(p) == p.county.nunique() * p.month.nunique(),
            "verified_rows": sum(r["status"] == "verified_against_supplied_sources" for r in checks),
            "model_reproduced": False}}


def status_value(value):
    if empty(value):
        return None
    try:
        parsed = float(value)
        if parsed in (0, 1):
            return int(parsed)
    except (ValueError, TypeError):
        pass
    raise AuditError("Strategy status must be 0, 1 or blank; unsupported codes require review.")


def verify_exposures(panel, tracker):
    """Tidy tracker schema: county, month, menu, slot, show, started, discontinued.

    A show=0 menu-month still needs a placeholder row (slot=0) to distinguish an
    explicit skipped snapshot from an absent record. No absent month is filled.
    """
    require(tracker, ["county", "month", "menu", "slot", "show", "started", "discontinued"], "Tracker")
    p, tr = prepare(panel), prepare(tracker)
    keyed(p, ["county", "month"], "Panel")
    keyed(tr, ["county", "month", "menu", "slot"], "Tracker")
    if not set(tr.menu) <= {"OEND", "MOUD"}:
        raise AuditError("Only declared OEND/MOUD menus are supported.")
    states, snapshots, anomalies = {}, [], []
    for (county, menu), data in tr.groupby(["county", "menu"], sort=True):
        last = None
        previous_month = None
        for month, records in data.groupby("month", sort=True):
            rows = records.to_dict("records")
            show_values = {status_value(r["show"]) for r in rows}
            if len(show_values) != 1:
                raise AuditError("Conflicting show flags within a county/menu/month.")
            show = next(iter(show_values))
            gap = previous_month is not None and shift(previous_month, 1) != month
            if gap:
                last = None  # An absent event is not explicit no-change.
            if show == 1:
                pairs = [(status_value(r["started"]), status_value(r["discontinued"])) for r in rows]
                lo = sum(s == 1 and v == 0 for s, v in pairs)
                hi = sum(s != 0 and v != 1 and str(r["slot"]) != "0" for r, (s, v) in zip(rows, pairs))
                value = sum(s == 1 and v != 1 for s, v in pairs)
                last = {"lo": lo, "hi": hi, "value": value, "source_rows": ";".join(sorted({str(r["_source_row"]) for r in rows})),
                    "source_files": ";".join(sorted({str(r["source_file"]) for r in rows})),
                    "source_records": [{k: r.get(k) for k in ["source_file", "source_sheet", "_source_row", "version",
                        "started_field", "discontinued_field", "dictionary_file", "dictionary_sheet",
                        "dictionary_started_row", "dictionary_discontinued_row"]} for r in rows]}
                action = "snapshot_candidate"
                for r, (started, disc) in zip(rows, pairs):
                    locator = {k: r.get(k) for k in ["source_sheet", "version", "started_field", "discontinued_field",
                        "dictionary_file", "dictionary_sheet", "dictionary_started_row", "dictionary_discontinued_row"]}
                    if started is None and str(r["slot"]) != "0":
                        anomalies.append({**locator, "county": county, "menu": menu, "month": month, "source_row": r["_source_row"],
                            "source_file": r["source_file"], "reason": "unknown_started_status", "status": "needs_review"})
                    if started == 1 and disc is None:
                        anomalies.append({**locator, "county": county, "menu": menu, "month": month, "source_row": r["_source_row"],
                            "source_file": r["source_file"], "reason": "unknown_discontinuation", "status": "needs_review"})
                    if started == 1 and not empty(r.get("start_month")) and as_month(r["start_month"]) > month:
                        anomalies.append({**locator, "county": county, "menu": menu, "month": month, "source_row": r["_source_row"],
                            "source_file": r["source_file"], "reason": "start_date_after_event_month", "status": "needs_review"})
            elif show == 0:
                action = "carry_previous_snapshot_candidate" if last is not None else "insufficient_history"
            else:
                last = None
                action = "unknown_show_status"
            states[county, month, menu] = None if last is None else dict(last)
            snapshots.append({"county": county, "month": month, "menu": menu, "action": action,
                "lo": last["lo"] if last else None, "hi": last["hi"] if last else None,
                "source_rows": last["source_rows"] if last else None, "status": "candidate_replay_only"})
            previous_month = month
    checks = []
    for r in p.to_dict("records"):
        for menu in ["OEND", "MOUD"]:
            for lag, months in [("L1", [shift(r["month"], -1)]), ("L13", [shift(r["month"], -j) for j in [1, 2, 3]])]:
                variable = f"{menu}_{lag}"
                if variable not in p:
                    raise AuditError(f"Panel missing {variable}.")
                history = [states.get((r["county"], month, menu)) for month in months]
                ready = all(x is not None for x in history)
                low = sum(x["lo"] for x in history) / len(history) if ready else None
                high = sum(x["hi"] for x in history) / len(history) if ready else None
                candidate = sum(x["value"] for x in history) / len(history) if ready else None
                match = ready and not empty(r[variable]) and math.isclose(candidate, number(r[variable], variable), abs_tol=1e-9)
                status = "insufficient_history" if not ready else "candidate_match" if match else "candidate_mismatch"
                checks.append({"county": r["county"], "month": r["month"], "variable": variable,
                    "panel_row": r["_source_row"], "panel_file": r["source_file"], "panel_value": r[variable],
                    "candidate_lo": low, "candidate_hi": high, "permissive_candidate_matches": bool(match),
                    "permissive_candidate_value": candidate,
                    "exposure_months": ";".join(months), "snapshot_source_rows": "|".join(x["source_rows"] for x in history) if ready else None,
                    "snapshot_sources": [x["source_records"] for x in history] if ready else None,
                    "status": status, "needs_review": not ready or low != high or not match})
    return {"lag_checks": checks, "snapshots": snapshots, "anomalies": anomalies,
        "summary": {"checks": len(checks), "candidate_matches": sum(r["permissive_candidate_matches"] for r in checks),
            "uncertain_or_mismatched": sum(r["needs_review"] for r in checks), "original_algorithm_verified": False}}


SPEC_FIELDS = ["exposure_window", "outcome_window", "exposure_scale", "missing_rule", "distribution",
               "county_effects", "time_terms", "prior_mortality", "population_offset", "annual_constraint", "standard_errors"]


def compare_specifications(specs):
    if not isinstance(specs, list) or len(specs) != 2 or any(not isinstance(s, dict) for s in specs):
        raise AuditError("Supply exactly two explicitly identified specifications.")
    if any(not s.get("spec_id") or not s.get("source") for s in specs):
        raise AuditError("Every specification needs a spec_id and actual source locator.")
    return [{"item": field, "specification_A": specs[0].get(field, "unknown"),
        "specification_B": specs[1].get(field, "unknown"), "source_A": specs[0]["source"],
        "source_B": specs[1]["source"], "status": "user_supplied_specification_not_implementation_validation"}
        for field in SPEC_FIELDS]
