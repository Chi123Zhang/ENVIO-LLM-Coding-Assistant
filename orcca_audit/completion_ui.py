"""Session-only completion workbench; private inputs are server-gated."""
from io import BytesIO
from datetime import datetime, timezone
import hashlib
import zipfile

import pandas as pd
import streamlit as st

from .adapters import tracker_workbooks_to_tidy
from .completion import (binary_scenarios, digest, evidence_candidate, identify_issues,
    prepare_evidence, reviewed_exports)
from .engine import AuditError
from .exports import report_bundle, safe_csv
from .io import read_table, read_workbook
from .policy import private_uploads_allowed


def synthetic_inputs():
    records = []
    for month in range(1, 8):
        records.append({"county": "SYNTHETIC_A", "month": f"2024-{month:02d}", "menu": "OEND",
            "slot": "1", "version": "SYNTHETIC", "show": 1,
            "started": None if month in (4, 7) else 1,
            "discontinued": None if month in (5, 6) else 0, "start_month": "2024-01",
            "source_file": "synthetic_tracker.csv", "_source_row": month + 1})
    evidence = pd.DataFrame([
        {"county": "SYNTHETIC_A", "month": "2024-04", "menu": "OEND", "slot": "1",
         "version": "SYNTHETIC", "text": "SYNTHETIC meeting record\nstarted=1", "source_ref": "SYNTHETIC note A"},
        {"county": "SYNTHETIC_A", "month": "2024-06", "menu": "OEND", "slot": "1",
         "version": "SYNTHETIC", "text": "SYNTHETIC record B\ndiscontinued=0", "source_ref": "SYNTHETIC note B"},
        {"county": "SYNTHETIC_A", "month": "2024-06", "menu": "OEND", "slot": "1",
         "version": "SYNTHETIC", "text": "SYNTHETIC conflicting record C\ndiscontinued=1", "source_ref": "SYNTHETIC note C"},
    ])
    evidence["source_file"] = "synthetic_evidence.csv"
    evidence["_source_row"] = range(2, len(evidence) + 2)
    return pd.DataFrame(records), evidence


def template_zip():
    data, evidence = synthetic_inputs()
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("synthetic_tracker.csv", safe_csv(data.drop(columns=["source_file", "_source_row"]).to_dict("records")))
        archive.writestr("synthetic_evidence.csv", safe_csv(evidence.drop(columns=["source_file", "_source_row"]).to_dict("records")))
        archive.writestr("README.txt", "SYNTHETIC software practice only. No study evidence or performance labels.\n"
            "Tracker: county,month,menu,slot,version,show,started,discontinued,start_month.\n"
            "Evidence: county,month,menu,slot,version,text,source_ref. month means event-applicable month.\n"
            "Map status codes against your supplied dictionary first. Blank is unknown, not zero.\n"
            "Explicit extraction accepts standalone started=0/1, discontinued=0/1, start_month=YYYY-MM lines.\n"
            "Only copy assertions actually present in sources; do not invent them to make extraction succeed.\n"
            "Free narrative is retrieved but not automatically interpreted. Source references require human verification.\n")
    return output.getvalue()


def render():
    st.title("ORCCA Evidence Completion")
    st.caption("Review missing fields, inspect evidence, and export approved changes.")
    st.sidebar.page_link("app.py", label="ENVIO Interview Coding")
    st.sidebar.page_link("pages/02_ORCCA_Research_Findings.py", label="Research findings")
    st.sidebar.page_link("pages/01_ORCCA_Research_Audit.py", label="Data audit")
    st.sidebar.page_link("pages/03_ORCCA_Evidence_Completion.py", label="Evidence completion")
    st.warning("Original fields are preserved. Source-based changes and statistical scenarios are kept separate. Suppressed death counts are not filled, and H2a is not refitted.")
    private = private_uploads_allowed()
    from .workflow import current_project, apply_completion
    from .workflow_ui import navigation, project_header
    navigation()
    project = current_project(st.session_state, private)
    if st.session_state.get("workflow_active") and not project:
        st.warning("No current project inputs. Reload them in Workflow; sample completion data are not substituted.")
        st.page_link("pages/04_ORCCA_Workflow.py", label="Load inputs in Workflow")
        return
    options = (["Current web project"] if project else []) + (["Sample data", "Local research files"] if private else ["Sample data"])
    mode = st.radio("Data source", options, key="completion_mode")
    if not private:
        st.info("Public mode uses sample data only. Research uploads and external LLM generation are disabled.")
    st.download_button("Download sample files and instructions", template_zip(), "synthetic_completion_inputs.zip", "application/zip")
    try:
        if mode == "Current web project":
            project_header(project)
            if 'tracker' not in project['data']:
                st.warning('This project has no tracker. Add one in Workflow; sample data are not substituted.')
                return
            raw, evidence_frame = project['data']['tracker'], project['completion_evidence']
            run_id = project['run_id']
            classification = project['classification']
        elif mode == "Sample data":
            raw, evidence_frame = synthetic_inputs()
            run_id = digest([raw.to_json(), evidence_frame.to_json(), "completion-v1"])
            classification = "synthetic_public"
        else:
            st.info("Files are processed locally. Replacing or removing an input clears earlier candidates and decisions. Evidence needs an applicable event month; an interview date alone is insufficient.")
            input_kind = st.radio("Tracker format", ["Tidy CSV/TSV", "Original Tracker and dictionary XLSX"], key="completion_input_kind")
            if input_kind == "Tidy CSV/TSV":
                uploaded = st.file_uploader("Tracker records", type=["csv", "tsv"], key="completion_tracker")
                files = [uploaded]
            else:
                uploaded = st.file_uploader("Tracker workbook", type=["xlsx"], key="completion_workbook")
                dictionary = st.file_uploader("REDCap dictionary", type=["xlsx"], key="completion_dictionary")
                files = [uploaded, dictionary]
            evidence_upload = st.file_uploader("Source excerpts CSV with county, applicable month, strategy and version (optional)", type=["csv", "tsv"], key="completion_evidence")
            if any(f is None for f in files):
                _clear_run()
                st.info("Add the required files to continue. Previous results are cleared; sample data are not substituted.")
                return
            blobs = [f.getvalue() for f in files]
            run_id = digest([[f.name, hashlib.sha256(b).hexdigest()] for f, b in zip(files, blobs)] +
                ([[evidence_upload.name, hashlib.sha256(evidence_upload.getvalue()).hexdigest()]] if evidence_upload else []) + ["completion-v1"])
            if input_kind == "Tidy CSV/TSV":
                raw = read_table(blobs[0], uploaded.name)
            else:
                sheets = read_workbook(blobs[0], positional_sheets=("Sheet1",))
                dd = read_table(blobs[1], dictionary.name, sheet="ORCCAT_DD")
                raw, _ = tracker_workbooks_to_tidy(sheets, dd, uploaded.name)
                st.caption("The existing adapter converts the workbooks to tidy records. Check field meanings against the supplied dictionary.")
            evidence_frame = read_table(evidence_upload.getvalue(), evidence_upload.name) if evidence_upload else None
            classification = "private_local"
        data, issues = identify_issues(raw)
        documents = prepare_evidence(evidence_frame) if evidence_frame is not None else []
    except (AuditError, ValueError, OSError) as exc:
        _clear_run()
        st.error(str(exc))
        return
    if st.session_state.get("completion_run") != run_id:
        _clear_run()
        st.session_state["completion_run"] = run_id
    with st.expander("Input version"):
        st.code(run_id, language=None)
    st.session_state.setdefault("completion_candidates", {})
    st.session_state.setdefault("completion_decisions", {})
    tabs = st.tabs(["Missing fields", "Evidence and scenarios", "Review decisions", "Export reviewed data"])
    with tabs[0]:
        a, b, c = st.columns(3)
        a.metric("Records", len(data))
        b.metric("Missing or unknown fields", len(issues))
        c.metric("Eligible for review", sum(x["eligible"] for x in issues))
        view = pd.DataFrame(issues)
        if not view.empty:
            cols = ["county", "month", "menu", "slot", "field", "original_value", "eligible", "reason"]
            st.dataframe(view[cols].rename(columns={x: x.replace("_", " ").title() for x in cols}), hide_index=True)
        st.caption("Skipped questions, no-update records and dependent fields with unknown start status are held for review. Blank does not mean no. Use the audit page to check uncertain meanings in populated fields.")
    with tabs[1]:
        eligible = {x["issue_id"]: x for x in issues if x["eligible"]}
        if eligible:
            selected = st.selectbox("Field to review", list(eligible), format_func=lambda k:
                " / ".join(str(eligible[k][x]) for x in ("county", "month", "menu", "slot", "field")), key="completion_issue_" + run_id)
            issue = eligible[selected]
            backend = st.selectbox("Search method", ["bm25", "hybrid"], key="completion_backend")
            st.caption("Search is limited to the same county, applicable month, strategy and version. Automatic candidates require an explicit field assignment in the source. Narrative passages may remain unresolved.")
            if st.button("Find a source-based candidate", key="completion_retrieve"):
                try:
                    with st.spinner("Loading hybrid models and searching..." if backend == "hybrid" else "Searching source evidence..."):
                        result = evidence_candidate(issue, documents, run_id, backend)
                except Exception:
                    result = {"status": "retrieval_error_no_fallback", "candidates": [], "citations": []}
                st.session_state["completion_search"] = (selected, backend, result)
                for c in result["candidates"]:
                    st.session_state["completion_candidates"][c["candidate_id"]] = c
            search = st.session_state.get("completion_search")
            if search and search[:2] == (selected, backend):
                labels = {"source_candidate_unreviewed": "A source-based candidate is ready for review.",
                    "no_exact_county_time_strategy_version_evidence": "No evidence matches this county, month, strategy and version.",
                    "conflicting_source_assertions": "Sources disagree. Resolve the conflict before filling this field.",
                    "retrieved_but_no_explicit_field_assertion": "Related text was found, but it does not give an explicit value.",
                    "hybrid_unavailable_no_fallback": "Hybrid search is unavailable. No fallback was used.",
                    "start_after_applicable_month_requires_review": "The start date is later than the applicable month. Review the dates.",
                    "retrieval_error_no_fallback": "Search failed. No fallback was used."}
                st.write("Search result:", labels.get(search[2]["status"], search[2]["status"].replace("_", " ")))
                for doc in search[2].get("citations", []):
                    st.caption(f"{doc['source_file']} · CSV ending line {doc['line_start']} · Source reference (as supplied): {doc.get('upstream_source_as_supplied', '')}")
                    st.code(doc["text"], language=None)
                for candidate in search[2]["candidates"]:
                    _show_candidate(candidate)
            st.subheader("Statistical sensitivity scenarios")
            st.write("Explore how an unresolved binary status could affect the data. Scenario values remain separate from source-based changes.")
            with st.expander("Model and assumptions"):
                st.write("The Beta–Bernoulli baseline uses a Beta(1,1) prior and at least three earlier reports from the same county, strategy and version. It generates 20 scenarios. Exchangeability and ignorable missingness are unverified; repeated monthly states may be dependent. These are sensitivity checks, not recovered values or formal multiple imputation.")
            acknowledged = st.checkbox("I have reviewed the assumptions and will use these as sensitivity scenarios only.", key="completion_assumptions")
            seed = int(st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=2026, key="completion_seed"))
            if st.button("Create statistical scenarios", disabled=not acknowledged, key="completion_statistical"):
                try:
                    source_check = evidence_candidate(issue, documents, run_id, "bm25")
                    if source_check["status"] in {"conflicting_source_assertions", "start_after_applicable_month_requires_review"}:
                        raise AuditError("Resolve conflicting sources or dates before creating statistical scenarios.")
                    result = binary_scenarios(issue, data, run_id, assumptions_acknowledged=acknowledged, seed=seed)
                    st.session_state["completion_stat_result"] = (selected, seed, result)
                    for c in result["candidates"]:
                        st.session_state["completion_candidates"][c["candidate_id"]] = c
                except AuditError as exc:
                    st.error(str(exc))
            stats = st.session_state.get("completion_stat_result")
            if stats and stats[:2] == (selected, seed):
                st.write(stats[2]["status"].replace("_", " "))
                for candidate in stats[2]["candidates"]:
                    _show_candidate(candidate)
                if not stats[2]["candidates"]:
                    st.caption("At least three earlier comparable reports are required.")
        else:
            st.info("No fields are eligible for automatic candidates. Review branching and status rules first.")
    with tabs[2]:
        candidates = st.session_state["completion_candidates"]
        if candidates:
            chosen = st.selectbox("Candidate to review", list(candidates), format_func=lambda k:
                " / ".join(str(candidates[k][x]) for x in ("county", "month", "menu", "slot", "field")) +
                (" — source" if candidates[k]["kind"] == "source_evidence" else " — statistical scenarios"), key="completion_review_" + run_id)
            _show_candidate(candidates[chosen])
            with st.form("completion_review_form", clear_on_submit=False):
                reviewer = st.text_input("Reviewer", key="completion_reviewer")
                decision = st.radio("Decision", ["Not decided", "Accept", "Reject"], key="completion_decision_" + chosen)
                reason = st.text_area("Reason — check meaning, dates and strategy; state the purpose of statistical scenarios", key="completion_reason_" + chosen)
                save = st.form_submit_button("Save decision")
            if save:
                if decision == "Not decided" or not reviewer.strip() or not reason.strip():
                    st.error("Choose Accept or Reject and enter a reviewer and reason.")
                elif decision == "Accept" and any(cid != chosen and candidates[cid]["issue_id"] == candidates[chosen]["issue_id"] and
                        old["decision"] == "approve" for cid, old in st.session_state["completion_decisions"].items()):
                    st.error("This field already has an accepted candidate. Reject it explicitly before accepting another; earlier decisions are not changed automatically.")
                else:
                    prior = st.session_state["completion_decisions"]
                    prior[chosen] = {"candidate_id": chosen, "run_id": run_id,
                        "decision": "approve" if decision == "Accept" else "reject", "reviewer": reviewer.strip(), "reason": reason.strip(),
                        "reviewed_at_utc": datetime.now(timezone.utc).isoformat()}
                    st.success("Decision saved for this session. Original fields are unchanged.")
            st.dataframe(pd.DataFrame(st.session_state["completion_decisions"].values()), hide_index=True)
        else:
            st.info("No candidates yet. Missing evidence or a failed search leaves the field unresolved.")
    with tabs[3]:
        exported = reviewed_exports(data, issues, list(st.session_state["completion_candidates"].values()),
            list(st.session_state["completion_decisions"].values()), run_id)
        exported["missing_issues"] = issues
        exported["all_candidates"] = list(st.session_state["completion_candidates"].values())
        st.write("Only accepted source-based values appear in the __reviewed columns. Statistical values are exported separately as scenario overlays. Pending and rejected candidates are not applied.")
        st.dataframe(pd.DataFrame(exported["derived_records"]), hide_index=True)
        st.download_button("Download reviewed data, scenarios and decisions", report_bundle(exported, classification,
            {"input_fingerprint": run_id, "method_version": "completion-v1", "external_llm_calls": 0}),
            "orcca_reviewed_completion.zip", "application/zip", key="completion_download")
        st.caption("Download before closing the session. These review decisions are not independent evaluation labels, and no accuracy improvement has been established.")
        if mode == "Current web project":
            st.subheader("Apply accepted source values and rerun")
            st.write("Creates a derived tracker version and recalculates the audit. Original inputs are retained. Previous finding approvals expire; statistical scenarios are never applied to audit values.")
            if st.button("Create derived version and rerun audit", key="workflow_apply_completion"):
                try:
                    apply_completion(st.session_state, project, data, issues,
                        list(st.session_state["completion_candidates"].values()), list(st.session_state["completion_decisions"].values()))
                    st.rerun()
                except AuditError as exc:
                    st.error(str(exc))
            st.page_link("pages/04_ORCCA_Workflow.py", label="Return to Workflow to generate and review current findings")


def _show_candidate(candidate):
    st.caption(" / ".join(str(candidate[k]) for k in ("county", "month", "menu", "slot", "field")))
    if candidate["kind"] == "source_evidence":
        st.write("Proposed value:", candidate["value"])
        for source in candidate["evidence"]:
            st.caption(f"{source['source_file']} · line {source['line']}")
            st.code(source["quote"], language=None)
        st.caption("Check the source meaning and timing before accepting this value.")
    else:
        a, b = st.columns(2)
        a.metric("Earlier reports", candidate["n_support"])
        b.metric("Model Pr(status = 1)", f"{candidate['p_mean']:.1%}")
        lo, hi = candidate["p_interval_assuming_model"]
        st.caption(f"95% interval under the working model: {lo:.1%}–{hi:.1%}. This is not a validated uncertainty interval for the study.")
        st.write("Scenario values:", candidate["draws"])
    with st.expander("Candidate details and provenance"):
        st.json(candidate)


def _clear_run():
    # Inputs stay in their widgets, derived state never survives input changes.
    for key in list(st.session_state):
        if key.startswith("completion_") and key not in {"completion_mode", "completion_input_kind",
            "completion_tracker", "completion_workbook", "completion_dictionary", "completion_evidence"}:
            del st.session_state[key]
    for key, value in {"completion_reviewer": "", "completion_reason": "", "completion_decision": "Not decided",
        "completion_assumptions": False, "completion_backend": "bm25"}.items():
        st.session_state[key] = value
