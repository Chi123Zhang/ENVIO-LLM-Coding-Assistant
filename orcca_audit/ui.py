"""ORCCA page, isolated from ENVIO Interview Coding state and exports."""
import json
import hashlib
import os
from io import BytesIO
import zipfile

import pandas as pd
import streamlit as st

from . import VERSION
from .demo import demo_data, public_documents
from .engine import AuditError, audit_consistency, verify_exposures, compare_specifications
from .exports import report_bundle
from .extensions import EXTENSIONS
from .io import read_table, read_workbook, normalize_mortality, extract_document
from .policy import private_uploads_allowed, external_generation_allowed
from .retrieval import PUBLIC_QUESTIONS, retrieve, explain_with_external_model


def _show_rows(rows):
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No records in this result section.")


def render():
    st.title("ORCCA Research Audit")
    st.caption("Source-linked data and measurement review · No model fitting")
    st.sidebar.page_link("app.py", label="ENVIO Interview Coding")
    st.sidebar.page_link("pages/01_ORCCA_Research_Audit.py", label="ORCCA Research Audit")
    private = private_uploads_allowed()
    st.info("Local private mode: uploads stay in this session; external generation for uploaded research evidence is disabled."
        if private else "Public Demo · Every bundled value and document is synthetic. Do not upload ORCCA research data to this public app.")
    st.warning("Verified source checks, candidate-rule matches, model explanations and unknowns are different evidence levels. This app does not reproduce H2a or establish improved RAG accuracy.")
    if st.button("Clear this audit session", key="orcca_clear"):
        for key in list(st.session_state):
            if key.startswith("orcca_") and key != "orcca_clear":
                del st.session_state[key]
        st.rerun()
    data = demo_data()
    classification = "synthetic_public_demo"
    documents = public_documents()
    provenance = {"module_version": VERSION, "synthetic": True}
    if private:
        st.subheader("Local input sources")
        st.caption("Normalized CSV/TSV tables, or the original local Tracker + dictionary XLSX pair. Raw files are not saved to the repository or a shared index.")
        supplied = {}
        for role, label, types in [
            ("panel", "Panel table", ["csv", "tsv", "xls", "xlsx"]),
            ("monthly", "Monthly mortality source", ["csv", "tsv", "xls", "xlsx"]),
            ("annual", "Annual mortality source", ["csv", "tsv", "xls", "xlsx"]),
            ("tracker", "Normalized tidy tracker (optional)", ["csv", "tsv"]),
            ("raw_tracker", "Original Tracker workbook (optional)", ["xlsx"]),
            ("dictionary", "Data dictionary workbook (optional)", ["xlsx"]),
            ("specs", "Two model specifications JSON (optional)", ["json"]),
            ("evidence", "Evidence PDF or text (optional)", ["pdf", "txt", "md"])]:
            supplied[role] = st.file_uploader(label, type=types, key="orcca_upload_" + role)
        if any(supplied.values()):
            required = ["panel", "monthly", "annual"]
            if not all(supplied[k] is not None for k in required):
                st.error("Supply panel, monthly and annual sources together. Demo data will not be substituted.")
                return
            try:
                if (supplied["raw_tracker"] is None) != (supplied["dictionary"] is None):
                    raise AuditError("Supply the original Tracker and dictionary together.")
                if supplied["tracker"] is not None and supplied["raw_tracker"] is not None:
                    raise AuditError("Choose either a normalized tracker or the workbook pair, not both.")
                data = {k: read_table(supplied[k].getvalue(), supplied[k].name) for k in required}
                data["monthly"], data["annual"] = normalize_mortality(data["monthly"]), normalize_mortality(data["annual"])
                if supplied["tracker"] is not None:
                    data["tracker"] = read_table(supplied["tracker"].getvalue(), supplied["tracker"].name)
                elif supplied["raw_tracker"] is not None and supplied["dictionary"] is not None:
                    from .adapters import tracker_workbooks_to_tidy
                    dictionary = read_table(supplied["dictionary"].getvalue(), supplied["dictionary"].name, sheet="ORCCAT_DD")
                    data["tracker"], refs = tracker_workbooks_to_tidy(read_workbook(supplied["raw_tracker"].getvalue(), positional_sheets=("Sheet1",)), dictionary, supplied["raw_tracker"].name)
                    data["field_references"] = refs.to_dict("records")
                    with st.expander("Dictionary field provenance"):
                        st.dataframe(refs, hide_index=True)
                data["specifications"] = json.loads(supplied["specs"].getvalue()) if supplied["specs"] is not None else []
                if supplied["specs"] is not None:
                    compare_specifications(data["specifications"])  # Validate before rendering any tab.
                documents = extract_document(supplied["evidence"].getvalue(), supplied["evidence"].name) if supplied["evidence"] is not None else []
                classification, provenance = "private_local_session", {"module_version": VERSION, "synthetic": False,
                    "inputs": {k: dict(v.attrs) for k, v in data.items() if isinstance(v, pd.DataFrame)},
                    "uploaded_sources": {k: {"source_file": v.name, "sha256": hashlib.sha256(v.getvalue()).hexdigest()}
                        for k, v in supplied.items() if v is not None}}
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, AuditError) else "Input parsing failed; no audit or demo substitution was performed.")
                return
        else:
            st.caption("No private inputs supplied; the current results use the synthetic demo.")
    else:
        # No file_uploader is constructed in public mode: private documents cannot
        # accidentally be sent to the server through the ORCCA page.
        with st.expander("Download the synthetic input fixture"):
            buf = BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for name in ["panel", "monthly", "annual", "tracker"]:
                    z.writestr(f"synthetic_{name}.csv", data[name].drop(columns=["source_file", "_source_row"], errors="ignore").to_csv(index=False))
            st.download_button("Download synthetic inputs", buf.getvalue(), "synthetic_audit_inputs.zip", "application/zip", key="orcca_demo_download")
    results = {"field_references": data["field_references"]} if "field_references" in data else {}
    tabs = st.tabs(["Data Consistency Audit", "EBP Exposure Verification", "Model Specification Comparison",
                    "Evidence-Grounded RAG", "Audit Report Export"])
    with tabs[0]:
        st.subheader("Data Consistency Audit")
        try:
            results["consistency"] = audit_consistency(data["panel"], data["monthly"], data["annual"])
            summary = results["consistency"]["summary"]
            a, b, c = st.columns(3)
            a.metric("Panel rows", summary["panel_rows"])
            b.metric("Counties", summary["counties"])
            c.metric("Rows matching supplied sources", summary["verified_rows"])
            st.json(summary)
            _show_rows(results["consistency"]["annual_windows"])
            st.caption("Annual compatibility includes outside-window months. An impossible partial-window=annual equality does not prove the original model used that equality.")
            with st.expander("Row-level source checks and anomalies"):
                _show_rows(results["consistency"]["row_checks"])
                _show_rows(results["consistency"]["anomalies"])
        except AuditError as exc:
            st.error(str(exc)); results["consistency"] = {"status": "input_error", "message": str(exc)}
    with tabs[1]:
        st.subheader("EBP Exposure Verification")
        if "tracker" not in data:
            st.info("Supply a tidy tracker or the Tracker/dictionary workbook pair. Exposure verification has not run.")
        else:
            try:
                results["exposures"] = verify_exposures(data["panel"], data["tracker"])
                st.json(results["exposures"]["summary"])
                st.caption("L1 uses t−1; L13 averages t−1,t−2,t−3. A permissive candidate match is not a confirmed status or original algorithm.")
                _show_rows(results["exposures"]["lag_checks"])
                st.subheader("Needs human review")
                _show_rows(results["exposures"]["anomalies"])
            except AuditError as exc:
                st.error(str(exc)); results["exposures"] = {"status": "input_error", "message": str(exc)}
    with tabs[2]:
        st.subheader("Model Specification Comparison")
        st.caption("Each specification needs its own source locator. Missing implementation details remain unknown; RR scales and outcome windows are not assumed interchangeable.")
        if len(data["specifications"]) == 2:
            try:
                results["specifications"] = compare_specifications(data["specifications"])
                _show_rows(results["specifications"])
            except AuditError as exc:
                st.error(str(exc))
        else:
            st.info("Upload exactly two source-linked specification records. No private study specification is bundled.")
    with tabs[3]:
        st.subheader("Evidence-Grounded RAG")
        preset = st.selectbox("Synthetic explanation question", PUBLIC_QUESTIONS, key="orcca_preset")
        query = st.text_input("Retrieval query (English or field identifier)", value=preset, key="orcca_query")
        backend = st.selectbox("Retrieval backend", ["bm25", "hybrid"], key="orcca_backend",
            help="Hybrid reuses ENVIO FAISS/BM25/reranking with cached local weights. Missing weights produce an explicit failure.")
        fingerprint = hashlib.sha256(json.dumps([query, backend, [(d["chunk_id"], d["sha256"], d["source_file"]) for d in documents]], sort_keys=True).encode()).hexdigest()
        if st.button("Retrieve source evidence", key="orcca_retrieve"):
            st.session_state.pop("orcca_explanation", None)
            try:
                st.session_state["orcca_retrieval"] = retrieve(query, documents, backend)
            except Exception:
                st.session_state["orcca_retrieval"] = {"status": "retrieval_error_no_fallback", "backend": backend, "citations": []}
            st.session_state["orcca_retrieval_fingerprint"] = fingerprint
        result = st.session_state.get("orcca_retrieval")
        # Do not show prior dataset/query results after input or query changes.
        if result and st.session_state.get("orcca_retrieval_fingerprint") == fingerprint and all(
            any(c["sha256"] == d["sha256"] for d in documents) for c in result.get("citations", [])):
            st.write("Retrieval status:", result["status"])
            if result["status"] != "retrieved_evidence_unreviewed":
                st.warning("Evidence retrieval did not complete. No RAG answer or fallback success is reported.")
            for citation in result.get("citations", []):
                st.markdown(f"**{citation['source_file']}** · page {citation.get('page') or 'n/a'}, lines {citation['line_start']}–{citation['line_end']}")
                st.code(citation["text"], language=None)
                st.caption("Source hash: " + citation["sha256"])
            results["retrieval"] = result
            enabled = query in PUBLIC_QUESTIONS and external_generation_allowed(result.get("citations", [])) and bool(os.environ.get("ORCCA_LLM_MODEL"))
            st.caption("Extractive evidence above is not an LLM explanation. Uploaded research evidence is never eligible for this external-model demo.")
            if st.button("Generate synthetic-only explanation", key="orcca_generate", disabled=not enabled):
                st.session_state["orcca_explanation"] = explain_with_external_model(query, result, os.environ["ORCCA_LLM_MODEL"])
            if "orcca_explanation" in st.session_state:
                st.json(st.session_state["orcca_explanation"])
                results["model_explanation"] = st.session_state["orcca_explanation"]
    with tabs[4]:
        st.subheader("Audit Report Export")
        st.caption("Session-scoped report, source checks and anomaly tables. Raw uploads are excluded. Exports may still contain study identifiers in local private mode; handle accordingly.")
        st.download_button("Download audit report and anomalies", report_bundle(results, classification, provenance),
            "orcca_audit_report.zip", "application/zip", key="orcca_report_download")
        st.info("Human evaluation: not run. No labels or RAG accuracy/time-saving metrics are bundled.")
    with st.expander("Future research modules"):
        st.json(EXTENSIONS)
