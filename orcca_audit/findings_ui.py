"""Research results and original-source retrieval, separate from demo arithmetic."""
import os

import pandas as pd
import streamlit as st

from .engine import AuditError
from .exports import report_bundle
from .findings import LEVELS, evidence_for, load_bundle, retrieval_fingerprint
from .policy import private_uploads_allowed
from .retrieval import retrieve


def _evidence(doc):
    kind = {"original_excerpt": "Source excerpt", "deterministic_receipt": "Calculation record",
            "approved_summary": "Approved public summary"}[doc["evidence_kind"]]
    st.markdown(f"**{doc['source_file']}** · {kind}")
    page = f"Page {doc['page']}, " if doc.get("page") else ""
    st.caption(f"{page}lines/rows {doc['line_start']}–{doc['line_end']}")
    if doc.get("text_version"):
        st.caption(doc["text_version"])
    st.code(doc["text"], language=None)
    with st.expander("Exact source reference and checksum"):
        st.write(doc.get("locator", "See the page and row reference above."))
        st.code(doc["sha256"], language=None)


def render():
    st.title("ORCCA Research Findings")
    st.caption("Findings from the supplied materials, with evidence and open questions.")
    st.sidebar.page_link("app.py", label="ENVIO Interview Coding")
    st.sidebar.page_link("pages/01_ORCCA_Research_Audit.py", label="Data audit")
    st.sidebar.page_link("pages/02_ORCCA_Research_Findings.py", label="Research findings")
    st.sidebar.page_link("pages/03_ORCCA_Evidence_Completion.py", label="Evidence completion")
    private = private_uploads_allowed()
    path = os.environ.get("ORCCA_FINDINGS_BUNDLE")
    approved_hash = os.environ.get("ORCCA_PUBLIC_FINDINGS_SHA256")
    if not path and not private:
        from .public_review import bundled_public_review
        path, approved_hash = bundled_public_review()
    if not path:
        st.info("No research review is configured. Demo results are not used as study findings.")
        st.write("Research summaries require source checks and researcher review before publication.")
        return
    try:
        bundle, digest = load_bundle(path, private_mode=private,
            approved_sha256=approved_hash)
        from .english_export import english_results
        bundle["documents"] = english_results({"evidence_excerpts": bundle["documents"]})["evidence_excerpts"]
    except (AuditError, OSError) as exc:
        st.error(str(exc) if isinstance(exc, AuditError) else "The research review could not be read. No other data were substituted.")
        return
    st.info("Local research workspace. Source excerpts stay on this computer." if private else
            "Public summary. Search covers approved summaries only; original study files are not published.")
    st.caption(f"Reviewed {bundle.get('review_date', 'date not supplied')}")
    with st.expander("Review version"):
        st.code(digest, language=None)
    st.warning("These are data and methods checks. H2a has not been reproduced, and H3–H5 have not been tested.")
    if st.button("Reset search", key="findings_reset"):
        for key in list(st.session_state):
            if key.startswith("findings_") and key != "findings_reset":
                del st.session_state[key]
        # Explicit values also reset existing browser widgets; deleting keys alone
        # can let Streamlit restore their previous frontend values.
        for question in bundle["questions"]:
            st.session_state["findings_query_" + question["question_id"]] = question["query"]
        if bundle["questions"]:
            st.session_state["findings_question"] = bundle["questions"][0]["question_id"]
        else:
            st.session_state["findings_query"] = ""
        st.session_state["findings_backend"] = "bm25"
        st.session_state["findings_reset_notice"] = True
        st.rerun()
    st.caption("Resets the question, query, search method and results. Findings and sources stay available.")
    if st.session_state.pop("findings_reset_notice", False):
        st.success("Search reset. Defaults restored and previous results cleared. Findings and sources are unchanged.")
    tabs = st.tabs(["Findings", "Evidence search", "H3–H5 readiness", "Methods and evaluation", "Export"])
    with tabs[0]:
        st.write("Each finding includes its basis, limits and supporting records.")
        for item in bundle["findings"]:
            st.subheader(item["finding_id"] + " · " + item["title"])
            st.caption(LEVELS[item["level"]])
            st.write(item["statement"])
            st.markdown("**Basis:** " + item["verification_method"])
            st.markdown("**Limits:** " + item["limitations"])
            if item.get("metrics"):
                st.dataframe(pd.DataFrame(item["metrics"]), hide_index=True)
            with st.expander("Sources and supporting records", expanded=False):
                for doc in evidence_for(item, bundle):
                    _evidence(doc)
    retrieval_result = None
    with tabs[1]:
        questions = {q["question_id"]: q for q in bundle["questions"]}
        if questions:
            selected = st.selectbox("Review question", list(questions),
                format_func=lambda key: questions[key]["question"], key="findings_question")
            q = questions[selected]
            st.caption("Evidence coverage: " + {"answerable": "Supported within this review", "partial": "Partly supported",
                "insufficient_evidence": "Insufficient evidence"}[q["status"]])
            st.markdown("**Review note — pending researcher confirmation**")
            st.write(q["review_answer"])
            st.caption("This note was prepared in advance with AI assistance and needs researcher review. Search retrieves source records; it does not generate this answer. Review notes are excluded from the index.")
            query_key = "findings_query_" + selected
            st.session_state.setdefault(query_key, q["query"])
            query = st.text_input("Search terms or field names", key=query_key)
        else:
            query = st.text_input("Search terms or field names", key="findings_query")
        backend = st.selectbox("Search method", ["bm25", "hybrid"], key="findings_backend")
        if backend == "hybrid":
            st.caption("Hybrid runs in this app. The first public search may download fixed model weights; search text is not sent to an inference service.")
        fingerprint = retrieval_fingerprint(digest, query, backend)
        if st.button("Find supporting records", key="findings_retrieve"):
            try:
                with st.spinner("Loading search models and finding records..." if backend == "hybrid" else "Finding records..."):
                    result = retrieve(query, bundle["documents"], backend)
            except Exception:
                result = {"status": "retrieval_error_no_fallback", "backend": backend, "citations": []}
            st.session_state["findings_result"] = result
            st.session_state["findings_fingerprint"] = fingerprint
        if st.session_state.get("findings_fingerprint") == fingerprint:
            retrieval_result = st.session_state.get("findings_result")
            if retrieval_result:
                labels = {"retrieved_evidence_unreviewed": "Supporting records found. Check their relevance before use.",
                    "hybrid_unavailable_no_fallback": "Hybrid search is unavailable. No fallback was used.",
                    "empty_index": "No source records are indexed.",
                    "retrieval_failed": "No matching records found.",
                    "unsupported_query_language_or_empty": "Enter English search terms or a field name.",
                    "retrieval_error_no_fallback": "Search failed. No fallback was used."}
                st.write("Search status:", labels.get(retrieval_result["status"], retrieval_result["status"].replace("_", " ")))
                if retrieval_result["status"] != "retrieved_evidence_unreviewed":
                    st.warning("No evidence was retrieved. No fallback answer was generated.")
                for doc in retrieval_result.get("citations", []):
                    _evidence(doc)
        st.info("Check whether each passage supports the claim. This page does not call an external LLM or infer causal effects.")
    with tabs[2]:
        for h in bundle["hypotheses"]:
            st.subheader(h["hypothesis"] + " · " + h["definition"])
            st.markdown("**Available:** " + h["available"])
            st.markdown("**Missing:** " + h["missing"])
            st.markdown("**Next step:** " + h["next_step"])
        st.caption("Text needs a defensible county and applicable date before it can become a monthly variable.")
    with tabs[3]:
        st.subheader("How the findings were produced")
        st.write("Counts and consistency results come from Python checks. Design comparisons come from reviewing the supplied documents. Retrieval helps locate supporting passages; it does not establish the findings or fill unknown values by itself.")
        st.subheader("What the workflow does")
        st.dataframe(pd.DataFrame([
            {"Task": "Source tracing", "Current implementation": "File, page, row or field references accompany source excerpts."},
            {"Task": "Cross-file checks", "Current implementation": "Source records and reproducible checks are shown with their limits."},
            {"Task": "Evidence gaps", "Current implementation": "Unknowns, review items and failed searches are reported separately."},
            {"Task": "LLM generation", "Current implementation": "Disabled here. Review notes are prepared in advance, not generated during search."},
            {"Task": "Accuracy and time", "Current implementation": "Human evaluation is incomplete; no improvement has been measured."},
        ]), hide_index=True)
        evaluation = bundle.get("evaluation", {})
        a, b = st.columns(2)
        a.metric("Pilot cases", evaluation.get("development_pilot_cases", 0))
        b.metric("Human-labelled cases", evaluation.get("human_labelled_cases", 0))
        with st.expander("Evaluation record"):
            st.json(evaluation)
        st.write("Planned comparison: rules, an LLM without retrieval, and an LLM with retrieval. Two independent reviewers will label cases and resolve disagreements. Test splits should separate counties or documents; reference answers must stay out of the index.")
        st.caption("Pilot size is not the number of labelled cases. Software checks and research performance are reported separately.")
    with tabs[4]:
        st.write("After extracting the ZIP, open START_HERE.html for the formatted English report. All exported text is English. Use evidence_index.csv for source locations; translated records are marked with original and translated text hashes.")
        exported = {"research_findings": bundle["findings"], "hypotheses": bundle["hypotheses"],
            "evidence_excerpts": bundle["documents"],
            "sources": bundle["sources"], "human_evaluation": bundle.get("evaluation", {})}
        if retrieval_result:
            exported["retrieval"] = retrieval_result
        st.download_button("Download findings and evidence", report_bundle(exported, bundle["classification"],
            {"review_date": bundle.get("review_date"), "bundle_sha256": digest,
             "audit_run": bundle.get("audit_run", {})}),
            "orcca_research_findings.zip", "application/zip", key="findings_download")
        st.caption("CSV and JSON files are technical exports. Local reports may contain private source text; share only with authorized researchers. Original files are unchanged.")
