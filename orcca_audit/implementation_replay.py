"""Replay implementation evidence searches and the first H3 association in a private session."""
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import t

from .retrieval import validate_documents


class SearchIndex:
    def __init__(self, documents):
        from rag_system import BM25Okapi, tokenize_for_bm25
        self.documents = documents
        tokens = [tokenize_for_bm25(doc["text"]) for doc in documents]
        self.terms = list(map(set, tokens))
        self.index = BM25Okapi(tokens) if any(tokens) else None

    def search(self, query, top_k):
        from rag_system import tokenize_for_bm25
        tokens = tokenize_for_bm25(query)
        if self.index is None or not tokens:
            return []
        terms = set(tokens)
        scores = self.index.get_scores(tokens)
        return [self.documents[i]["chunk_id"] for i in np.argsort(scores)[::-1]
                if terms.intersection(self.terms[i])][:top_k]


def first_link(frame):
    keys = ["input_version", "menu", "county", "mediator_month"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate analysis key.")
    def slope(d):
        months = pd.PeriodIndex(d.mediator_month, freq="M").asi8
        if not (pd.PeriodIndex(d.attendance_month, freq="M").asi8 == months - 1).all():
            raise ValueError("Attendance must precede the implementation month by one month.")
        x = np.column_stack([np.ones(len(d)), (months-months.mean())/12,
            pd.get_dummies(d.county, drop_first=True, dtype=float), d.attendance/10])
        if len(d) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
            raise ValueError("Insufficient independent observations")
        return np.linalg.lstsq(x, d.active_snapshot.to_numpy(float), rcond=None)[0][-1]
    rows = []
    for (version, menu), d in frame.groupby(keys[:2]):
        d = d.dropna(subset=["attendance", "active_snapshot"])
        counties = d.county.unique()
        row = dict(input_version=version, menu=menu, n=len(d), counties=len(counties))
        try:
            if len(counties) < 4:
                raise ValueError("Too few county clusters for delete-one inference")
            beta = slope(d)
            replicates = np.array([slope(d[d.county != county]) for county in counties])
            g = len(counties)
            se = np.sqrt((g-1)/g*np.square(replicates-replicates.mean()).sum())
            margin = t.ppf(.975, g-1)*se
            row.update(coefficient=beta, CI_low=beta-margin, CI_high=beta+margin,
                       p_approx=2*t.sf(abs(beta/se), g-1) if se else np.nan)
        except (ValueError, np.linalg.LinAlgError) as error:
            row["reason"] = str(error)
        rows.append(row)
    table = pd.DataFrame(rows)
    if "p_approx" in table:
        ordered = table.p_approx.dropna().sort_values()
        table.loc[ordered.index, "p_holm"] = np.minimum(1, np.maximum.accumulate(
            ordered.to_numpy() * (len(table)-np.arange(len(ordered)))))
    return table


def run(raw):
    with ZipFile(BytesIO(raw)) as z:
        def read(name):
            if z.getinfo(name).file_size > 50_000_000:
                raise ValueError("Input exceeds the review size limit")
            return z.read(name)
        memory = json.loads(read("source_memory.json"))
        cases = json.loads(read("cases_without_reference_values.json"))
        frame = pd.read_csv(BytesIO(read("PRIVATE_first_link_included_rows.csv")))
        findings = pd.read_csv(BytesIO(read("source_linked_findings.csv")))
        prior = json.loads(read("run_summary.json"))
        site_questions = json.loads(read("site_queries.json")) if "site_queries.json" in z.namelist() else []
    docs = memory["documents"]
    digest = sha256(json.dumps(docs, sort_keys=True).encode()).hexdigest()
    if digest != memory["memory_sha256"] or any(d.get("role") != "original_source" for d in docs):
        raise ValueError("Source identity or document role check failed.")
    validate_documents(docs)
    by_id = {d["chunk_id"]: d for d in docs}
    indexes = {}
    cache = {}
    def search(scope, query, k):
        if scope not in indexes:
            indexes[scope] = SearchIndex([d for d in docs if
                (not scope or d["county"] == scope[0]) and (len(scope)<2 or d["month"] == scope[1])])
        return indexes[scope].search(query, k)
    counts = Counter()
    audit = []
    for case in cases:
        county = case["county"]
        for label, scope in [("global", ()), ("county", (county,))]:
            fusion = defaultdict(float)
            for q in case["questions"]:
                ids = search(scope, q, 5)
                if scope:
                    cache[digest, scope, q, 5] = ids
                for rank, ident in enumerate(ids, 1):
                    fusion[ident] += 1/(60+rank)
            for ident in sorted(fusion, key=lambda i: (-fusion[i], i))[:5]:
                doc = by_id[ident]
                counts[label] += 1
                counts[label+"_off"] += int(doc["county"] != county)
                audit.append(dict(case_id=case["case_id"], scope=label, target_county=county,
                                  file=doc["source_file"], page=doc["page"], excerpt=doc["text"]))
    for case in cases:
        for q in case["questions"]:
            scope = (case["county"],)
            ids = cache[digest, scope, q, 5]
            if ids != search(scope, q, 5):
                raise ValueError("Replay differs from newly retrieved evidence.")
            counts["verified_cache_hits"] += 1
    sites = []
    for case in site_questions:
        scope = (case["county"], case["month"])
        key = digest, scope, case["query"], 3
        if key not in cache:
            cache[key] = search(scope, case["query"], 3)
        for ident in cache[key]:
            doc = by_id[ident]
            sites.append(dict(cell=case["cell"], county=scope[0], month=scope[1],
                              file=doc["source_file"], page=doc["page"], excerpt=doc["text"]))
    return dict(counts=dict(counts), tests=first_link(frame), citations=pd.DataFrame(audit),
                site_citations=pd.DataFrame(sites), site_questions=len(site_questions),
                findings=findings, prior_calls=int(prior["actual_model_calls"]))


def render(epoch):
    import streamlit as st
    from zipfile import BadZipFile, ZIP_DEFLATED
    with st.expander("Replay implementation evidence and H3 first-link tests"):
        st.write("Upload the private replay inputs to repeat the county-scoped BM25 search, check cached citations and refit the attendance-to-implementation models.")
        data = st.file_uploader("Implementation replay inputs", type=["zip"], key="implementation_inputs"+epoch)
        state = "implementation_replay_result"
        key = sha256(data.getbuffer()).hexdigest() if data else None
        if st.session_state.get(state, (None,))[0] != key or not data:
            st.session_state.pop(state, None)
        if st.button("Run implementation replay", disabled=data is None):
            st.session_state.pop(state, None)
            try:
                with st.spinner("Checking source versions, repeating retrieval and fitting models…"):
                    st.session_state[state] = (key, run(data.getvalue()))
            except (BadZipFile, KeyError, ValueError, TypeError, UnicodeError) as exc:
                st.error(f"Could not read the study package. {exc}")
        if state not in st.session_state:
            return
        r = st.session_state[state][1]
        c = Counter(r["counts"])
        st.write(f"Off-county citations: {c['global_off']} of {c['global']} across all counties; "
                 f"{c['county_off']} of {c['county']} after restricting to the target county. "
                 f"{c['verified_cache_hits']} cached query results matched a fresh search.")
        st.caption("This measures geographic consistency and exact reuse. Relevance requires independent review.")
        st.dataframe(r["tests"], hide_index=True)
        st.caption("Exploratory first link: change in active strategy count per ten additional attendees in the previous month. County fixed effects, linear time trend, delete-one-county intervals and Holm-adjusted p-values.")
        st.write(f"Delivery-site questions with candidate evidence: {r['site_citations'].get('cell', pd.Series(dtype=str)).nunique()} of {r['site_questions']}.")
        st.dataframe(r["site_citations"], hide_index=True)
        st.write(f"Saved review: {len(r['findings'])} candidate findings. The prior run recorded {r['prior_calls']} model calls. This replay uses BM25 and makes no model calls.")
        st.dataframe(r["findings"], hide_index=True)
        buf = BytesIO()
        with ZipFile(buf, "w", ZIP_DEFLATED) as z:
            for name in ("tests","citations","site_citations","findings"):
                z.writestr(name+".csv",r[name].to_csv(index=False))
            z.writestr("retrieval_counts.json",json.dumps(r["counts"],indent=2))
        st.download_button("Download implementation replay results",buf.getvalue(),"implementation_replay_results.zip")
