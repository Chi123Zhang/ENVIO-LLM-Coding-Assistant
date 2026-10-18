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
    st.caption("识别缺失 → 查找证据 → 提出候选 → 人工确认 → 导出派生数据")
    st.sidebar.page_link("app.py", label="ENVIO Interview Coding")
    st.sidebar.page_link("pages/02_ORCCA_Research_Findings.py", label="研究发现与证据")
    st.warning("原始字段保留。来源补全与统计情景分开；不补出被抑制的死亡真值，不重新拟合 H2a。")
    private = private_uploads_allowed()
    mode = st.radio("材料模式", ["模拟练习", "本地研究材料"] if private else ["模拟练习"], key="completion_mode")
    if not private:
        st.info("公开环境仅开放模拟练习，真实数据上传与外部 LLM 生成关闭。")
    st.download_button("下载模拟文件与格式说明", template_zip(), "synthetic_completion_inputs.zip", "application/zip")
    try:
        if mode == "模拟练习":
            raw, evidence_frame = synthetic_inputs()
            run_id = digest([raw.to_json(), evidence_frame.to_json(), "completion-v1"])
            classification = "synthetic_public"
        else:
            st.info("仅在本机处理；更换或删除输入将清除此前的候选和确认记录。证据必须有事件适用月份，不能用访谈日期代替。")
            input_kind = st.radio("Tracker 格式", ["规范化 CSV/TSV", "原始 Tracker + 字典 XLSX"], key="completion_input_kind")
            if input_kind == "规范化 CSV/TSV":
                uploaded = st.file_uploader("待补全 Tracker", type=["csv", "tsv"], key="completion_tracker")
                files = [uploaded]
            else:
                uploaded = st.file_uploader("原始 Tracker 工作簿", type=["xlsx"], key="completion_workbook")
                dictionary = st.file_uploader("对应 REDCap 字典", type=["xlsx"], key="completion_dictionary")
                files = [uploaded, dictionary]
            evidence_upload = st.file_uploader("带县、适用月份、策略及版本的来源摘录 CSV（可选）", type=["csv", "tsv"], key="completion_evidence")
            if any(f is None for f in files):
                _clear_run()
                st.info("请提供必需文件；未加载模拟数据或保留旧结果。")
                return
            blobs = [f.getvalue() for f in files]
            run_id = digest([[f.name, hashlib.sha256(b).hexdigest()] for f, b in zip(files, blobs)] +
                ([[evidence_upload.name, hashlib.sha256(evidence_upload.getvalue()).hexdigest()]] if evidence_upload else []) + ["completion-v1"])
            if input_kind == "规范化 CSV/TSV":
                raw = read_table(blobs[0], uploaded.name)
            else:
                sheets = read_workbook(blobs[0], positional_sheets=("Sheet1",))
                dd = read_table(blobs[1], dictionary.name, sheet="ORCCAT_DD")
                raw, _ = tracker_workbooks_to_tidy(sheets, dd, uploaded.name)
                st.caption("原始工作簿经既有适配器转为规范化记录；字段映射仍需字典语义复核。")
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
    st.caption("本次输入标识：" + run_id)
    st.session_state.setdefault("completion_candidates", {})
    st.session_state.setdefault("completion_decisions", {})
    tabs = st.tabs(["缺失与不确定项", "证据与统计候选", "人工确认", "派生数据导出"])
    with tabs[0]:
        st.write({"records": len(data), "issues": len(issues), "eligible": sum(x["eligible"] for x in issues),
            "evidence_excerpts": len(documents)})
        st.dataframe(pd.DataFrame(issues), hide_index=True)
        st.caption("跳题、未更新或启动状态不明确的依赖字段暂缓处理；不把空值当作否。已有数值的语义不确定性仍需原审查页核查。")
    with tabs[1]:
        eligible = {x["issue_id"]: x for x in issues if x["eligible"]}
        if eligible:
            selected = st.selectbox("选择待补全项", list(eligible), format_func=lambda k:
                " / ".join(str(eligible[k][x]) for x in ("county", "month", "menu", "slot", "field")), key="completion_issue_" + run_id)
            issue = eligible[selected]
            backend = st.selectbox("证据检索方式", ["bm25", "hybrid"], key="completion_backend")
            st.caption("先按县、事件适用月份、策略及版本精确筛选，再复用 ENVIO 检索。自动提取只接受原文独立行中的明确字段赋值；自由叙述命中后仍可能无法确定。")
            if st.button("检索并提出来源候选", key="completion_retrieve"):
                try:
                    result = evidence_candidate(issue, documents, run_id, backend)
                except Exception:
                    result = {"status": "retrieval_error_no_fallback", "candidates": [], "citations": []}
                st.session_state["completion_search"] = (selected, backend, result)
                for c in result["candidates"]:
                    st.session_state["completion_candidates"][c["candidate_id"]] = c
            search = st.session_state.get("completion_search")
            if search and search[:2] == (selected, backend):
                st.write("本次检索状态：", search[2]["status"])
                for doc in search[2].get("citations", []):
                    st.caption(f"{doc['source_file']} · CSV物理结束行 {doc['line_start']} · 上游定位（用户提供）：{doc.get('upstream_source_as_supplied', '')}")
                    st.code(doc["text"], language=None)
                for candidate in search[2]["candidates"]:
                    st.json(candidate)
            st.subheader("统计敏感性情景（不当作真实值）")
            st.write("Beta(1,1) 先验；使用同县、策略、版本的更早已报告二元状态，至少3条。生成20个情景并显示工作假设下的后验概率区间。重复月状态相关，不能当作独立样本的实证保证。")
            acknowledged = st.checkbox("我理解同组可交换、可忽略缺失的假设尚未验证；这里只做敏感性情景，不能恢复真值或替代正式多重插补。", key="completion_assumptions")
            seed = int(st.number_input("随机种子", min_value=0, max_value=2**31 - 1, value=2026, key="completion_seed"))
            if st.button("生成统计情景候选", disabled=not acknowledged, key="completion_statistical"):
                try:
                    source_check = evidence_candidate(issue, documents, run_id, "bm25")
                    if source_check["status"] in {"conflicting_source_assertions", "start_after_applicable_month_requires_review"}:
                        raise AuditError("来源存在冲突或时间矛盾，不能用统计补全绕过；请先解决原始证据。")
                    result = binary_scenarios(issue, data, run_id, assumptions_acknowledged=acknowledged, seed=seed)
                    st.session_state["completion_stat_result"] = (selected, seed, result)
                    for c in result["candidates"]:
                        st.session_state["completion_candidates"][c["candidate_id"]] = c
                except AuditError as exc:
                    st.error(str(exc))
            stats = st.session_state.get("completion_stat_result")
            if stats and stats[:2] == (selected, seed):
                st.json(stats[2])
        else:
            st.info("没有可自动提出补全候选的字段；请先复核跳题及状态规则。")
    with tabs[2]:
        candidates = st.session_state["completion_candidates"]
        if candidates:
            chosen = st.selectbox("待人工判定的候选", list(candidates), format_func=lambda k:
                candidates[k]["issue_id"] + " / " + candidates[k]["kind"] + " / " + k[:8], key="completion_review_" + run_id)
            st.json(candidates[chosen])
            with st.form("completion_review_form", clear_on_submit=False):
                reviewer = st.text_input("标注者／复核者", key="completion_reviewer")
                decision = st.radio("人工决定", ["暂不决定", "接受", "拒绝"], key="completion_decision_" + chosen)
                reason = st.text_area("判定理由（来源含义、时间与策略是否一致；统计候选须说明用途）", key="completion_reason_" + chosen)
                save = st.form_submit_button("保存本次人工判定")
            if save:
                if decision == "暂不决定" or not reviewer.strip() or not reason.strip():
                    st.error("请明确选择接受或拒绝，并填写复核者及理由。")
                elif decision == "接受" and any(cid != chosen and candidates[cid]["issue_id"] == candidates[chosen]["issue_id"] and
                        old["decision"] == "approve" for cid, old in st.session_state["completion_decisions"].items()):
                    st.error("此字段已有接受的候选。请先明确拒绝原候选，再接受新的候选；不会自动改写原判定。")
                else:
                    prior = st.session_state["completion_decisions"]
                    prior[chosen] = {"candidate_id": chosen, "run_id": run_id,
                        "decision": "approve" if decision == "接受" else "reject", "reviewer": reviewer.strip(), "reason": reason.strip(),
                        "reviewed_at_utc": datetime.now(timezone.utc).isoformat()}
                    st.success("人工判定已保存到本次会话；原始字段未修改。")
            st.dataframe(pd.DataFrame(st.session_state["completion_decisions"].values()), hide_index=True)
        else:
            st.info("尚无候选。检索失败或证据不足不会生成虚假补全。")
    with tabs[3]:
        exported = reviewed_exports(data, issues, list(st.session_state["completion_candidates"].values()),
            list(st.session_state["completion_decisions"].values()), run_id)
        exported["missing_issues"] = issues
        exported["all_candidates"] = list(st.session_state["completion_candidates"].values())
        st.write("__reviewed 列仅加入人工接受的来源补全；统计值只出现在独立的情景覆盖表中。未确认或拒绝的候选不补入数据。")
        st.dataframe(pd.DataFrame(exported["derived_records"]), hide_index=True)
        st.download_button("下载派生数据、统计情景与复核记录", report_bundle(exported, classification,
            {"input_fingerprint": run_id, "method_version": "completion-v1", "external_llm_calls": 0}),
            "orcca_reviewed_completion.zip", "application/zip", key="completion_download")
        st.caption("仅保存在本次浏览器会话；请及时下载。此处复核记录不是正式独立评价标签，也未证明补全提高准确率。")


def _clear_run():
    # Inputs stay in their widgets, derived state never survives input changes.
    for key in list(st.session_state):
        if key.startswith("completion_") and key not in {"completion_mode", "completion_input_kind",
            "completion_tracker", "completion_workbook", "completion_dictionary", "completion_evidence"}:
            del st.session_state[key]
    for key, value in {"completion_reviewer": "", "completion_reason": "", "completion_decision": "暂不决定",
        "completion_assumptions": False, "completion_backend": "bm25"}.items():
        st.session_state[key] = value
