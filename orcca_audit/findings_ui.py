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
    kind = {"original_excerpt": "原始材料摘录", "deterministic_receipt": "程序计算记录",
            "approved_summary": "经批准公开的二级摘要"}[doc["evidence_kind"]]
    st.markdown(f"**{doc['source_file']}** · {kind}")
    st.caption(doc.get("locator", f"page {doc.get('page')}, lines {doc['line_start']}–{doc['line_end']}"))
    st.code(doc["text"], language=None)
    st.caption("摘录 SHA-256: " + doc["sha256"])


def render():
    st.title("ORCCA Research Findings")
    st.caption("从真实材料发现了什么 · 原始证据与计算记录 · H3–H5 数据准备")
    st.sidebar.page_link("app.py", label="ENVIO Interview Coding")
    st.sidebar.page_link("pages/01_ORCCA_Research_Audit.py", label="模拟流程与数据核查")
    st.sidebar.page_link("pages/02_ORCCA_Research_Findings.py", label="真实研究发现与证据")
    private = private_uploads_allowed()
    path = os.environ.get("ORCCA_FINDINGS_BUNDLE")
    if not path:
        st.info("尚未配置可展示的真实研究审查摘要。此页不会用模拟结果代替真实发现。")
        st.write("模拟演示展示工具如何运行；研究发现需要经核查的材料、来源和研究者复核。公开研究摘要须单独确认后发布。")
        return
    try:
        bundle, digest = load_bundle(path, private_mode=private,
            approved_sha256=os.environ.get("ORCCA_PUBLIC_FINDINGS_SHA256"))
    except (AuditError, OSError) as exc:
        st.error(str(exc) if isinstance(exc, AuditError) else "研究摘要文件无法读取；没有替代或加载其他数据。")
        return
    st.info("本地研究审查：包含真实材料及原文摘录，仅在本机查看。" if private else
            "公开研究审查摘要：检索范围仅限已批准公开的摘要，原始研究材料没有发布。")
    st.caption(f"材料日期：{bundle.get('review_date', '未提供')} · 审查包 SHA-256: {digest}")
    st.warning("下列内容是数据与方法审查发现。候选计算相容不等于原算法正确；未复现 H2a，也未检验 H3–H5 的效应。")
    if st.button("重置证据查询", key="findings_reset"):
        for key in list(st.session_state):
            if key.startswith("findings_") and key != "findings_reset":
                del st.session_state[key]
        st.rerun()
    tabs = st.tabs(["研究发现", "证据检索与解释", "H3–H5 材料准备", "RAG 能力与评价", "报告导出"])
    with tabs[0]:
        st.write("每条结论列出支持范围和限制；展开后可查看原文、程序计算记录及来源定位。")
        for item in bundle["findings"]:
            st.subheader(item["finding_id"] + " · " + item["title"])
            st.caption(LEVELS[item["level"]])
            st.write(item["statement"])
            st.markdown("**如何得到：** " + item["verification_method"])
            st.markdown("**仍不能断言：** " + item["limitations"])
            if item.get("metrics"):
                st.dataframe(pd.DataFrame(item["metrics"]), hide_index=True)
            with st.expander("查看这条发现的来源与证据", expanded=False):
                for doc in evidence_for(item, bundle):
                    _evidence(doc)
    retrieval_result = None
    with tabs[1]:
        questions = {q["question_id"]: q for q in bundle["questions"]}
        if questions:
            selected = st.selectbox("选择研究问题", list(questions),
                format_func=lambda key: questions[key]["question"], key="findings_question")
            q = questions[selected]
            st.caption("可回答性：" + {"answerable": "可回答", "partial": "部分可回答",
                "insufficient_evidence": "证据不足"}[q["status"]])
            st.markdown("**已整理的审查解释（非 LLM 生成）：**")
            st.write(q["review_answer"])
            st.caption("这段解释来自审查人员整理与程序结果；下方检索单独返回材料，不将解释本身放入索引。")
            query = st.text_input("证据检索词（英文或字段名，可编辑）", value=q["query"],
                key="findings_query_" + selected)
        else:
            query = st.text_input("证据检索词（英文或字段名）", key="findings_query")
        backend = st.selectbox("检索方式", ["bm25", "hybrid"], key="findings_backend")
        fingerprint = retrieval_fingerprint(digest, query, backend)
        if st.button("检索支持材料", key="findings_retrieve"):
            try:
                result = retrieve(query, bundle["documents"], backend)
            except Exception:
                result = {"status": "retrieval_error_no_fallback", "backend": backend, "citations": []}
            st.session_state["findings_result"] = result
            st.session_state["findings_fingerprint"] = fingerprint
        if st.session_state.get("findings_fingerprint") == fingerprint:
            retrieval_result = st.session_state.get("findings_result")
            if retrieval_result:
                st.write("检索状态：", retrieval_result["status"])
                if retrieval_result["status"] != "retrieved_evidence_unreviewed":
                    st.warning("本次没有成功取得证据。不会将检索失败标为 RAG 成功，也不会自动生成答案。")
                for doc in retrieval_result.get("citations", []):
                    _evidence(doc)
        st.info("检索命中不保证证据支持结论。该页面没有外部模型调用、自动补数或因果结论生成。")
    with tabs[2]:
        for h in bundle["hypotheses"]:
            st.subheader(h["hypothesis"] + " · " + h["definition"])
            st.markdown("**已有材料：** " + h["available"])
            st.markdown("**仍缺：** " + h["missing"])
            st.markdown("**下一步：** " + h["next_step"])
        st.caption("材料可得性不是统计识别保证。没有县—时间定位的文本不能自动转成月度变量。")
    with tabs[3]:
        st.subheader("当前实现了哪些能力")
        st.dataframe(pd.DataFrame([
            {"能力": "来源追溯", "当前状态": "显示文件、页码/行号/字段及摘录哈希"},
            {"能力": "跨材料核查", "当前状态": "并排展示原始材料、确定性计算及其限制"},
            {"能力": "可见的证据缺口", "当前状态": "未知、待复核和检索失败单独标注"},
            {"能力": "LLM 生成", "当前状态": "本页未启用；现有审查解释不是模型生成"},
            {"能力": "准确率与节省时间", "当前状态": "尚未完成真实人工评价，没有提升结论"},
        ]), hide_index=True)
        evaluation = bundle.get("evaluation", {})
        st.write("当前人工评价准备：", evaluation)
        st.write("比较规则/字典、No-RAG LLM、RAG LLM。先由两名研究者独立标注，再保留分歧与裁决；正式测试按县或文档分组，并排除答案泄漏。")
        st.caption("开发试点数量不等于已标注样本数。软件测试、检索成功与研究性能分别报告。")
    with tabs[4]:
        exported = {"research_findings": bundle["findings"], "hypotheses": bundle["hypotheses"],
            "evidence_excerpts": bundle["documents"],
            "sources": bundle["sources"], "human_evaluation": bundle.get("evaluation", {})}
        if retrieval_result:
            exported["retrieval"] = retrieval_result
        st.download_button("下载研究发现与当前证据记录", report_bundle(exported, bundle["classification"],
            {"review_date": bundle.get("review_date"), "bundle_sha256": digest,
             "audit_run": bundle.get("audit_run", {})}),
            "orcca_research_findings.zip", "application/zip", key="findings_download")
        st.caption("本地报告可含研究标识和摘录，请按研究材料管理；原始文件没有被修改。")
