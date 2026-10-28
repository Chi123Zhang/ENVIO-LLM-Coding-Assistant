"""Reviewed English translations for existing review metadata and receipts.

No model calls. Unknown Chinese text is rejected rather than omitted or guessed.
Original input files are never edited.
"""
import copy
import hashlib
import re

from .engine import AuditError

CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
TRANSLATIONS = {
    "宽松候选值匹配不是原始算法核实。64是滞后核查条目数，不是64个独立错误。":
        "A permissive candidate-value match does not verify the original algorithm. The 64 items are lag checks, not 64 independent errors.",
    "仅限本次提供的文件；未提供不等于研究团队没有。Table 4是来源清单，不是所列数据本身。":
        "Scope is limited to the supplied files; missing from this set does not mean unavailable to the research team. Table 4 lists sources, not the underlying data.",
    "results.csv 表头第1行、OEND主结果第2行、MOUD主结果第16行":
        "results.csv: header at row 1; main OEND result at row 2; main MOUD result at row 16",
    "panel 数据结构及来源核查；由本地代码计算，非原始材料原文。":
        "Panel structure and source checks; computed by local code, not an original source quotation.",
    "暴露候选规则回放与状态不确定性；由本地代码计算，非原始材料原文。":
        "Candidate exposure-rule replay and status uncertainty; computed by local code, not an original source quotation.",
    "全年兼容性与部分年份等式核查；由本地代码计算，非原始材料原文。":
        "Full-year compatibility and partial-year equality checks; computed by local code, not an original source quotation.",
    "来源明确的规格差异，不是重新拟合；由本地代码计算，非原始材料原文。":
        "Source-linked specification differences, without refitting; computed by local code, not an original source quotation.",
    "当前资料范围与材料缺口；由本地代码计算，非原始材料原文。":
        "Available materials and evidence gaps; computed by local code, not an original source quotation.",
}


def english_text(value):
    for source, target in TRANSLATIONS.items():
        value = value.replace(source, target)
    value = re.sub(r"CSV 物理行 (\d+–\d+)", r"CSV physical lines \1", value)
    value = re.sub(r"(\S+) 工作表 Excel 第(\d+)行；字段 (\S+)；版本 (\S+)",
        r"Worksheet \1; Excel row \2; field \3; version \4", value)
    value = re.sub(r"(\S+) 工作表 Excel 第(\d+)行；保留原始空值，未自动判为否",
        r"Worksheet \1; Excel row \2; original blanks retained, not automatically coded as No", value)
    value = re.sub(r"年度原始 TSV 物理行1和(\d+)；(.*)", r"Annual source TSV physical lines 1 and \1; \2", value)
    value = re.sub(r"月度原始 TSV 物理行 (\d+–\d+)；含窗口外月份",
        r"Monthly source TSV physical lines \1; includes months outside the analysis window", value)
    value = re.sub(r"本地结果幻灯片 PDF 物理第(\d+)页（PPTX的PDF版本）",
        r"Local results slides, physical PDF page \1 (PDF version of the PPTX)", value)
    value = re.sub(r"Drive 只读预览第(\d+)页；提取行号仅指该页摘录，不是原 DOCX 字节校验",
        r"Read-only Drive preview, page \1; line numbers refer to the extracted page, not verification of original DOCX bytes", value)
    if CJK.search(value):
        raise AuditError("English export blocked: an additional reviewed translation is needed. No text was silently removed.")
    return value


def english_results(results):
    result = copy.deepcopy(results)
    # Keep the hash of pre-translation text and label every translated record.
    for doc in result.get("evidence_excerpts", []) + result.get("retrieval", {}).get("citations", []):
        original = doc.get("text", "")
        translated = english_text(original)
        if translated != original:
            doc["original_text_sha256"] = hashlib.sha256(original.encode()).hexdigest()
            doc["text"] = translated
            doc["sha256"] = hashlib.sha256(translated.encode()).hexdigest()
            doc["text_version"] = "English translation; original text retained in the local source review"
            doc["translation_method"] = "Reviewed fixed translation; no external model"
    def convert(value):
        if isinstance(value, str):
            return english_text(value)
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {english_text(str(k)): convert(v) for k, v in value.items()}
        return value
    return convert(result)
