"""Language conversion integrity; synthetic text, no human reference labels."""
import hashlib

import pytest

from orcca_audit.english_export import CJK, english_results, english_text
from orcca_audit.engine import AuditError


def test_metadata_translation_preserves_locations():
    assert english_text("CSV 物理行 1–16") == "CSV physical lines 1–16"
    assert english_text("SHEET 工作表 Excel 第12行；字段 field_id；版本 1.0") == "Worksheet SHEET; Excel row 12; field field_id; version 1.0"
    assert "original blanks retained" in english_text("TEST 工作表 Excel 第108行；保留原始空值，未自动判为否")


def test_translated_receipt_has_two_hashes_without_mutating_input():
    text = "宽松候选值匹配不是原始算法核实。64是滞后核查条目数，不是64个独立错误。"
    doc = {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
    result = english_results({"evidence_excerpts": [doc]})
    translated = result["evidence_excerpts"][0]
    assert not CJK.search(translated["text"])
    assert translated["original_text_sha256"] == doc["sha256"]
    assert translated["sha256"] == hashlib.sha256(translated["text"].encode()).hexdigest()
    assert "English translation" in translated["text_version"]
    assert doc["text"] == text
    assert english_results(result) == result


def test_unknown_text_requires_translation_instead_of_silent_deletion():
    with pytest.raises(AuditError, match="additional reviewed translation"):
        english_text("未配置的合成测试文字")
