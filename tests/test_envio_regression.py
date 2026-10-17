"""Original functions, synthetic fixtures, mocked external model responses.

No study transcripts/labels, API calls, or quality claims are used by these tests.
"""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def legacy():
    # Load original definitions without running the interactive top-level UI.
    # Function bodies and constants are unchanged, as checked by the hash test.
    tree = ast.parse((ROOT / "app.py").read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Try)):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
            nodes.append(node)
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "app.py"), "exec"), scope)
    return scope


def test_original_upload_coding_and_rag_refinement(legacy):
    file = io.BytesIO(b"SYNTHETIC transcript: community support."); file.name = "synthetic_interview.txt"
    assert "SYNTHETIC" in legacy["load_transcript_text"](file)
    reply = {"segments": [{"text": "SYNTHETIC community support.", "codes": ["social_support"], "rationale": "Synthetic test only."}]}
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])
    result = legacy["run_initial_llm_coding"]("SYNTHETIC community support.", legacy["CODEBOOK"], client)
    assert result[0]["codes"] == ["social_support"]
    refined = legacy["run_rag_refinement_coding"](result, "SYNTHETIC community support.", legacy["CODEBOOK"], client, "Synthetic source evidence.")
    assert refined[0]["codes"] == ["social_support"]
    assert client.chat.completions.create.call_count == 2


def test_original_evaluation_and_paired_summary(legacy):
    frame = pd.DataFrame({"segment_index": [1, 2], "codes": ["social_support", "stigma"]})
    comparison = legacy["compare_llm_human"](frame, frame, ["social_support", "stigma"])
    assert set(comparison.code) == {"social_support", "stigma", "MACRO_AVERAGE"}
    paired = legacy["build_rag_comparison_table"](comparison, comparison)
    assert len(paired) == 2  # Synthetic same-input software test, not research performance.


def test_original_stakeholder_synthesis_interface(legacy):
    fn = Mock(return_value="Synthetic engineer-oriented summary.")
    legacy["generate_role_adaptive_thematic_summary"] = fn
    text = legacy["apply_role_adaptive_summary"]("Synthetic base summary.", {"role": "engineer"}, "Explain")
    assert text == "Synthetic engineer-oriented summary."
    assert fn.call_args.kwargs["profile"]["role"] == "engineer"


def test_original_pdf_csv_zip_export(legacy, tmp_path):
    frame = pd.DataFrame({"participant_id": ["SYNTHETIC"], "source_type": ["interview"],
        "codes": ["social_support"], "text": ["Synthetic test only."]})
    path = tmp_path / "synthetic.pdf"
    legacy["generate_pdf_report"](frame, pd.Series({"social_support": 1}), output_file=str(path))
    assert path.read_bytes().startswith(b"%PDF")
    csv = tmp_path / "synthetic.csv"; frame.to_csv(csv, index=False)
    package = legacy["create_zip_from_files"]([str(path), str(csv)])
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == {"synthetic.pdf", "synthetic.csv"}


def test_actual_expression_layer_with_mocked_model(monkeypatch):
    import expression_layer
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Synthetic researcher summary."))])
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unit-test-placeholder")
    monkeypatch.setattr(expression_layer, "OpenAI", Mock(return_value=client))
    result = expression_layer.generate_role_adaptive_thematic_summary(
        "Synthetic source themes.", profile={"role": "researcher"})
    assert result == "Synthetic researcher summary."
    assert "Synthetic source themes." in client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
