"""Protect every pre-existing file, not just the six core modules."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ADDITION = ('st.sidebar.page_link("app.py", label="ENVIO Interview Coding")\n'
    'st.sidebar.page_link("pages/01_ORCCA_Research_Audit.py", label="ORCCA Research Audit")\n'
    'st.sidebar.caption("ENVIO workspace: authorized interview materials only. Do not upload ORCCA research files to this public page.")\n')
BRIDGE_ADDITIONS = ['# ORCCA additive bridge: fingerprint current inputs without running a model.\nfrom orcca_audit.envio_bridge import input_signature, capture_run\n_envio_bridge_settings = {"user_id": user_id, "mode": mode, "query": query,\n    "use_rag_context": use_rag_context, "paired_comparison": run_rag_comparison,\n    "codebook": CODEBOOK, "model": "gpt-4o-mini", "merge_related_files": merge_related_files,\n    "human_alignment": use_human_segments_for_alignment,\n    "human_file_sha": input_signature([human_coding_file] if human_coding_file else [], "", {}),\n    "role": manual_role, "infer_profile": use_resume_profile, "manual_override": allow_manual_override}\n_envio_bridge_signature = input_signature(uploaded_files, url_input, _envio_bridge_settings)\n\n', '    # A failed or new run cannot leave an earlier coding run eligible for transfer.\n    st.session_state.pop("envio_orcca_capture", None)\n', '            # Capture existing private-session outputs only; no extra API call.\n            capture_run(st.session_state, summary_df, source_records,\n                _envio_bridge_signature, _envio_bridge_settings)\n', '\n# Additive ORCCA handoff, rendered independently of the Run button rerun.\nfrom orcca_audit.envio_bridge_ui import render as render_orcca_handoff\nrender_orcca_handoff(_envio_bridge_signature)\n']
REQ_ADDITION = '\n# ORCCA local-private workbook parsing; no study data or model configuration.\nopenpyxl>=3.1.5,<4\n'


def test_all_existing_files_preserved_except_documented_additions():
    baseline = json.loads((ROOT / "docs/baseline_main.json").read_text())
    assert baseline["main_commit"] == "f716bb4f93cc5854adb004d4b4e239be84cd1184"
    for name, expected in baseline["original_files_sha256"].items():
        p = ROOT / name
        assert p.is_file(), f"Original file missing: {name}"
        data = p.read_bytes()
        if name == "app.py":
            assert APP_ADDITION.encode() in data
            data = data.replace(APP_ADDITION.encode(), b"", 1)
            for block in BRIDGE_ADDITIONS:
                assert data.count(block.encode()) == 1, 'Unexpected bridge modification'
                data = data.replace(block.encode(), b'', 1)
        elif name == "requirements.txt":
            assert data.endswith(REQ_ADDITION.encode())
            data = data[:-len(REQ_ADDITION.encode())]
        assert hashlib.sha256(data).hexdigest() == expected, f"Unexpected change to {name}"
