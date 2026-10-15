"""Protect every pre-existing file, not just the six core modules."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ADDITION = ('st.sidebar.page_link("app.py", label="ENVIO Interview Coding")\n'
    'st.sidebar.page_link("pages/01_ORCCA_Research_Audit.py", label="ORCCA Research Audit")\n'
    'st.sidebar.caption("ENVIO workspace: authorized interview materials only. Do not upload ORCCA research files to this public page.")\n')
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
        elif name == "requirements.txt":
            assert data.endswith(REQ_ADDITION.encode())
            data = data[:-len(REQ_ADDITION.encode())]
        assert hashlib.sha256(data).hexdigest() == expected, f"Unexpected change to {name}"
