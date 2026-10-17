"""Entirely synthetic public fixtures. Not research cases, labels or findings."""
from pathlib import Path
import hashlib

import pandas as pd

from .engine import shift


def public_documents():
    path = Path(__file__).with_name("PUBLIC_DEMO_METHODS.md")
    text = path.read_text(encoding="utf-8")
    return [{"chunk_id": "public-synthetic-protocol:1", "source_file": "orcca_audit/PUBLIC_DEMO_METHODS.md",
        "page": None, "line_start": 1, "line_end": len(text.splitlines()), "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(), "classification": "synthetic_public"}]


def demo_data():
    monthly, annual, tracker, panel = [], [], [], []
    for c, county in enumerate(["DEMO_A", "DEMO_B", "DEMO_C"]):
        counts = []
        for j in range(1, 13):
            y = 10 + c + j
            counts.append(y)
            monthly.append({"county": county, "month": f"2024-{j:02d}", "Deaths": y,
                "source_file": "synthetic_monthly.csv", "_source_row": len(monthly) + 2})
        annual.append({"county": county, "year": 2024, "Deaths": sum(counts), "population": 10000 + 1000 * c,
            "source_file": "synthetic_annual.csv", "_source_row": len(annual) + 2})
        for j in range(1, 9):
            for menu in ["OEND", "MOUD"]:
                tracker.append({"county": county, "month": f"2024-{j:02d}", "menu": menu,
                    "slot": 1, "show": 1, "started": 1,
                    "discontinued": None if county == "DEMO_A" and j == 1 and menu == "MOUD" else 0,
                    "start_month": "2024-06" if county == "DEMO_B" and j == 4 and menu == "OEND" else "2024-01",
                    "source_file": "synthetic_tracker.csv", "_source_row": len(tracker) + 2})
        for j in range(4, 10):
            y = counts[j - 1]
            panel.append({"county": county, "date": f"2024-{j:02d}-01", "Deaths": y,
                "y_lo": y, "y_hi": y, "suppressed": 0, "population": 10000 + 1000 * c,
                "OEND_L1": 1, "OEND_L13": 1, "MOUD_L1": 1, "MOUD_L13": 1,
                "source_file": "synthetic_panel.csv", "_source_row": len(panel) + 2})
    # Two protocol versions in an actual committed synthetic document, not a paper.
    specs = [
        {"spec_id": "synthetic_monthly", "source": "orcca_audit/PUBLIC_DEMO_METHODS.md lines 6-7",
         "exposure_window": "previous 3 months", "outcome_window": "one month", "exposure_scale": "raw count"},
        {"spec_id": "synthetic_three_month", "source": "orcca_audit/PUBLIC_DEMO_METHODS.md lines 8-9",
         "exposure_window": "previous 3 months", "outcome_window": "subsequent 3-month sum", "exposure_scale": "standardized log; details unknown"},
    ]
    return {"panel": pd.DataFrame(panel), "monthly": pd.DataFrame(monthly), "annual": pd.DataFrame(annual),
            "tracker": pd.DataFrame(tracker), "specifications": specs}
