"""Schema-only synthetic workbook adapter tests; no study records or labels."""
import pandas as pd
import pytest

from orcca_audit.adapters import tracker_workbooks_to_tidy
from orcca_audit.engine import AuditError


def fixture():
    row = {"hcs_orc2record_id": "SYNTHETIC_ID", "redcap_event_name": "january_2024_arm_1",
        "hcs_orc2version": "SYNTHETIC_VERSION", "_source_row": 2,
        "hcs_orc2_showorc09": 1, "hcs_orc2_showorc10": 1}
    definitions = []
    for base, suffix in [("hcs_orc209", "l"), ("hcs_orc210", "m")]:
        for slot in range(1, 31):
            for field in [f"{base}{suffix}_{slot}", f"{base}v_{slot}"]:
                row[field] = 1 if slot == 1 and suffix in field else 0 if slot == 1 else None
                definitions.append({"Version": "SYNTHETIC_VERSION", "Variable / Field Name": field,
                    "_source_row": len(definitions) + 2})
    sheets = {"orcca_tracker_v2": pd.DataFrame([row]),
        "Sheet1": pd.DataFrame({"id": ["SYNTHETIC_ID"], "county": ["DEMO_A"]})}
    dictionary = pd.DataFrame(definitions)
    dictionary.attrs.update(source_file="synthetic_dictionary.xlsx", sheet="ORCCAT_DD")
    return sheets, dictionary


def test_adapter_retains_exact_versioned_fields_and_dictionary_locations():
    sheets, dictionary = fixture()
    tidy, refs = tracker_workbooks_to_tidy(sheets, dictionary, "synthetic_tracker.xlsx")
    assert len(tidy) == 2
    assert set(tidy.county) == {"DEMO_A"}
    assert tidy.loc[tidy.menu == "MOUD", "started_field"].iloc[0] == "hcs_orc210m_1"
    assert refs.dictionary_file.eq("synthetic_dictionary.xlsx").all()
    with pytest.raises(AuditError, match="definition is absent"):
        tracker_workbooks_to_tidy(sheets, dictionary.iloc[1:])
    sheets["Sheet1"] = pd.concat([sheets["Sheet1"], sheets["Sheet1"]])
    with pytest.raises(AuditError, match="unique"):
        tracker_workbooks_to_tidy(sheets, dictionary)
