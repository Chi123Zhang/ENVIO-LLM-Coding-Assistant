"""Local-only REDCap workbook adapter; contains schema identifiers, no records.

Questionnaire text, value labels and county mappings come only from the user's
workbooks. Nothing is discovered from the repository or written into an index.
"""
import pandas as pd

from .engine import AuditError, empty, require


def event_month(event):
    import calendar
    parts = str(event).split("_")
    months = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
    if len(parts) != 4 or parts[0] not in months or parts[2:] != ["arm", "1"]:
        raise AuditError("Unrecognized REDCap event; an explicit event-month map is required.")
    return f"{int(parts[1]):04d}-{months[parts[0]]:02d}"


def tracker_workbooks_to_tidy(sheets, dictionary, filename="local_tracker.xlsx"):
    if "orcca_tracker_v2" not in sheets or "Sheet1" not in sheets:
        raise AuditError("Tracker needs its main sheet and explicit county mapping sheet.")
    tracker, mapping = sheets["orcca_tracker_v2"], sheets["Sheet1"]
    require(tracker, ["hcs_orc2record_id", "redcap_event_name", "hcs_orc2version"], "Tracker workbook")
    require(dictionary, ["Version", "Variable / Field Name"], "Dictionary workbook")
    if dictionary.duplicated(["Version", "Variable / Field Name"]).any():
        raise AuditError("Duplicate version/field definitions in the supplied dictionary.")
    if mapping.shape[1] < 2 or mapping.iloc[:, 0].map(empty).any() or mapping.iloc[:, 0].duplicated().any():
        raise AuditError("County lookup needs unique nonempty IDs in its first column.")
    ids = {r.iloc[0]: r.iloc[1] for _, r in mapping.iterrows()}
    definitions = {(str(r["Version"]), str(r["Variable / Field Name"])): r for _, r in dictionary.iterrows()}
    tidy, references = [], []
    for _, row in tracker.iterrows():
        county = ids.get(row["hcs_orc2record_id"])
        if empty(county):
            raise AuditError("County mapping is absent for a Tracker record.")
        month, version = event_month(row["redcap_event_name"]), str(row["hcs_orc2version"])
        for menu, base, start, date_suffix, show in [
            ("OEND", "hcs_orc209", "l", "o", "hcs_orc2_showorc09"),
            ("MOUD", "hcs_orc210", "m", "p", "hcs_orc2_showorc10")]:
            if show not in tracker:
                raise AuditError("Missing menu show/update field.")
            found = False
            for slot in range(1, 31):
                sf, df, tf = f"{base}{start}_{slot}", f"{base}v_{slot}", f"{base}_{slot}"
                for field in [sf, df]:
                    if field not in tracker or (version, field) not in definitions:
                        raise AuditError("A required strategy field/version definition is absent.")
                if all(empty(row.get(f)) for f in [tf, sf, df]):
                    continue
                found = True
                sm, sy = row.get(f"{base}{date_suffix}_{slot}_mo"), row.get(f"{base}{date_suffix}_{slot}_yr")
                started_at = f"{int(sy):04d}-{int(sm):02d}" if not empty(sm) and not empty(sy) else None
                tidy.append({"county": county, "month": month, "menu": menu, "slot": slot,
                    "show": row[show], "started": row[sf], "discontinued": row[df], "start_month": started_at,
                    "_source_row": row["_source_row"], "source_file": filename, "source_sheet": "orcca_tracker_v2",
                    "version": version, "started_field": sf, "discontinued_field": df,
                    "dictionary_file": dictionary.attrs.get("source_file", "local_dictionary.xlsx"),
                    "dictionary_sheet": dictionary.attrs.get("sheet", "ORCCAT_DD"),
                    "dictionary_started_row": definitions[version, sf]["_source_row"],
                    "dictionary_discontinued_row": definitions[version, df]["_source_row"]})
                references.append({"menu": menu, "slot": slot, "version": version, "started_field": sf,
                    "dictionary_file": dictionary.attrs.get("source_file", "local_dictionary.xlsx"),
                    "dictionary_sheet": dictionary.attrs.get("sheet", "ORCCAT_DD"),
                    "discontinued_field": df, "dictionary_start_row": definitions[version, sf]["_source_row"],
                    "dictionary_discontinued_row": definitions[version, df]["_source_row"],
                    "status": "mapping_from_supplied_dictionary_needs_semantic_review"})
            if not found:
                tidy.append({"county": county, "month": month, "menu": menu, "slot": 0, "show": row[show],
                    "started": None, "discontinued": None, "start_month": None,
                    "_source_row": row["_source_row"], "source_file": filename, "source_sheet": "orcca_tracker_v2"})
    return pd.DataFrame(tidy), pd.DataFrame(references).drop_duplicates()
