"""Conservative reconstruction from explicit, version-defined no-change reports.

This is a rule-derived sensitivity dataset, never a human-approved recovery.
No model output or outcome value is used to decide a reconstruction.
"""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter

from .adapters import event_month
from .engine import AuditError
from .exports import safe_csv
from .h3_sources import code, CHOICES, BRANCH
from .h3_attendance_evidence import load_source
from .io import read_workbook, read_table
from .retrieval import retrieve

MENU_FIELDS = {'OEND': 'hcs_orc2b1', 'MOUD': 'hcs_orc2b2', 'Safer Prescribing': 'hcs_orc2b3'}
FIELDS = ['active_snapshot', 'venue_types']
METHOD = 'explicit_no_change_v1; adjacent existing Tracker month; prior known state; zero listed slots only'


def update_definitions(dictionary, versions, source_name):
    definitions, docs, logs = {}, [], []
    for version in sorted(versions):
        for field in ['hcs_orc2b', *MENU_FIELDS.values()]:
            selected = dictionary[dictionary.Version.astype(str).eq(version)
                                  & dictionary['Variable / Field Name'].eq(field)]
            if len(selected) != 1:
                raise AuditError('No unique version-specific no-change definition.')
            row = selected.iloc[0].to_dict()
            choices = str(row[CHOICES]).lower()
            if '0, no, there are no changes to report since last month' not in choices:
                raise AuditError('The no-change code meaning is not established by this dictionary.')
            if "changes to last month's orccat entries" not in str(row['Field Label']).lower():
                raise AuditError('The update question meaning requires review.')
            text = '\n'.join(str(row[k]) for k in ['Version', 'Variable / Field Name', 'Field Label', CHOICES, BRANCH])
            doc = dict(chunk_id=f'update-definition:{version}:{field}', source_file=source_name,
                       source_sheet='ORCCAT_DD', line_start=int(row['_source_row']),
                       field=field, version=version, text=text,
                       sha256=sha256(text.encode()).hexdigest(), role='original_source',
                       classification='private_local')
            definitions[version, field] = doc
            docs.append(doc)
    for doc in docs:
        result = retrieve(doc['field'] + ' no changes to last month ORCCAT entries',
                          [d for d in docs if d['version'] == doc['version']], backend='bm25', top_k=4)
        if doc['chunk_id'] not in {d['chunk_id'] for d in result['citations']}:
            raise AuditError('Required update definition was not retrieved.')
        logs.append(dict(version=doc['version'], target_field=doc['field'], **result))
    return definitions, docs, logs


def reconstruct_no_change(source_archive, tracker_file, dictionary_file):
    data, metadata = load_source(source_archive)
    for role, file in [('tracker', tracker_file), ('dictionary', dictionary_file)]:
        expected = [r for r in metadata['inputs'] if r['role'] == role]
        if len(expected) != 1 or expected[0]['sha256'] != sha256(file[1]).hexdigest():
            raise AuditError('The rule sources do not match the frozen reconstruction.')
    sheets = read_workbook(tracker_file[1], positional_sheets=('Sheet1',))
    raw, lookup = sheets['orcca_tracker_v2'], sheets['Sheet1']
    columns = {field:get_column_letter(i) for i,field in enumerate(raw.columns,1)}
    mapping = dict(zip(lookup.iloc[:, 0], lookup.iloc[:, 1]))
    rows = raw.to_dict('records')
    for row in rows:
        row['county'] = mapping[row['hcs_orc2record_id']]
        row['month'] = event_month(row['redcap_event_name'])
    dictionary = read_table(dictionary_file[1], dictionary_file[0], sheet='ORCCAT_DD')
    definitions, documents, retrievals = update_definitions(
        dictionary, set(raw.hcs_orc2version.astype(str)), dictionary_file[0])
    for field in FIELDS:
        data[field + '_original'] = data[field]
        data[field + '_basis'] = np.where(data[field].notna(), 'original_report', 'unknown')
    positions = {(r.county, r.mediator_month, r.menu): i for i, r in data.iterrows()}
    prior, ledger, audits = {}, [], []
    for row in sorted(rows, key=lambda r: (r['county'], r['month'])):
        county, month, version = row['county'], row['month'], str(row['hcs_orc2version'])
        global_flag = code(row.get('hcs_orc2b'))
        plan = code(row.get('hcs_orc2a'))
        for menu, menu_field in MENU_FIELDS.items():
            key = county, menu
            index = positions.get((county, month, menu))
            if index is None:
                prior.pop(key, None)
                continue
            current = data.loc[index]
            flag = code(row.get(menu_field))
            old = prior.get(key)
            adjacent = bool(old and str(pd.Period(old['month'], freq='M') + 1) == month)
            contradictory_flag = global_flag == '0' and flag == '1'
            explicit = (month != '2020-11' and plan == '1' and not contradictory_flag
                        and (global_flag == '0' or global_flag == '1' and flag == '0'))
            definition_field = 'hcs_orc2b' if global_flag == '0' else menu_field
            definition = definitions[version, definition_field]
            conflicts = []
            if explicit and adjacent:
                conflicts = [field for field in FIELDS if pd.notna(current[field])
                             and old['values'].get(field) is not None
                             and not np.isclose(current[field], old['values'][field])]
            can_carry = (explicit and adjacent and not conflicts and pd.notna(current.listed_slots)
                         and current.listed_slots == 0)
            values, anchors = {}, {}
            for field in FIELDS:
                value = data.at[index, field]
                anchor = dict(month=month, tracker_row=int(row['_source_row']))
                if pd.isna(value) and can_carry and old['values'].get(field) is not None:
                    value = old['values'][field]
                    anchor = old['anchors'][field]
                    data.at[index, field] = value
                    data.at[index, field + '_basis'] = 'dictionary_supported_no_change_not_human_adjudicated'
                    ledger.append(dict(county=county, tracker_month=month, outcome_month=current.outcome_month,
                        menu=menu, field=field, original_value=None, reconstructed_value=value,
                        tracker_source=tracker_file[0], tracker_sheet='orcca_tracker_v2', tracker_row=int(row['_source_row']),
                        update_response_cell=f"{columns.get(definition_field, '')}{int(row['_source_row'])}",
                        dictionary_source=dictionary_file[0], dictionary_sheet='ORCCAT_DD',
                        dictionary_row=definition['line_start'], dictionary_chunk_id=definition['chunk_id'],
                        original_snapshot_month=anchor['month'], original_snapshot_row=anchor['tracker_row'],
                        global_change_code=global_flag, menu_change_code=flag,
                        immediate_predecessor_month=old['month'], anchor_month=anchor['month'],
                        anchor_tracker_row=anchor['tracker_row'], dictionary_sha256=definition['sha256'],
                        method=METHOD, human_confirmed=False))
                values[field] = None if pd.isna(value) else float(value)
                anchors[field] = anchor
            audits.append(dict(county=county, month=month, menu=menu, tracker_row=int(row['_source_row']),
                version=version, action_plan_code=plan, global_change_code=global_flag, menu_change_code=flag,
                explicit_no_change=explicit, adjacent_predecessor=adjacent,
                listed_slots=current.listed_slots, conflicts=' | '.join(conflicts),
                contradictory_flags=contradictory_flag,
                rule_derived_fields=sum(data.at[index, f + '_basis'] != 'unknown'
                    and data.at[index, f + '_basis'] != 'original_report' for f in FIELDS)))
            if conflicts or contradictory_flag:
                prior.pop(key, None)
            else:
                prior[key] = dict(month=month, values=values, anchors=anchors)
    for field in FIELDS:
        observed = data[field + '_original'].notna()
        if not data.loc[observed, field].eq(data.loc[observed, field + '_original']).all():
            raise AuditError('An original recorded snapshot was changed.')
    data['tracker_snapshot_available'] = data.active_snapshot.notna()
    data['all_three_available'] = data.attendance.notna() & data.active_snapshot.notna() & data.outcome_available
    data['venue_path_available'] = data.attendance.notna() & data.venue_types.notna() & data.outcome_available
    coverage = data.groupby('menu', as_index=False)[['attendance_available', 'tracker_snapshot_available',
        'outcome_available', 'all_three_available', 'venue_path_available']].sum()
    metadata.update(rule_reconstruction=METHOD, rule_reconstructed_mediator_values=len(ledger),
                    rule_human_adjudicated=False, approved_attendance_county_months=0, imputed_values=0)
    output = BytesIO()
    with ZipFile(BytesIO(source_archive)) as source, ZipFile(output, 'w', ZIP_DEFLATED) as target:
        for name in source.namelist():
            if name not in {'county_month.csv','run_metadata.json','coverage.csv','START_HERE.html'}:
                target.writestr(name, source.read(name))
        target.writestr('county_month.csv', safe_csv(data))
        target.writestr('run_metadata.json', json.dumps(metadata, indent=2))
        target.writestr('coverage.csv', safe_csv(coverage))
        target.writestr('no_change_reconstruction_ledger.csv', safe_csv(ledger))
        target.writestr('no_change_rule_audit.csv', safe_csv(audits))
        target.writestr('definition_source_memory.json', json.dumps(documents, indent=2))
        target.writestr('definition_retrieval_log.json', json.dumps(retrievals, indent=2))
        target.writestr('START_HERE.html', '<h1>Dictionary-supported no-change reconstruction</h1><p>Rule-derived sensitivity inputs. No new attendance counts and no human labels. Inspect the ledger and original values before interpreting model output.</p>')
    return metadata, coverage, pd.DataFrame(ledger), pd.DataFrame(audits), output.getvalue()
