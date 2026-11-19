"""Retrospective donor estimates of currently delivering fixed sites."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from statistics import median_low
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from openpyxl.utils import get_column_letter

from .adapters import event_month
from .exports import safe_csv
from .h3_sources import BRANCH, code
from .io import read_table, read_workbook
from .pilot import branching


def count(value):
    token = code(value)
    return int(token) if token.isascii() and token.isdigit() else None


def extract(tracker, dictionary, *, include_reach_id=False):
    sheets = read_workbook(tracker[1], positional_sheets=('Sheet1',))
    raw, lookup = sheets['orcca_tracker_v2'], sheets['Sheet1']
    if lookup.iloc[:, 0].duplicated().any() or lookup.iloc[:, :2].isna().any().any():
        raise ValueError('Review missing or duplicate county lookup entries.')
    counties = dict(zip(lookup.iloc[:, 0], lookup.iloc[:, 1]))
    dd = read_table(dictionary[1], dictionary[0], sheet='ORCCAT_DD')
    if dd.duplicated(['Version', 'Variable / Field Name']).any():
        raise ValueError('Review duplicate version-specific dictionary definitions.')
    definitions = {(str(r['Version']), r['Variable / Field Name']): r for r in dd.to_dict('records')}
    columns = {name: get_column_letter(i) for i, name in enumerate(raw.columns, 1)}
    records = []
    for row in raw.to_dict('records'):
        version = str(row['hcs_orc2version'])
        for prefix, menu, sector, venue in [('hcs_orc209', 'OEND', 'c', 'd'), ('hcs_orc210', 'MOUD', 'b', 'e')]:
            for slot in range(1, 31):
                def field(suffix):
                    return f'{prefix}{suffix}_{slot}'
                target, plan = field('qb'), field('q')
                definition = definitions.get((version, target), {})
                if target not in columns or not code(row.get(field(''))):
                    continue
                if 'fixed physical locations are currently delivering' not in str(definition.get('Field Label', '')).lower():
                    continue
                if branching(definition.get(BRANCH), row) != 'applicable':
                    continue
                plan_definition = definitions.get((version, plan), {})
                plan_known = ('fixed physical locations' in str(plan_definition.get('Field Label', '')).lower()
                              and 'intended' in str(plan_definition.get('Field Label', '')).lower()
                              and branching(plan_definition.get(BRANCH), row) == 'applicable')
                actual = count(row.get(target))
                if code(row.get(target)) and actual is None:
                    raise ValueError('A recorded delivery count is invalid. Review the source cell.')
                records.append(dict(county=str(counties[row['hcs_orc2record_id']]).strip(),
                    month=event_month(row['redcap_event_name']), version=version, menu=menu, slot=slot,
                    strategy=code(row[field('')]), sector=code(row.get(field(sector))),
                    venue=code(row.get(field(venue))), planned=count(row.get(plan)) if plan_known else None, recorded=actual,
                    plan_cell=columns.get(plan,'')+str(row['_source_row']) if plan in columns else '',
                    target_cell=columns[target]+str(row['_source_row']), dictionary_row=definition['_source_row']))
                if include_reach_id:
                    records[-1]['reach_id'] = code(row.get(field('_reach_id')))
    if len({(r['county'], r['month'], r['menu'], r['slot']) for r in records}) != len(records):
        raise ValueError('Review duplicate strategy-slot records within a county-month.')
    return records


def estimate(rows):
    """Match the plan and strategy exactly, then prefer matching venue and sector."""
    results, history = [], []
    for month in sorted({r['month'] for r in rows}):
        current = [r for r in rows if r['month'] == month]
        for r in current:
            donors = [d for d in history if r['planned'] is not None and
                      all(d[k] == r[k] for k in ('version', 'menu', 'strategy', 'planned'))]
            # Use at most one record per county and slot to limit repeated-report weighting.
            latest = {(d['county'], d['slot']): d for d in donors}
            def score(d):
                return sum(bool(r[k]) and r[k] == d[k] for k in ('venue', 'sector'))
            best = max(map(score, latest.values()), default=0)
            selected = [d for d in latest.values() if best and score(d) == best]
            prediction = median_low(sorted(d['recorded'] for d in selected)) if len(selected) >= 2 else None
            results.append({**r, 'estimate': prediction,
                'full_delivery_scenario': r['planned'] if r['recorded'] is None else None, 'donor_groups': len(selected),
                'donor_cells': ' | '.join(d['target_cell'] for d in selected),
                'donor_months': ' | '.join(d['month'] for d in selected),
                'status': 'recorded' if r['recorded'] is not None else
                ('statistical estimate' if prediction is not None else 'unknown')})
        history.extend(r for r in current if r['recorded'] is not None)
    return results


def report(tracker, dictionary):
    rows = estimate(extract(tracker, dictionary))
    paired = [r for r in rows if r['recorded'] is not None and r['estimate'] is not None]
    metrics = []
    for key, method in [('estimate','Matched historical records'), ('planned','Planned sites baseline')]:
        errors = [abs(r[key]-r['recorded']) for r in paired]
        metrics.append(dict(method=method, paired_cells=len(errors),
            exact=sum(e == 0 for e in errors), mean_absolute_error=sum(errors)/len(errors) if errors else None))
    summary = dict(applicable_cells=len(rows), recorded=sum(r['recorded'] is not None for r in rows),
        missing=sum(r['recorded'] is None for r in rows), estimated=sum(r['status']=='statistical estimate' for r in rows),
        still_unknown=sum(r['status']=='unknown' for r in rows), metrics=metrics,
        tracker_sha256=sha256(tracker[1]).hexdigest(), dictionary_sha256=sha256(dictionary[1]).hexdigest())
    summary['estimated_by_county'] = dict(Counter(r['county'] for r in rows if r['status']=='statistical estimate'))
    plan_validation = [r for r in rows if r['recorded'] is not None and r['planned'] is not None]
    errors = [abs(r['recorded'] - r['planned']) for r in plan_validation]
    summary['plan_validation'] = dict(recorded_pairs=len(errors), exact=errors.count(0),
        mean_absolute_error=sum(errors)/len(errors) if errors else None)
    summary['full_delivery_scenarios'] = sum(r['full_delivery_scenario'] is not None for r in rows)
    data = pd.DataFrame(rows)
    out = BytesIO()
    with ZipFile(out,'w',ZIP_DEFLATED) as z:
        z.writestr('summary.json',json.dumps(summary,indent=2))
        z.writestr('delivery_site_review.csv',safe_csv(data))
        z.writestr('README.txt','Outcome: number of fixed sites currently delivering services. Original values are preserved. full_delivery_scenario assumes every planned site delivered and is not a verified count. '
            'Estimates are unverified. Validation compares both methods on the same recorded cells. '
            'Each prediction uses observed donors from earlier reporting months only. Reports may have been entered later. '
            'Donors match menu, version, strategy code and planned count. Among these, use the best venue/sector matches '
            'and the lower median across at least two county-slot groups. Donor cells refer to the orcca_tracker_v2 worksheet. '
            'No generated values feed back into memory or the H3 models. This analysis does not test mediation.')
    return summary, data, out.getvalue()
