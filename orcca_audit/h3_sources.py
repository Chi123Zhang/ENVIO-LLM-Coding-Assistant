"""Rebuild H3 coverage from source files in memory."""

from hashlib import sha256
import html
from io import BytesIO
import json
import re
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

from .adapters import event_month
from .engine import AuditError, empty, require
from .exports import safe_csv
from .io import read_table, read_workbook
from .pilot import branching

MENUS = {
    'OEND': ('09', 'l', 'v', 'd', 'o', 'w'),
    'MOUD': ('10', 'm', 'v', 'e', 'p', 'w'),
    'Safer Prescribing': ('11', 'g', 'q', 'c', 'j', 'r'),
}
BRANCH = 'Branching Logic (Show field only if...)'
CHOICES = 'Choices, Calculations, OR Slider Labels'


def code(value):
    return '' if empty(value) else str(value).strip().removesuffix('.0')


def date_pair(row, field):
    y, m = (code(row.get(field + suffix)) for suffix in ('_yr', '_mo'))
    if not y and not m:
        return ''
    if not re.fullmatch(r'\d{4}', y) or not re.fullmatch(r'0?[1-9]|1[0-2]', m):
        return None
    return f'{y}-{int(m):02d}'


def tracker_measurements(tracker_file, dictionary_file):
    sheets = read_workbook(tracker_file[1], positional_sheets=('Sheet1',))
    if not {'orcca_tracker_v2', 'Sheet1'} <= sheets.keys():
        raise AuditError('Tracker requires the main worksheet and Sheet1 county lookup.')
    raw, lookup = sheets['orcca_tracker_v2'], sheets['Sheet1']
    require(raw, ['hcs_orc2record_id', 'redcap_event_name', 'hcs_orc2version'], 'Tracker')
    if lookup.shape[1] < 3 or lookup.iloc[:, 0].map(empty).any() or lookup.iloc[:, 0].duplicated().any():
        raise AuditError('Tracker county lookup needs unique IDs, with county names in column B.')
    counties = dict(zip(lookup.iloc[:, 0], lookup.iloc[:, 1]))
    dictionary = read_table(dictionary_file[1], dictionary_file[0], sheet='ORCCAT_DD')
    require(dictionary, ['Version', 'Variable / Field Name', 'Field Label', BRANCH, CHOICES], 'Dictionary')
    if dictionary.duplicated(['Version', 'Variable / Field Name']).any():
        raise AuditError('Duplicate dictionary version and field.')
    definitions = {(str(row['Version']), row['Variable / Field Name']): row
                   for row in dictionary.to_dict('records')}
    cells = {field: get_column_letter(i) for i, field in enumerate(raw.columns, 1)}
    snapshots, evidence, keys = [], [], set()
    for row in raw.to_dict('records'):
        county = counties.get(row['hcs_orc2record_id'])
        if empty(county):
            raise AuditError('Tracker record missing a county mapping.')
        county = str(county).strip()
        month, version = event_month(row['redcap_event_name']), str(row['hcs_orc2version'])
        if (county, month) in keys:
            raise AuditError('Duplicate county and month in Tracker.')
        keys.add((county, month))
        for menu, (n, start, stop, venue, start_date, stop_date) in MENUS.items():
            base = 'hcs_orc2' + n
            states, venues = [], []
            for slot in range(1, 31):
                sf, ef, vf = (f'{base}{suffix}_{slot}' for suffix in (start, stop, venue))
                started, ended = (code(row.get(field)) for field in (sf, ef))
                if not (code(row.get(f'{base}_{slot}')) or started or ended):
                    continue
                sd, ed, vd = (definitions.get((version, field), {}) for field in (sf, ef, vf))
                sb, eb = (branching(d.get(BRANCH), row) if d else 'definition_missing' for d in (sd, ed))
                begin, end = (date_pair(row, f'{base}{suffix}_{slot}') for suffix in (start_date, stop_date))
                state = 'unresolved'
                valid_dates = (
                    begin is not None and end is not None
                    and (started != '1' or bool(begin) and begin <= month)
                    and (ended != '1' or bool(end) and end <= month)
                    and (not begin or not end or begin <= end)
                    and (started != '0' or not begin)
                )
                binary = all({choice.strip().lower() for choice in str(d.get(CHOICES, '')).split('|')}
                             == {'0, no', '1, yes'} for d in (sd, ed))
                verified = (
                    binary and 'has implementation started' in str(sd.get('Field Label', '')).lower()
                    and 'has this strategy been discontinued' in str(ed.get('Field Label', '')).lower()
                )
                if verified and valid_dates:
                    if eb == 'applicable' and ended == '1':
                        state = 'discontinued'
                    elif sb == 'applicable' and started == '0' and ended != '1':
                        state = 'not_started'
                    elif sb == eb == 'applicable' and started == '1' and ended == '0':
                        state = 'active'
                decoded = dict(re.findall(r'(?:^|\|)\s*(\d+)\s*,\s*([^|]+)', str(vd.get(CHOICES, ''))))
                label = decoded.get(code(row.get(vf)), '').strip()
                states.append(state)
                if state == 'active':
                    venues.append(label)
                line = row['_source_row']
                evidence.append(dict(county=county, tracker_month=month, menu=menu, slot=slot, version=version,
                                     tracker_sheet='orcca_tracker_v2', tracker_row=line,
                                     start_cell=f'{cells[sf]}{line}' if sf in cells else '',
                                     stop_cell=f'{cells[ef]}{line}' if ef in cells else '',
                                     venue_cell=f'{cells[vf]}{line}' if vf in cells else '',
                                     start_dictionary_row=sd.get('_source_row'), stop_dictionary_row=ed.get('_source_row'),
                                     venue_dictionary_row=vd.get('_source_row'), start_code=started, stop_code=ended,
                                     start_rule=sb, stop_rule=eb, start_month=begin, stop_month=end,
                                     state=state, venue=label))
            unresolved = states.count('unresolved')
            complete = bool(states) and not unresolved
            snapshots.append(dict(county=county, mediator_month=month, menu=menu, tracker_row=row['_source_row'],
                                  listed_slots=len(states), unresolved_slots=unresolved,
                                  active_snapshot=states.count('active') if complete else None,
                                  venue_types=len(set(venues)) if complete and all(venues) else None))
    if not keys:
        raise AuditError('Tracker has no records.')
    return pd.DataFrame(snapshots), pd.DataFrame(evidence), sorted({key[0] for key in keys})


def mortality(file, key, pattern, columns):
    data = read_table(file[1], file[0])
    require(data, [key, 'County', 'County Code', *columns], file[0])
    data = data[data[key].astype(str).str.fullmatch(pattern)].copy()
    data['fips'] = data['County Code'].map(code)
    data['county'] = data.County.astype(str).str.replace(r' County, NY$', '', regex=True).str.strip()
    if not data.fips.str.fullmatch(r'\d{5}').all() or data.duplicated(['fips', key]).any():
        raise AuditError('CDC requires valid FIPS and unique county and period records.')
    return data


def attendance_counts(files, counties):
    rows = []
    for name, raw in files:
        table = read_table(raw, name)
        require(table, ['county', 'date', 'Name', 'Attendance'], name)
        table['county'] = table.county.astype(str).str.strip().str.replace(r'^Copy of ', '', regex=True, case=False).str.casefold()
        table['county'] = table.county.map({county.casefold(): county for county in counties})
        table['date'] = pd.to_datetime(table.date, format='mixed', errors='coerce').dt.normalize()
        table['person'] = table.Name.fillna('').str.split().str.join(' ').str.casefold()
        table['present'] = table.Attendance.fillna('').str.strip().str.casefold().eq('present')
        valid = table.county.notna() & table.date.notna() & ~table.person.isin(['', 'name', 'present', 'not present'])
        rows.append(table[valid])
    records = pd.concat(rows)
    meetings = []
    for (county, date), group in records.groupby(['county', 'date']):
        # A list with no explicit marks cannot establish zero attendance.
        if not group.present.any():
            continue
        sources = [
            f'{name}: {",".join(source._source_row.astype(str))}'
            for name, source in group.groupby('source_file')
        ]
        meetings.append((county, date.strftime('%Y-%m'), group.loc[group.present, 'person'].nunique(), ' | '.join(sources)))
    months = pd.DataFrame(meetings, columns=['county', 'attendance_month', 'present', 'attendance_source'])
    return months.groupby(['county', 'attendance_month'], as_index=False).agg(
        attendance=('present', 'mean'), attendance_source=('attendance_source', ' | '.join)
    )


def rebuild(files, attendance_files, start, end, *, synthetic):
    snapshots, evidence, counties = tracker_measurements(files['tracker'], files['dictionary'])
    monthly = mortality(files['monthly'], 'Month Code', r'\d{4}/(0[1-9]|1[0-2])', ['Deaths'])
    annual = mortality(files['annual'], 'Year Code', r'\d{4}', ['Population', 'Deaths'])
    county_codes = monthly[['county', 'fips']].drop_duplicates()
    county_codes = county_codes[county_codes.county.isin(counties)]
    if (county_codes.county.duplicated().any() or county_codes.fips.duplicated().any()
        or set(county_codes.county) != set(counties)):
        raise AuditError('Resolve the county-to-FIPS mapping against the monthly source.')
    checked = annual.merge(county_codes, on='fips')
    if not checked.county_x.eq(checked.county_y).all():
        raise AuditError('Annual and monthly county names disagree for a FIPS code.')
    monthly['outcome_month'] = monthly['Month Code'].astype(str).str.replace('/', '-', regex=False)
    deaths = pd.to_numeric(monthly.Deaths, errors='coerce')
    deaths = deaths.where(deaths.ge(0) & deaths.mod(1).eq(0))
    suppressed = monthly.Deaths.eq('Suppressed')
    monthly['y_lo'] = deaths.mask(suppressed, 1)
    monthly['y_hi'] = deaths.mask(suppressed, 9)
    annual['year'] = annual['Year Code'].astype(str)
    annual['population'] = pd.to_numeric(annual.Population, errors='coerce')
    periods = pd.period_range(start, end, freq='M')
    if not 1 <= len(periods) <= 120:
        raise AuditError('Choose an outcome window of 1–120 months.')
    if len(counties) * len(periods) > 12000:
        raise AuditError('Limit reconstruction to 12,000 county-months.')
    data = pd.MultiIndex.from_product([counties, periods], names=['county', 'period']).to_frame(index=False)
    data['outcome_month'] = data.period.astype(str)
    data['year'] = data.outcome_month.str[:4]
    data['mediator_month'] = (data.period - 1).astype(str)
    data['attendance_month'] = (data.period - 2).astype(str)
    data = data.drop(columns='period').merge(county_codes, on='county', validate='many_to_one')
    data = data.merge(monthly[['fips', 'outcome_month', 'y_lo', 'y_hi', 'Deaths', '_source_row']]
                      .rename(columns={'Deaths': 'cdc_deaths', '_source_row': 'monthly_source_row'}),
                      on=['fips', 'outcome_month'], how='left', validate='one_to_one')
    data = data.merge(annual[['fips', 'year', 'population', 'Deaths', '_source_row']]
                      .rename(columns={'Deaths': 'annual_deaths', '_source_row': 'annual_source_row'}),
                      on=['fips', 'year'], how='left', validate='many_to_one')
    data = data.merge(attendance_counts(attendance_files, counties),
                      on=['county', 'attendance_month'], how='left', validate='one_to_one')
    data = data.merge(pd.DataFrame({'menu': list(MENUS)}), how='cross').merge(
        snapshots, on=['county', 'mediator_month', 'menu'], how='left', validate='one_to_one')
    data['attendance_available'] = data.attendance.notna()
    data['tracker_snapshot_available'] = data.active_snapshot.notna()
    data['outcome_available'] = data.y_lo.notna() & data.population.gt(0)
    data['all_three_available'] = data.attendance_available & data.tracker_snapshot_available & data.outcome_available
    data['venue_path_available'] = data.attendance_available & data.venue_types.notna() & data.outcome_available
    columns = ['attendance_available', 'tracker_snapshot_available', 'outcome_available',
               'all_three_available', 'venue_path_available']
    coverage = data.groupby('menu', sort=False, as_index=False)[columns].sum()
    manifest = {
        'classification': 'synthetic demonstration' if synthetic else 'private research',
        'counties': len(counties), 'county_months': len(counties) * len(periods),
        'outcome_window': [str(periods[0]), str(periods[-1])],
        'lags': {'attendance': 2, 'tracker': 1},
        'suppression_interval': [1, 9], 'imputed_values': 0, 'models_fitted': 0,
        'inputs': [{'role': role, 'file': name, 'sha256': sha256(raw).hexdigest()}
                   for role, (name, raw) in [*files.items(), *(('attendance', x) for x in attendance_files)]],
    }
    report = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 source reconstruction</title>'
        '<style>body{font:16px system-ui;max-width:1050px;margin:3rem auto;padding:1rem}'
        'td,th{padding:.5rem;text-align:left}table{border-collapse:collapse}</style>'
        '<h1>H3 source reconstruction</h1><p>' + html.escape(manifest['classification']) + '</p>'
        f'<p>{manifest["county_months"]} county-months across {manifest["counties"]} counties.</p>'
        '<p>Attendance in t−2 → Tracker implementation in t−1 → CDC death counts in t.</p>'
        '<h2>Source coverage</h2>' + coverage.to_html(index=False)
        + '<p>Active snapshots count listed entries with applicable, defined start and stop answers. '
        'Unreported months and unresolved status remain unknown. Venue types describe recorded deployment breadth. '
        'Attendance counts explicit present marks per meeting, averaged across observed meetings. '
        'Meetings with no present marks remain unknown. Membership and staff roles require review.</p>'
        '<p>Suppressed counts retain the 1–9 interval. Confirm the CDC query filters against the research endpoint. '
        'This report reconstructs inputs and reports coverage. Model fitting and imputation have not run. '
        'The complete H3 scope also requires confirmed reach and nonfatal outcome data.</p>'
        '<p>county_month.csv contains three menu rows per county-month. tracker_source_cells.csv contains '
        'cell references and dictionary rows. run_metadata.json records file hashes and the time window.</p></html>'
    )
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('START_HERE.html', report)
        archive.writestr('county_month.csv', safe_csv(data))
        archive.writestr('tracker_source_cells.csv', safe_csv(evidence))
        archive.writestr('coverage.csv', safe_csv(coverage))
        archive.writestr('run_metadata.json', json.dumps(manifest, indent=2))
    return manifest, coverage, output.getvalue()


def demo_sources():
    book = Workbook()
    tracker = book.active
    tracker.title = 'orcca_tracker_v2'
    county_map = book.create_sheet('Sheet1')
    county_map.append(['ID', 'County'])
    keys = ['hcs_orc2record_id', 'redcap_event_name', 'hcs_orc2version']
    n = 'hcs_orc209'
    tracker.append(keys + [n + x for x in ('_1', 'l_1', 'v_1', 'd_1', 'o_1_yr', 'o_1_mo')])
    for cid in range(2):
        county_map.append([cid + 1, f'Demo {cid + 1}'])
        for month in range(1, 5):
            if cid == 1 and month == 3:
                continue
            tracker.append([cid + 1, f'{pd.Timestamp(2024, month, 1):%B}_2024_arm_1'.lower(), 'DEMO',
                            1, 1, None if cid == 0 and month == 3 else 0, cid + 1, 2024, 1])
    dictionary = Workbook()
    sheet = dictionary.active
    sheet.title = 'ORCCAT_DD'
    sheet.append(['Version', 'Variable / Field Name', 'Field Label', BRANCH, CHOICES])
    sheet.append(['DEMO', n+'l_1', 'Has implementation started?', '', '1, Yes | 0, No'])
    sheet.append(['DEMO', n+'v_1', 'Has this strategy been discontinued?', '', '1, Yes | 0, No'])
    sheet.append(['DEMO', n+'d_1', 'Venue', '', '1, Community center | 2, Mobile unit'])
    files = {}
    for role, workbook in [('tracker', book), ('dictionary', dictionary)]:
        buffer = BytesIO()
        workbook.save(buffer)
        files[role] = (f'synthetic_{role}.xlsx', buffer.getvalue())
    monthly, annual, attendance = [], [], []
    for i in range(1, 3):
        county, fips = f'Demo {i}', f'99{i:03d}'
        annual.append({'County': county, 'County Code': fips, 'Year Code': 2024, 'Population': 50000, 'Deaths': 25})
        for month in range(1, 7):
            monthly.append({'County': county, 'County Code': fips, 'Month Code': f'2024/{month:02d}',
                            'Deaths': 'Suppressed' if month == 4 else 0 if month == 5 else 10})
            if month != 2 or i == 1:
                attendance.append({'county': county, 'date': f'2024-{month:02d}-10', 'Name': 'Demo Person', 'Attendance': 'present'})
    for role, rows in [('monthly', monthly), ('annual', annual)]:
        files[role] = (f'synthetic_{role}.xls', pd.DataFrame(rows).to_csv(index=False, sep='\t').encode())
    return files, [('synthetic_attendance.csv', safe_csv(attendance))]
