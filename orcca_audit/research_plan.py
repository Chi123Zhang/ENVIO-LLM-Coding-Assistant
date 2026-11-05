"""Researcher-defined questions and deterministic data requirements; no inference."""
from copy import deepcopy
import hashlib
import json
import re

from .engine import AuditError, empty

MODES = ['H3 mediation preparation', 'Implementation record review', 'Custom question (planning only)']
DEFAULT = {
    'mode': MODES[0], 'hypothesis': 'H3',
    'question': 'Do EBP implementation and reach help explain the relationship between coalition processes and later overdose outcomes?',
    'unit': 'County-month', 'exposure': '', 'mediator': '', 'outcome': '',
    'start_month': '', 'end_month': '', 'time_order': '', 'confounders': '', 'definition_source': ''}


def normalize(plan):
    if not isinstance(plan, dict): raise AuditError('Save a research question first.')
    p = {key: str(plan.get(key, '')).strip() for key in DEFAULT}
    if p['mode'] not in MODES: raise AuditError('Choose a supported planning route.')
    if not p['hypothesis'] or not p['question'] or not p['unit']:
        raise AuditError('A question label, research question and analysis unit are required.')
    if any(len(v) > 2000 for v in p.values()): raise AuditError('Keep each planning field below 2,000 characters.')
    from .english_export import english_text
    for v in p.values(): english_text(v)
    for key in ('start_month', 'end_month'):
        if p[key] and not re.fullmatch(r'\d{4}-(?:0[1-9]|1[0-2])', p[key]):
            raise AuditError('Planning months must use YYYY-MM.')
    if p['start_month'] and p['end_month'] and p['start_month'] > p['end_month']:
        raise AuditError('The planning end month precedes the start month.')
    return p


def fingerprint(plan):
    return hashlib.sha256(json.dumps(normalize(plan), sort_keys=True).encode()).hexdigest()


def requirements(data):
    """Presence is an inventory result, never validation of a measure or causal design."""
    p = normalize(data.get('research_plan', DEFAULT))
    rows = []
    def add(component, selected, status, basis, next_step):
        rows.append({'component': component, 'selected_measure': selected or 'Not specified',
                     'status': status, 'basis': basis, 'next_step': next_step})
    records = data['study_records']; process = data.get('coalition_process')
    dictionary = data.get('dictionary')
    add('Tracker records', 'Implementation records', 'Supplied',
        f'{len(records)} county-month records; no claim about independent sample size.',
        'Review versions, visibility rules, strategy identity and event dates.')
    add('Version-specific definitions', 'REDCap dictionary', 'Supplied; source review needed' if dictionary is not None else 'Not supplied',
        'Field definitions describe questions, not unobserved county-month values.',
        'Supply the applicable dictionary and check the exact version-field rows.')
    if p['mode'] != MODES[1]:
        selected = process is not None and bool(p['exposure']) and p['exposure'] in set(process.measure.astype(str))
        add('Coalition process / exposure (X)', p['exposure'],
            'Named measure supplied; validation pending' if selected else 'Measure not linked' if process is not None else 'Not supplied',
            'A process table is supplied.' if process is not None else 'No defined coalition-process table was supplied.',
            'Link the selected measure to a source, county and applicable time. Attendance is not selected automatically.')
    add('Implementation / reach' + (' (M)' if p['mode'] == MODES[0] else ''), p['mediator'],
        'Candidate source available; measure not validated',
        'Tracker data are available. Strategy counts and actual reach are different measures.',
        'Specify the exact field or construction rule, version and treatment of no-update records.')
    if p['mode'] != MODES[1]:
        add('Later outcome' + (' (Y)' if p['mode'] == MODES[0] else ''), p['outcome'],
            'Source supplied; definition pending' if 'monthly' in data or 'annual' in data else 'Not supplied',
            'Mortality files are present.' if 'monthly' in data or 'annual' in data else 'No outcome source was supplied.',
            'Verify outcome definition, population, geography, suppression and follow-up.')
    lo, hi = records.month.min(), records.month.max()
    requested = (p['start_month'] or 'Unspecified') + ' to ' + (p['end_month'] or 'Unspecified')
    outside = bool((p['start_month'] and p['start_month'] < lo) or (p['end_month'] and p['end_month'] > hi))
    add('Requested time coverage', requested, 'Outside Tracker coverage' if outside else 'Coverage needs confirmation',
        f'Tracker event months span {lo} to {hi}; this does not establish outcome or process coverage.',
        'Confirm coverage for every required source and distinguish event time from report-entry time.')
    add('Design and temporal order', p['time_order'], 'Not established',
        'A planning statement is not an identification check.',
        'Prespecify the time order and confounders before statistical analysis. No effect model is fitted.')
    return rows


def save(state, plan):
    p = normalize(plan)
    current = state.get('workflow_project')
    state['research_plan_saved'] = deepcopy(p)
    if current and current['data'].get('_study_kind') == 'h3':
        if fingerprint(current['data'].get('research_plan', DEFAULT)) == fingerprint(p): return current
        from .workflow import start_project
        data = dict(current['data']); data['research_plan'] = p
        current = start_project(state, data, current['documents'], current['completion_evidence'], current['manifest'],
            private=current['classification'] == 'private_research', parent=current['run_id'], ledger=current['completion_ledger'])
    state['study_query'] = p['question']
    state.pop('study_search_result', None)
    return current
