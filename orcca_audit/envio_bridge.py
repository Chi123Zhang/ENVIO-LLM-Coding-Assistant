"""Explicit private-session handoff of ENVIO coding, not an ORCCA codebook.

Only exact original quotations enter retrieval. Labels and review answers remain
outside the index. No model, persistence, county inference or numeric imputation.
"""
from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import re

from .completion import digest
from .engine import AuditError
from .policy import private_uploads_allowed
from .workflow import start_project


def input_signature(files, urls, settings):
    return digest([[{'name': f.name, 'sha256': hashlib.sha256(f.getvalue()).hexdigest()}
        for f in files or []], urls, settings])


def capture_run(state, frame, sources, signature, settings):
    # Recording an existing ENVIO result makes no new API call and no API claim.
    state.pop('envio_orcca_capture', None)
    if not private_uploads_allowed():return
    names = [s['source_name'] for s in sources]
    if len(names) != len(set(names)):
        return  # Ambiguous merged names must not create a misleading handoff.
    state['envio_orcca_capture'] = {'capture_id': digest([signature, frame.to_json()]),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_signature': signature, 'rows': frame.to_dict('records'),
        'sources': {s['source_name']: s['source_text'] for s in sources},
        'settings': deepcopy(settings), 'codebook_scope': 'ENVIO; applicability to ORCCA requires human review'}


def transfer_segment(state, project, capture, row_index, quote, reviewer, reason,
                     relevance_confirmed, county='', month=''):
    if not private_uploads_allowed() or project['classification'] != 'private_research':
        raise AuditError('ENVIO handoff requires a private project and private server authorization.')
    if state.get('workflow_project') is not project or state.get('envio_orcca_capture') != capture:
        raise AuditError('The coding run or project changed. Select the current run again.')
    if not relevance_confirmed or not reviewer.strip() or not reason.strip():
        raise AuditError('Confirm ORCCA relevance and provide a reviewer and applicability reason.')
    from .english_export import english_text
    for v in (reviewer, reason):english_text(v)
    try:
        if not isinstance(row_index,int) or not 0 <= row_index < len(capture['rows']):raise IndexError
        row = capture['rows'][row_index]
    except (IndexError, TypeError):raise AuditError('Select a current coded segment.') from None
    raw = capture['sources'].get(row['source_name'], '')
    if not quote.strip() or quote not in str(row['text']) or quote not in raw:
        raise AuditError('The quotation must occur exactly in both the coded segment and the extracted source. Reworded model text is not evidence.')
    if not str(row.get('codes', '')).strip() or str(row.get('source_type', '')).lower() == 'error':
        raise AuditError('This segment has no usable coding output.')
    county, month = county.strip(), month.strip()
    counties = set(project['data']['panel']['county'].astype(str))
    if county and county not in counties:raise AuditError('County must match this project; leave blank if unknown.')
    if month and (not county or not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', month)):
        raise AuditError('An applicable month needs an explicit county and YYYY-MM. Leave unknown scope blank.')
    start = raw.find(quote)
    # Multiple identical occurrences cannot be assigned a unique source locator.
    if raw.find(quote, start+1) >= 0:raise AuditError('Quote occurs more than once. Select a longer unique passage.')
    source_hash = hashlib.sha256(raw.encode()).hexdigest()
    cid = 'envio-' + digest([capture['capture_id'], row_index, quote])[:20]
    previous = project['data'].get('implementation_evidence', [])
    if any(x['candidate_id'] == cid for x in previous):raise AuditError('This quotation is already linked to this project.')
    doc = {'chunk_id': cid, 'source_id': row['source_name'], 'source_file': row['source_name'],
        'page': None, 'line_start': raw[:start].count('\n')+1,
        'line_end': raw[:start+len(quote)].count('\n')+1,
        'locator': f'ENVIO extracted source text, characters {start+1}–{start+len(quote)}; original page not retained',
        'text': quote, 'sha256': hashlib.sha256(quote.encode()).hexdigest(),
        'classification': 'private_local', 'evidence_kind': 'original_excerpt'}
    record = {'candidate_id': cid, 'source_chunk_id': cid, 'source_file': row['source_name'],
        'extracted_source_sha256': source_hash, 'capture_id': capture['capture_id'],
        'segment_index': row.get('segment_index'), 'envio_codes': str(row['codes']),
        'model_rationale': str(row.get('rationale', '')), 'run_settings': capture['settings'],
        'codebook_scope': capture['codebook_scope'], 'county': county, 'applicable_month': month,
        'scope_status': 'human_assigned_unverified' if county else 'unknown',
        'reviewer': reviewer.strip(), 'applicability_reason': reason.strip(),
        'linked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'linked_candidate_not_validated_measure', 'parent_run_id': project['run_id']}
    data = dict(project['data']); data['implementation_evidence'] = previous + [record]
    manifest = deepcopy(project['manifest'])
    manifest['envio_handoff'] = {'count': len(data['implementation_evidence']),
        'ledger_sha256': digest(data['implementation_evidence']), 'policy': 'Original quotations only in retrieval; labels excluded'}
    return start_project(state, data, project['documents']+[doc], project['completion_evidence'], manifest,
        private=True, parent=project['run_id'], ledger=project['completion_ledger'])
