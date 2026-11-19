"""Scoped attendance evidence retrieval and a separate review gate for H3 inputs.

Source memory contains original PDF text only. Retrieval never establishes a
headcount and never fills a value. This module has no model or network calls.
"""
from hashlib import sha256
from io import BytesIO
import json
import re
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from pypdf import PdfReader

from .engine import AuditError
from .exports import safe_csv
from .retrieval import retrieve
from .h3_evidence_memory import MAX_BYTES, MAX_FILES, VERSION, cached_search, fingerprint

QUERY = 'attendance roll call present participants meeting headcount people joined sign in'
REVIEW_COLUMNS = ['case_id', 'run_id', 'county', 'attendance_month', 'question',
                  'source_locators', 'original_evidence', 'decision', 'reviewer',
                  'reviewer_type', 'meeting_counts_json', 'definition_confirmed', 'review_reason']


def load_source(archive):
    with ZipFile(BytesIO(archive)) as z:
        data = pd.read_csv(z.open('county_month.csv'), dtype={'fips': str})
        metadata = json.loads(z.read('run_metadata.json'))
    return data, metadata


def source_run_id(archive):
    # Bind decisions to source content and settings, not ZIP timestamps or an
    # unrelated model-output member added to the download container.
    with ZipFile(BytesIO(archive)) as z:
        metadata = json.loads(z.read('run_metadata.json'))
        frozen = dict(county_month_sha256=sha256(z.read('county_month.csv')).hexdigest(),
                      inputs=metadata.get('inputs', []), lags=metadata.get('lags', {}),
                      outcome_window=metadata.get('outcome_window', []))
    return sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def parse_pdf(name, raw, counties):
    if len(raw) > 20 * 1024 * 1024 or not raw.startswith(b'%PDF'):
        raise AuditError('Provide a PDF under 20 MB.')
    county = [c for c in counties if re.search(r'(?<![A-Za-z])' + re.escape(c) + r'(?![A-Za-z])', name, re.I)]
    iso = re.search(r'(20\d{2})-(\d{2})-(\d{2})', name)
    compact = re.search(r'_(\d{2})(\d{2})(20\d{2})', name)
    date = iso.group(0) if iso else (f'{compact[3]}-{compact[1]}-{compact[2]}' if compact else '')
    if len(county) != 1 or not date:
        raise AuditError('PDF filename must identify exactly one county and a YYYY-MM-DD or MMDDYYYY date.')
    date = pd.Timestamp(date).strftime('%Y-%m-%d')
    reader = PdfReader(BytesIO(raw))
    if len(reader.pages) > 150:
        raise AuditError('PDF exceeds the 150-page evidence limit.')
    digest = sha256(raw).hexdigest()
    docs = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ''
        # Short overlapping source segments fit the context window of a local model.
        words = list(re.finditer(r'\S+', text))
        for start in range(0, len(words), 140):
            end = min(start + 180, len(words))
            excerpt = text[words[start].start():words[end - 1].end()]
            docs.append(dict(chunk_id=f'{digest[:20]}:p{number}:w{start}', source_file=name,
                pdf_sha256=digest, county=county[0], month=date[:7], meeting_date=date,
                scope_basis='filename county and meeting date; requires substantive review',
                page=number, line_start=text[:words[start].start()].count('\n') + 1,
                text=excerpt, sha256=sha256(excerpt.encode()).hexdigest(),
                classification='private_local', role='original_source'))
    return docs


def build_evidence_review(source_archive, pdf_files, *, synthetic=False, parse_cache=None, query_cache=None):
    data, metadata = load_source(source_archive)
    expected = 'synthetic demonstration' if synthetic else 'private research'
    if metadata['classification'] != expected:
        raise AuditError('Evidence review classification does not match the current reconstruction.')
    if not 1 <= len(pdf_files) <= MAX_FILES or sum(len(raw) for _, raw in pdf_files) > MAX_BYTES:
        raise AuditError('Provide one to 300 meeting PDFs, at most 200 MB total.')
    parse_cache = {} if parse_cache is None else parse_cache
    query_cache = {} if query_cache is None else query_cache
    inventory = []
    run_id = source_run_id(source_archive)
    documents, seen = [], set()
    for name, raw in pdf_files:
        digest = sha256(raw).hexdigest()
        if digest in seen:
            inventory.append(dict(source_file=name, sha256=digest, status='duplicate_bytes'))
            continue
        seen.add(digest)
        key = fingerprint([VERSION, name, digest, sorted(data.county.unique())])
        try:
            reused = key in parse_cache
            if not reused:
                parsed = parse_pdf(name, raw, sorted(data.county.unique()))
                if not parsed:
                    raise AuditError('No extractable text found.')
                parse_cache[key] = [dict(d, document_type='meeting') for d in parsed]
            documents.extend(parse_cache[key])
            inventory.append(dict(source_file=name, sha256=digest, status='indexed',
                                  chunks=len(parse_cache[key]), parse_reused=reused))
        except Exception as error:
            inventory.append(dict(source_file=name, sha256=digest, status='excluded', reason=str(error)[:240]))
    if synthetic:
        for doc in documents:
            doc['classification']='synthetic'
    if not documents:
        raise AuditError('No extractable PDF evidence was supplied.')
    cases, logs, forms = [], [], []
    gaps = data[data.attendance.isna()].groupby(['county', 'attendance_month'], sort=True)
    for (county, month), group in gaps:
        case_id = f'{county.upper()}-{month}'
        scoped = [d for d in documents if d['county'] == county and d['month'] == month]
        result = cached_search(QUERY, scoped, query_cache, top_k=5)
        citations = result['citations']
        potential = group.loc[group.active_snapshot.notna() & group.outcome_available, 'menu'].tolist()
        case = dict(case_id=case_id, run_id=run_id, county=county, attendance_month=month,
                    outcome_month=group.outcome_month.iloc[0], original_value=None,
                    menus_ready_except_attendance=' | '.join(potential),
                    potential_menu_paths=len(potential), source_pdfs=len({d['pdf_sha256'] for d in scoped}),
                    retrieved_chunks=len(citations), retrieval_status=result['status'],
                    proposed_attendance=None, system_assessment='Not established by retrieval; review original attendance evidence',
                    human_confirmed=False)
        locators = ' | '.join(f"{d['source_file']} p.{d['page']} lines from {d['line_start']} [{d['chunk_id']}]" for d in citations)
        cases.append(dict(case, source_locators=locators))
        logs.append(dict(case_id=case_id, query=QUERY, county=county, month=month, **result))
        forms.append(dict(case_id=case_id, run_id=run_id, county=county, attendance_month=month,
            question='What is the mean distinct explicitly present attendance per observed meeting in this county-month? Identify meeting dates and primary count evidence, or retain unknown.',
            source_locators=locators, original_evidence='\n\n'.join(d['text'] for d in citations),
            **{key: '' for key in REVIEW_COLUMNS[7:]}))
    queue = pd.DataFrame(cases).sort_values(['potential_menu_paths', 'source_pdfs', 'case_id'], ascending=[False, False, True])
    manifest = dict(run_id=run_id, classification=expected,
        candidate_county_months=metadata['county_months'], missing_attendance_cases=len(cases),
        source_pdfs=sum(r['status']=='indexed' for r in inventory), source_chunks=len(documents),
        uploaded_files=len(pdf_files), excluded_files=sum(r['status']=='excluded' for r in inventory),
        duplicate_files=sum(r['status']=='duplicate_bytes' for r in inventory),
        parsed_from_memory=sum(r.get('parse_reused',False) for r in inventory),
        retrieval_memory_hits=sum(bool(r['citations']) and r.get('memory_reused',False) for r in logs),
        file_inventory=inventory,
        cases_with_same_county_month_pdf=int(queue.source_pdfs.gt(0).sum()),
        actual_bm25_queries=sum(bool(r['citations']) and not r.get('memory_reused',False) for r in logs),
        priority_cases_with_evidence=int((queue.source_pdfs.gt(0) & queue.potential_menu_paths.gt(0)).sum()),
        human_confirmed_values=0, applied_values=0, llm_calls=0, external_model_calls=0,
        memory_kind='versioned original-source retrieval memory; no generated answers or reviewer labels',
        evidence_scope='Supplied PDFs only. Filename dates are retrieval filters, not verified measurement dates.')
    memory = dict(run_id=run_id, documents=documents,
                  memory_sha256=sha256(json.dumps(documents, sort_keys=True).encode()).hexdigest())
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as z:
        z.writestr('review_queue.csv', safe_csv(queue))
        z.writestr('A_independent_reviewer.csv', safe_csv(pd.DataFrame(forms, columns=REVIEW_COLUMNS)))
        z.writestr('source_memory.json', json.dumps(memory, indent=2))
        z.writestr('retrieval_log.json', json.dumps(logs, indent=2))
        z.writestr('run_manifest.json', json.dumps(manifest, indent=2))
    return manifest, queue, logs, memory, output.getvalue()


def apply_attendance_reviews(source_archive, memory, reviews):
    """Apply explicit reviewer decisions to missing values only, retaining a ledger."""
    data, metadata = load_source(source_archive)
    if metadata['classification'] != 'private research':
        raise AuditError('Research additions require a private reconstruction.')
    run_id = source_run_id(source_archive)
    if memory.get('run_id') != run_id:
        raise AuditError('Source memory belongs to a different reconstruction.')
    documents = memory['documents']
    if sha256(json.dumps(documents, sort_keys=True).encode()).hexdigest() != memory['memory_sha256']:
        raise AuditError('Source memory changed.')
    by_id = {d['chunk_id']: d for d in documents}
    if len(by_id) != len(documents) or any(d.get('role') != 'original_source' for d in documents):
        raise AuditError('Only uniquely identified original source evidence is allowed.')
    if any(sha256(d['text'].encode()).hexdigest() != d['sha256'] for d in documents):
        raise AuditError('Evidence text changed.')
    required = set(REVIEW_COLUMNS)
    if not required <= set(reviews.columns) or reviews.case_id.duplicated().any():
        raise AuditError('Use the unique-case reviewer template.')
    if not set(reviews.decision.str.strip().str.lower()) <= {'', 'approved', 'rejected', 'unknown'}:
        raise AuditError('Decision must be approved, rejected, unknown or blank.')
    ledger = []
    for row in reviews[reviews.decision.str.lower().str.strip().eq('approved')].itertuples():
        if row.run_id != run_id or row.case_id != f'{row.county.upper()}-{row.attendance_month}':
            raise AuditError('Review case or run does not match the current inputs.')
        if row.reviewer_type.strip().lower() != 'human' or not row.reviewer.strip() or not row.review_reason.strip():
            raise AuditError('An approved review needs an identified human reviewer and a reason. AI outputs are not human approvals.')
        if row.definition_confirmed.strip().lower() != 'yes':
            raise AuditError('Confirm the distinct explicitly present attendance definition.')
        selected = data.county.eq(row.county) & data.attendance_month.eq(row.attendance_month)
        if not selected.any() or data.loc[selected, 'attendance'].notna().any():
            raise AuditError('Only currently missing attendance may be completed.')
        try:
            meetings = json.loads(row.meeting_counts_json)
        except (ValueError, TypeError) as error:
            raise AuditError('Provide a JSON list of meeting dates, counts, chunk IDs and exact quotes.') from error
        if not isinstance(meetings, list) or not meetings:
            raise AuditError('At least one observed meeting is required.')
        dates, counts = set(), []
        for meeting in meetings:
            date = pd.Timestamp(meeting['date']).strftime('%Y-%m-%d')
            count = meeting['count']
            if date in dates or date[:7] != row.attendance_month:
                raise AuditError('Meeting dates must be unique and belong to the target month.')
            dates.add(date)
            if isinstance(count, bool) or not isinstance(count, (float, int)) or not np.isfinite(count) or count < 0 or count != int(count):
                raise AuditError('Each meeting requires a nonnegative integer headcount.')
            doc = by_id.get(meeting.get('chunk_id'))
            quote = meeting.get('quote', '')
            if (not doc or doc['county'] != row.county or doc['month'] != row.attendance_month
                    or doc['meeting_date'] != date or not isinstance(quote, str)
                    or not quote.strip() or quote not in doc['text']):
                raise AuditError('Count evidence must be an exact quote from the same county and meeting date.')
            counts.append(count)
        value = float(np.mean(counts))
        data.loc[selected, 'attendance'] = value
        data.loc[selected, 'attendance_source'] = 'Evidence-reviewed addition: ' + row.case_id
        ledger.append(dict(case_id=row.case_id, run_id=run_id, county=row.county,
                           attendance_month=row.attendance_month, original_value=None,
                           reviewed_value=value, reviewer=row.reviewer, reviewer_type='human',
                           meetings=meetings, review_reason=row.review_reason))
    data['attendance_available'] = data.attendance.notna()
    data['all_three_available'] = data.attendance_available & data.active_snapshot.notna() & data.outcome_available
    data['venue_path_available'] = data.attendance_available & data.venue_types.notna() & data.outcome_available
    metadata.update(imputed_values=len(ledger), completion_method='source evidence with explicit human review',
                    approved_attendance_county_months=len(ledger))
    output = BytesIO()
    with ZipFile(BytesIO(source_archive)) as source, ZipFile(output, 'w', ZIP_DEFLATED) as target:
        for name in source.namelist():
            if name not in {'county_month.csv', 'run_metadata.json', 'coverage.csv', 'START_HERE.html'}:
                target.writestr(name, source.read(name))
        target.writestr('county_month.csv', safe_csv(data))
        target.writestr('run_metadata.json', json.dumps(metadata, indent=2))
        target.writestr('approved_attendance_ledger.json', json.dumps(ledger, indent=2))
        target.writestr('coverage.csv', safe_csv(data.groupby('menu', as_index=False)[[
            'attendance_available', 'tracker_snapshot_available', 'outcome_available',
            'all_three_available', 'venue_path_available']].sum()))
        target.writestr('START_HERE.html', '<h1>Evidence-reviewed derived H3 inputs</h1><p>Read the review ledger and metadata. These are derived inputs; original files remain unchanged.</p>')
    return data, ledger, output.getvalue()
