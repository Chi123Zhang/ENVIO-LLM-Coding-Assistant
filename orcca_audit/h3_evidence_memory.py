"""Session-owned source parsing and retrieval caches for the H3 evidence corpus."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePath
import re

import pandas as pd
from docx import Document
from pypdf import PdfReader

from .engine import AuditError
from .retrieval import retrieve, validate_documents

VERSION = 'h3-evidence-memory-2'
MAX_FILES = 300
MAX_BYTES = 200 * 1024 * 1024


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cached_search(query, docs, cache, *, top_k=5, backend='bm25'):
    validate_documents(docs)
    from .hybrid import MODEL_REVISIONS
    key = fingerprint([VERSION, MODEL_REVISIONS, query, backend, top_k,
                       [(d['chunk_id'], d['sha256']) for d in docs]])
    if key in cache:
        return dict(cache[key], memory_reused=True)
    if backend == 'hybrid_rrf' and docs:
        from .hybrid import MODEL_REVISIONS, rrf_retrieve
        try:
            ranked,reused = rrf_retrieve(query,docs,top_k,cache.setdefault('scoped_vector_indexes',{}))
            result = dict(status='retrieved_evidence_unreviewed',backend=backend,
                citations=[dict(doc,score=score) for doc,score in ranked],
                explanation_kind='extractive_evidence_not_model_explanation',
                model_revisions=dict(MODEL_REVISIONS),inference_location='application_cpu',
                index_reused=reused)
        except Exception as error:
            result = dict(status='hybrid_unavailable_no_fallback',backend=backend,citations=[],
                          reason=type(error).__name__)
    else:
        result = retrieve(query, docs, backend=backend, top_k=top_k)
    cache[key] = result
    return dict(result, memory_reused=False)


def parse_support(name, raw, county, counties, document_type='interview', document_date=''):
    if county not in counties:
        raise AuditError('Map this file to a county in the source data.')
    if document_type not in {'interview', 'supporting_document'}:
        raise AuditError('Supporting documents need an interview or supporting_document type.')
    if len(raw) > 20 * 1024 * 1024:
        raise AuditError('Supporting file exceeds 20 MB.')
    date = pd.Timestamp(document_date).strftime('%Y-%m-%d') if document_date else ''
    extension = PurePath(name).suffix.lower()
    if extension == '.pdf':
        reader = PdfReader(BytesIO(raw))
        if len(reader.pages) > 150:
            raise AuditError('Supporting PDF exceeds 150 pages.')
        sections = [(i, p.extract_text() or '') for i, p in enumerate(reader.pages, 1)]
        locator_kind = 'page'
    elif extension == '.docx':
        document = Document(BytesIO(raw))
        text = '\n'.join([p.text for p in document.paragraphs] +
                         [' | '.join(c.text for c in row.cells) for table in document.tables for row in table.rows])
        sections, locator_kind = [(1, text)], 'paragraph_text'
    elif extension == '.txt':
        sections, locator_kind = [(1, raw.decode('utf-8-sig'))], 'line'
    else:
        raise AuditError('Upload PDF, DOCX or UTF-8 TXT. Convert legacy DOC files to DOCX or TXT first.')
    digest = sha256(raw).hexdigest()
    docs = []
    for page, text in sections:
        words = list(re.finditer(r'\S+', text))
        for start in range(0, len(words), 140):
            end = min(start + 180, len(words))
            excerpt = text[words[start].start():words[end-1].end()]
            docs.append(dict(chunk_id=f'{digest[:20]}:p{page}:w{start}', source_file=name,
                pdf_sha256=digest, county=county, month='', meeting_date='',
                document_date=date, document_type=document_type, locator_kind=locator_kind,
                scope_basis='User-supplied county and document metadata; meeting date not inferred',
                page=page, line_start=text[:words[start].start()].count('\n')+1,
                text=excerpt, sha256=sha256(excerpt.encode()).hexdigest(),
                classification='private_local', role='original_source'))
    if not docs:
        raise AuditError('No extractable text found.')
    return docs


def build_support(files, mapping, counties, cache):
    if len(files) > MAX_FILES or sum(len(raw) for _, raw in files) > MAX_BYTES:
        raise AuditError('Use at most 300 documents and 200 MB per batch.')
    if mapping.source_file.duplicated().any():
        raise AuditError('Each supplementary filename needs one metadata row.')
    lookup = mapping.fillna('').set_index('source_file').to_dict('index')
    docs, inventory, seen = [], [], set()
    for name, raw in files:
        info = lookup.get(name, {})
        digest = sha256(raw).hexdigest()
        key = fingerprint([VERSION, name, digest, info])
        record = dict(source_file=name, sha256=digest, **info)
        try:
            if digest in seen:
                inventory.append(dict(record, status='duplicate_bytes'))
                continue
            seen.add(digest)
            if key not in cache:
                cache[key] = parse_support(name, raw, info.get('county',''), counties,
                    info.get('document_type','interview'), info.get('document_date',''))
                reused = False
            else:
                reused = True
            docs.extend(cache[key])
            inventory.append(dict(record, status='indexed', chunks=len(cache[key]), parse_reused=reused))
        except Exception as exc:
            inventory.append(dict(record, status='excluded', reason=str(exc)[:240]))
    return docs, inventory
