"""Supplementary H3 documents with explicit county metadata and session memory."""
from io import BytesIO
from hashlib import sha256
import json
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import streamlit as st

from .engine import AuditError
from .h3_attendance_evidence import load_source, source_run_id
from .h3_evidence_memory import build_support, cached_search, fingerprint


def render_support(archive):
    st.subheader('Supplementary interviews and documents')
    st.caption('Upload PDF, DOCX or TXT and a source metadata CSV. Map each file to its county. '
               'The document date describes the interview or source. Attendance needs evidence '
               'for a particular meeting. Legacy DOC files can be converted to TXT or DOCX.')
    epoch = str(st.session_state.get('h3_source_epoch', 0))
    files = st.file_uploader('Supplementary H3 documents', type=['pdf','docx','txt'],
        accept_multiple_files=True, key='h3_support_files_'+epoch) or []
    metadata = st.file_uploader('Supplementary document metadata CSV', type=['csv'],
        key='h3_support_metadata_'+epoch)
    raw_files = [(f.name,f.getvalue()) for f in files]
    columns = ['source_file','county','document_type','document_date']
    template = pd.DataFrame([dict(source_file=name,county='',document_type='interview',
                                 document_date='') for name,_ in raw_files],columns=columns)
    st.download_button('Download document metadata template',template.to_csv(index=False),
                       'h3_document_metadata.csv','text/csv')
    raw = metadata.getvalue() if metadata else b''
    key = fingerprint([source_run_id(archive),[(n,sha256(b).hexdigest()) for n,b in raw_files],
                       sha256(raw).hexdigest()])
    stored = st.session_state.get('h3_support_corpus')
    if stored and stored[0]!=key:
        st.session_state.pop('h3_support_corpus',None)
        stored = None
    if st.button('Index supplementary H3 documents',disabled=not raw_files or not raw):
        try:
            mapping = pd.read_csv(BytesIO(raw),dtype=str,keep_default_na=False)
            if not set(columns)<=set(mapping.columns):
                raise AuditError('Metadata needs source_file, county, document_type and document_date.')
            data,_ = load_source(archive)
            result = build_support(raw_files,mapping,sorted(data.county.unique()),
                                    st.session_state.setdefault('h3_support_parse_memory',{}))
            stored = key,result
            st.session_state['h3_support_corpus']=stored
        except Exception as exc:
            st.error('Supplementary indexing failed. '+str(exc)[:240])
    if not stored:
        return
    docs, inventory = stored[1]
    st.write(f'{sum(r["status"]=="indexed" for r in inventory)} files indexed; '
             f'{sum(r["status"]=="excluded" for r in inventory)} excluded; {len(docs)} source segments.')
    st.dataframe(pd.DataFrame(inventory),hide_index=True)
    if not docs:
        return
    county = st.selectbox('Supplementary evidence county',sorted({d['county'] for d in docs}))
    question = st.text_input('Question for supplementary evidence',
                            value='coalition attendance meeting implementation start stop barriers')
    backend = st.selectbox('Supplementary retrieval method',['bm25','hybrid_rrf'])
    if st.button('Search supplementary H3 evidence',disabled=not docs):
        scope = [d for d in docs if d['county']==county]
        cache = st.session_state.setdefault('h3_query_memory',{})
        with st.spinner('Searching source evidence…'):
            result = cached_search(question,scope,cache,backend=backend)
        st.session_state['h3_support_search']=(key,county,question,backend,result)
    search = st.session_state.get('h3_support_search')
    if search and search[:4]==(key,county,question,backend):
        result = search[4]
        st.caption(f'{result["status"]}; reused retrieval: {result["memory_reused"]}')
        for doc in result['citations']:
            st.write(f'{doc["source_file"]} · {doc["locator_kind"]} {doc["page"]} · line {doc["line_start"]}')
            st.text(doc['text'])
    buf = BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED) as z:
        z.writestr('source_inventory.csv',pd.DataFrame(inventory).to_csv(index=False))
        z.writestr('original_source_memory.json',json.dumps(docs,ensure_ascii=False))
    st.download_button('Download supplementary source memory',buf.getvalue(),
                       'h3_supplementary_memory.zip','application/zip')
