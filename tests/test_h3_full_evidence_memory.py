from io import BytesIO
import json
from zipfile import ZipFile

import pandas as pd
import pytest
from docx import Document

from orcca_audit.h3_attendance_evidence import build_evidence_review
from orcca_audit.h3_ai_completion import run_completion,prepare_experiment,generation_key
from orcca_audit.h3_evidence_memory import build_support
from test_h3_ai_completion import fixture


def test_large_corpus_and_retrieval_reuse_preserve_scope():
    archive,memory,_ = fixture()
    # Repeat bytes are uploaded aliases, not independent meeting evidence.
    from reportlab.pdfgen.canvas import Canvas
    b=BytesIO();c=Canvas(b);c.drawString(40,740,'Attendance was 12 people present.');c.save()
    files=[(f'Demo 2_2024-02-10_copy{i}.pdf',b.getvalue()) for i in range(79)]
    cache,queries={},{}
    first=build_evidence_review(archive,files,synthetic=True,parse_cache=cache,query_cache=queries)
    second=build_evidence_review(archive,files,synthetic=True,parse_cache=cache,query_cache=queries)
    assert first[0]['uploaded_files']==79 and first[0]['duplicate_files']==78
    assert second[0]['parsed_from_memory']==1
    assert second[0]['actual_bm25_queries']==0
    assert all(d['county']=='Demo 2' and d['month']=='2024-02' for d in second[3]['documents'])


def test_changed_document_context_invalidates_generation_cache():
    _,_,p=fixture()
    unknown=json.dumps(dict(status='unknown',count=None,chunk_id='',quote='',reason='Unknown.'))
    calls=[]
    def generate(t):
        calls.append(t);return dict(raw_response=unknown)
    cache={}
    a=run_completion(p,generate,{},output_cache=cache)
    b=run_completion(p,generate,{},output_cache=cache)
    assert len(calls)==1 and a['summary']['llm_calls']==1 and b['summary']['llm_calls']==0
    p['tasks'][0]['evidence'][0]['text']+=' Updated source.'
    run_completion(p,generate,{},output_cache=cache)
    assert len(calls)==2


def test_interview_date_never_becomes_meeting_attendance_date():
    archive,_,_=fixture()
    b=BytesIO();d=Document();d.add_paragraph('Our coalition has 28 members.');d.save(b)
    files=[('interview.docx',b.getvalue()),('unmapped.txt',b'test')]
    mapping=pd.DataFrame([dict(source_file='interview.docx',county='Demo 2',
        document_type='interview',document_date='2024-02-10')])
    docs,inventory=build_support(files,mapping,['Demo 1','Demo 2'],{})
    assert inventory[1]['status']=='excluded'
    assert docs[0]['meeting_date']==docs[0]['month']==''
    assert docs[0]['document_date']=='2024-02-10'


def test_prior_raw_answers_are_rechecked_after_import():
    from orcca_audit.h3_ai_completion import compare_experiment,import_generation_memory
    archive,_,p=fixture()
    raw=json.dumps(dict(status='unknown',count=None,chunk_id='',quote='',reason='Unknown.'))
    r=run_completion(p,lambda t:dict(raw_response=raw),{})
    package=compare_experiment(p,r)[2]
    cache=import_generation_memory(package,archive,archive)
    repeated=run_completion(p,None,{},output_cache=cache)
    assert repeated['summary']['llm_calls']==0
    assert repeated['summary']['generation_memory_hits']==len(p['tasks'])
    assert repeated['summary']['candidate_county_months']==0
