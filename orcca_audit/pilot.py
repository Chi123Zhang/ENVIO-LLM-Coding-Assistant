"""Small, source-only completion pilot. References never enter retrieval memory.

Observed masked values are recorded-value references, not human truth labels.
No fitting, external inference, carry-forward, or modification of study inputs.
"""
from copy import deepcopy
from io import BytesIO
import json
import re
import time
import zipfile
from collections import Counter
from pathlib import Path
import hashlib
import platform
from importlib.metadata import version

import pandas as pd

from .completion import canonical, digest
from .engine import AuditError, empty
from .exports import safe_csv
from .retrieval import retrieve, validate_documents


def branching(expression, row):
    """Fail closed; only flat REDCap binary equalities joined by AND or OR."""
    if empty(expression):return 'applicable'
    parts=re.split(r'\s+(and|or)\s+',str(expression).strip(),flags=re.I)
    operators={x.lower() for x in parts[1::2]}
    if len(operators)>1:return 'unresolved_branching'
    values=[]
    for term in parts[::2]:
        match=re.fullmatch(r"\[([a-zA-Z0-9_]+)\]\s*=\s*'([01])'",term.strip())
        if not match:return 'unresolved_branching'
        field,value=match.groups()
        if field not in row or empty(row[field]):values.append(None)
        else:values.append(str(row[field]).removesuffix('.0')==value)
    if operators=={'or'}:
        result=True if True in values else None if None in values else False
    else:result=False if False in values else None if None in values else True
    return 'applicable' if result is True else 'structural_skip' if result is False else 'unresolved_branching'


def inventory(project):
    """Inspect only slot-1 implementation-start fields with exact dictionary rules."""
    data=project['data'];raw=data['raw_tracker'];dd=data.get('dictionary')
    if dd is None:raise AuditError('Add a version-specific source dictionary before the pilot.')
    definitions={(str(r['Version']),str(r['Variable / Field Name'])):r for r in dd.to_dict('records')}
    records=data['study_records'].to_dict('records')
    if len(raw)!=len(records):raise AuditError('Tracker and record-map lengths differ.')
    rows=[]
    for record,r in zip(records,raw.to_dict('records')):
        version=str(r['hcs_orc2version'])
        for menu,field in [('OEND','hcs_orc209l_1'),('MOUD','hcs_orc210m_1')]:
            d=definitions.get((version,field));value=r.get(field)
            branch=branching(d.get('Branching Logic (Show field only if...)'),r) if d else 'definition_unavailable'
            choices=str(d.get('Choices, Calculations, OR Slider Labels','')) if d else ''
            options={x.strip().lower() for x in choices.split('|')}
            binary_definition=options=={'1, yes','0, no'}
            try:code=canonical(value,'started')
            except AuditError:code=None;branch='invalid_status_code'
            status=('observed_applicable' if code is not None else 'unknown_applicable') if branch=='applicable' and binary_definition else branch
            if branch=='applicable' and not binary_definition:status='coding_requires_review'
            row={**record,'version':version,'menu':menu,'slot':'1','field':'started','source_field':field,
                'source_sheet':'orcca_tracker_v2','source_row':r.get('_source_row',record.get('source_row')),
                'recorded_value':code,'recorded_presence':'blank_or_unknown' if code is None else 'recorded_binary',
                'eligibility':status,'branch_rule':str(d.get('Branching Logic (Show field only if...)') or '') if d else '',
                'dictionary_row':d.get('_source_row','') if d else '',
                'question':str(d.get('Field Label','Definition unavailable')) if d else 'Definition unavailable',
                'strategy_identity':'slot 1; cross-month identity not assumed'}
            row['case_id']='P-'+digest([record['county'],record['month'],version,field])[:16]
            rows.append(row)
    return rows


def freeze_memory(project):
    """Source memory only. No Tracker values, labels, decisions or generated reports.

    Only dictionary definitions are admitted automatically. Independent scoped
    source assertions require their own future provenance review before scoring.
    This strict first pilot deliberately excludes uploaded answer-like excerpts.
    """
    docs=[]
    for source in project['documents']:
        if source.get('field') not in {'hcs_orc209l_1','hcs_orc210m_1'}:continue
        if source.get('evidence_kind')!='original_excerpt':continue
        if source.get('role','source_excerpt')!='source_excerpt':continue
        d=deepcopy(source);d['memory_role']='source_definition_only';docs.append(d)
    validate_documents(docs)
    return {'schema_version':1,'project_fingerprint':project['input_fingerprint'],
        'memory_hash':digest(docs),'documents':docs,'frozen':True,
        'policy':'Only source dictionary definitions. No target records, neighboring snapshots, labels, reviewer decisions or generated outputs. No county-state inference from definitions.'}


def _sample(pool, n):
    # Deterministic interleaving across county/menu/version/value; no random seed search.
    groups={}
    for r in sorted(pool,key=lambda r:r['case_id']):
        key=(r['county'],r['menu'],r['version'],r['recorded_value'])
        groups.setdefault(key,[]).append(r)
    selected=[]
    while groups and len(selected)<n:
        for key in sorted(list(groups),key=str):
            selected.append(groups[key].pop(0))
            if not groups[key]:del groups[key]
            if len(selected)==n:break
    return selected


def run(project, limit=24):
    if project['data'].get('_study_kind')!='h3':raise AuditError('Load a current H3 project first.')
    if project.get('parent_run_id') or project.get('completion_ledger'):
        raise AuditError('Use an original input version for evaluation, not completed/derived records.')
    if not 4<=limit<=48:raise AuditError('Use 4–48 observed masking cases.')
    started=time.perf_counter();rows=inventory(project);memory=freeze_memory(project)
    counties=sorted({r['county'] for r in rows})
    if len(counties)<2:raise AuditError('At least two counties are needed for a county-disjoint pilot.')
    # Freeze the split before any output or scores. This is a development pilot,
    # not a pristine confirmatory test set after researchers inspect these outputs.
    dev=set(counties[::2]);test=set(counties[1::2])
    for r in rows:r['split']='development' if r['county'] in dev else 'heldout_county_pilot'
    eligible=[r for r in rows if r['eligibility']=='observed_applicable']
    cases=_sample([r for r in eligible if r['county'] in dev],limit//2)+_sample([r for r in eligible if r['county'] in test],limit-limit//2)
    if not cases:raise AuditError('No observed, branch-applicable binary start statuses are available.')
    # No values are sent to either method. Definitions cannot establish a county state.
    predictions=[];retrievals=[]
    for r in cases:
        query=f"{r['menu']} implementation started strategy {r['source_field']}"
        docs=[d for d in memory['documents'] if str(d.get('version'))==r['version']]
        t=time.perf_counter();result=retrieve(query,docs,'bm25',top_k=3)
        elapsed=time.perf_counter()-t
        retrievals.append({'case_id':r['case_id'],'query':query,'backend':'bm25','status':result['status'],
            'memory_hash':memory['memory_hash'],'citations':result['citations'],'seconds':elapsed})
        for method in ['rules_only','bm25_source_memory']:
            # Rules use an exact version/field dictionary lookup. BM25 is a
            # broader lexical retrieval. Neither source type asserts a status.
            found=([d for d in docs if d.get('field')==r['source_field']] if method=='rules_only' else result['citations'])
            state_evidence=[d for d in found if all(str(d.get(k,''))==str(r[k]) for k in ['county','month','menu','slot','version']) and d.get('memory_role')=='independent_state_evidence']
            if state_evidence:raise AuditError('Independent state sources need a separate provenance-reviewed protocol; this pilot admits definitions only.')
            predictions.append({'case_id':r['case_id'],'split':r['split'],'method':method,'proposed_value':None,
                'status':'abstain_no_independent_county_time_state_evidence',
                'reason':'Field definitions specify meaning and branching, not the actual county-month status.',
                'retrieved_ids':';'.join(d['chunk_id'] for d in found),
                'target_definition_retrieved':any(d.get('field')==r['source_field'] for d in found),
                'memory_hash':memory['memory_hash'] if method!='rules_only' else '',
                'method_seconds':elapsed if method!='rules_only' else None})
    metrics=[]
    for split in ['development','heldout_county_pilot']:
        for method in ['rules_only','bm25_source_memory']:
            pred=[p for p in predictions if p['split']==split and p['method']==method];n=len(pred)
            metrics.append({'split':split,'method':method,'masked_cases':n,'proposed':0,'correct':0,'incorrect':0,'unknown':n,
                'correct_per_masked_case':0.0 if n else None,'incorrect_per_masked_case':0.0 if n else None,
                'unknown_fraction':1.0 if n else None,'accuracy_among_proposed':None,
                'reference_kind':'Observed source value, not independent human adjudication','human_review_seconds':None})
    root=Path(__file__).resolve().parents[1]
    code={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['orcca_audit/pilot.py','orcca_audit/retrieval.py','rag_system.py']}
    return {'schema_version':1,'project_run_id':project['run_id'],'project_fingerprint':project['input_fingerprint'],
        'execution':{'python':platform.python_version(),'packages':{p:version(p) for p in ['pandas','numpy','rank-bm25']},
            'code_sha256':code,'protocol_version':'source-definitions-feasibility-v1'},
        'classification':project['classification'],'manifest':deepcopy(project['manifest']),
        'protocol':'Slot-1 started status only. Exact version dictionary, supported flat binary branching. Whole-county development/heldout split. Target values and all raw Tracker snapshots excluded from retrieval. No automated value inference from a definition.',
        'inventory':rows,'masked_cases':cases,'predictions':predictions,'retrievals':retrievals,'memory':memory,'metrics':metrics,
        'summary':{'inventory_cells':len(rows),'eligibility_counts':dict(Counter(r['eligibility'] for r in rows)),
            'masking_cases':len(cases),'recorded_class_counts':dict(Counter(str(r['recorded_value']) for r in cases)),
            'development_counties':sorted(dev),'heldout_counties':sorted(test),'actual_bm25_calls':len(retrievals),
            'human_annotations':0,'external_llm_calls':0,'hybrid_run':False,'actual_missing_values_completed':0,
            'memory_updates_from_test_answers':0,'software_seconds':time.perf_counter()-started,
            'accuracy_improvement_established':False},
        'limitations':['Abstention is expected without independent county/time state evidence; retrieval is not numerical imputation.',
            'Slot-1 screening is not an inventory of all strategies. Raw blanks may be skip, no update, or unresolved reporting.',
            'Recorded values can be wrong. Masking observed cells does not represent the mechanism of real missingness.',
            'No independent human labels or review-time measurements; no mediation or effect model. No RAG superiority claim.']}


def exports(result):
    """Blind human packages separate from system output and sealed references."""
    templates=[]
    selected=list(result['masked_cases'])
    # Include true unknowns and skip/uncertain cases in the annotation pilot.
    selected+=_sample([r for r in result['inventory'] if r['eligibility']!='observed_applicable'],max(0,28-len(selected)))
    for r in selected:
        docs=[d for d in result['memory']['documents'] if d.get('field')==r['source_field'] and str(d.get('version'))==r['version']]
        templates.append({k:r[k] for k in ['case_id','county','month','menu','version','source_file','source_sheet','source_row','source_field','question']})
        templates[-1].update({'source_definition':'\n\n'.join(d['text'] for d in docs),
            'dictionary_locator':'; '.join(f"{d['source_file']} / {d.get('locator',d['line_start'])}" for d in docs),
            'human_answer':'','answerability':'','missingness_type':'','reason':'','annotator':'','elapsed_seconds':''})
    comparison=[]
    for p in result['predictions']:
        r=next(r for r in result['masked_cases'] if r['case_id']==p['case_id'])
        comparison.append({**p,'county':r['county'],'month':r['month'],'source_field':r['source_field'],
            'recorded_reference':r['recorded_value'],'input_value_after_mask':'','derived_candidate':p['proposed_value'],
            'approved_value':'','review_required':True})
    files={'protocol.json':json.dumps({k:v for k,v in result.items() if k not in {'inventory','masked_cases','predictions','retrievals','memory'}},indent=2,default=str),
        'field_review.csv':safe_csv(result['inventory']),
        'A_annotator_1.csv':safe_csv(templates),'A_annotator_2.csv':safe_csv(templates),
        'B_system_comparison.csv':safe_csv(comparison),
        'masked_input_cells.csv':safe_csv([{**{k:v for k,v in r.items() if k!='recorded_value'},'input_value':None} for r in result['masked_cases']]),
        'derived_candidates.csv':safe_csv([{k:v for k,v in p.items() if k!='recorded_reference'} for p in comparison]),
        'sealed_recorded_references.csv':safe_csv(result['masked_cases']),
        'metrics.csv':safe_csv(result['metrics']),
        'source_memory.json':json.dumps(result['memory'],indent=2,default=str),
        'retrieval_log.json':json.dumps(result['retrievals'],indent=2,default=str),
        'README.txt':INSTRUCTIONS}
    out=BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in files.items():z.writestr(name,content)
    return out.getvalue()


INSTRUCTIONS='''ORCCA completion pilot — development only
Give each annotator only their A_annotator file and authorized original sources.
Do not share B_system_comparison, sealed references, metrics or the full ZIP until both independent annotation files are locked.
Human answer: 0, 1, or unknown. Answerability: answerable, conflicting, insufficient.
Missingness type: observed, structural_skip, no_update, unreported, unresolved.
Use the source locator to inspect the original record, branching fields and dictionary.
Record the exact evidence and your reason. Leave unknown when evidence is insufficient.
Measure active review seconds; do not substitute software runtime for human time.
Use a third reviewer or documented consensus to resolve disagreement after independent submission.
Masked values are observed source references, not validated human truth. This pilot is not a final test set.
The source-only memory is frozen to this input fingerprint. Test answers, model outputs and review decisions must not enter it.
No original files were modified. No independent state evidence is currently admitted by this first pilot; both methods should abstain on values.
Before evaluating numerical recovery, obtain independent source records and verify county, applicable time, strategy identity and version without using target/derived values.
'''
