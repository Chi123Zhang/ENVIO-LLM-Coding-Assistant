"""Tracker-first H3 preparation. No mediation model or causal effect estimation."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import re

import pandas as pd

from .engine import AuditError, as_month, empty, require
from .adapters import tracker_workbooks_to_tidy, event_month
from .io import read_table, read_workbook, normalize_mortality, extract_document
from .workflow import start_project, _receipt
from .findings import validate_bundle


def is_study(project):return bool(project and project['data'].get('_study_kind')=='h3')


def counties(project):
    if is_study(project):return sorted(set(project['data']['study_records'].county))
    return sorted(set(project['data']['panel'].county.astype(str)))


def audit(data):
    raw,dd,records=data['raw_tracker'],data.get('dictionary'),data['study_records']
    definitions={}
    if dd is not None:
        require(dd,['Version','Variable / Field Name','Field Label'],'Dictionary')
        if dd.duplicated(['Version','Variable / Field Name']).any():raise AuditError('Duplicate dictionary version-field keys.')
        definitions={(str(x['Version']),str(x['Variable / Field Name'])):x for x in dd.to_dict('records')}
    profiles=[]
    # Initial completion semantics remain OEND/MOUD; all raw columns remain counted.
    for version in sorted(raw.hcs_orc2version.astype(str).unique()):
        vr=raw[raw.hcs_orc2version.astype(str)==version]
        for field in raw.columns:
            if not re.fullmatch(r'hcs_orc2(?:09|10)[a-z]*_1(?:_(?:mo|yr))?',str(field)):continue
            definition=definitions.get((version,field),{})
            profiles.append({'version':version,'field':field,'definition':definition.get('Field Label','Definition not supplied for this version'),
                'raw_nonblank':int(sum(not empty(x) for x in vr[field])),'version_records':len(vr),
                'definition_found':bool(definition),'dictionary_row':definition.get('_source_row',''),
                'source_file':dd.attrs.get('source_file','dictionary') if definition else '',
                'branching_evaluated':False})
    process=data.get('coalition_process')
    measures=sorted(process.measure.unique()) if process is not None else []
    structure={'records':len(records),'counties':records.county.nunique(),'event_months':records.month.nunique(),
        'first_event_month':records.month.min(),'last_event_month':records.month.max(),
        'questionnaire_versions':sorted(raw.hcs_orc2version.astype(str).unique()),
        'raw_columns':len([c for c in raw if c not in {'_source_row','source_file','source_sheet'}]),
        'county_month_key_unique':not records.duplicated(['county','month']).any()}
    readiness=[{'path_component':'Coalition process (X)','availability':'Supplied; measure definition needs review' if measures else 'Not supplied',
        'next_step':'Confirm one exposure, source, applicable time and confounders. Attendance is auxiliary unless explicitly selected as the exposure.'},
        {'path_component':'EBP implementation or reach (M)','availability':'Candidate Tracker fields available',
        'next_step':'Resolve versions, branching and strategy identity. Strategy counts, delivery sites and actual reach are different measures.'},
        {'path_component':'Later overdose outcome (Y)','availability':'Mortality source supplied; definition and timing need review' if 'monthly' in data else 'Not supplied',
        'next_step':'Confirm outcome query definition, follow-up, population and suppression. Do not reconstruct hidden deaths.'},
        {'path_component':'Temporal order and identification','availability':'Not established',
        'next_step':'Specify X before M before Y and confounders. Repeated county records are not independent counties.'}]
    result={'structure':structure,'field_review':profiles,'h3_readiness':readiness,
        'scope':{'hypothesis':'H3 only','mediation_estimated':False,'external_llm_calls':0,'h4_h5':'Deferred',
        'coalition_measures_supplied':measures,'completion_status':data.get('completion_status','Unavailable')}}
    if 'tracker' in data:
        from .completion import identify_issues
        _,issues=identify_issues(data['tracker'])
        result['completion_overview']={'unknown_fields':len(issues),'eligible_for_source_review':sum(x['eligible'] for x in issues),
            'basis':'Derived status records; blanks are candidates, not confirmed errors.'}
    return result


def parse_inputs(supplied, evidence_files):
    if supplied.get('raw_tracker') is None:raise AuditError('Add the original Tracker workbook. No panel is required for H3 preparation.')
    f=supplied['raw_tracker'];sheets=read_workbook(f.getvalue(),positional_sheets=('Sheet1',))
    if not {'orcca_tracker_v2','Sheet1'}<=set(sheets):raise AuditError('Tracker needs orcca_tracker_v2 and the explicit Sheet1 county lookup.')
    raw=sheets['orcca_tracker_v2'];mapping=sheets['Sheet1']
    require(raw,['hcs_orc2record_id','redcap_event_name','hcs_orc2version'],'Tracker')
    if mapping.iloc[:,0].map(empty).any() or mapping.iloc[:,0].duplicated().any():raise AuditError('County lookup IDs must be unique and nonempty.')
    lookup=dict(zip(mapping.iloc[:,0],mapping.iloc[:,1]))
    rows=[]
    for x in raw.to_dict('records'):
        county=lookup.get(x['hcs_orc2record_id'])
        if empty(county):raise AuditError('A Tracker record has no explicit county lookup.')
        rows.append({'county':str(county),'month':event_month(x['redcap_event_name']),
            'version':str(x['hcs_orc2version']),'source_file':f.name,'source_row':x['_source_row']})
    records=pd.DataFrame(rows)
    if records.duplicated(['county','month']).any():raise AuditError('Resolve duplicate county-month Tracker records first.')
    data={'_study_kind':'h3','raw_tracker':raw,'study_records':records,'completion_status':'Add the dictionary before status completion.'}
    documents=[]
    if supplied.get('dictionary') is not None:
        d=supplied['dictionary'];dd=read_table(d.getvalue(),d.name,sheet='ORCCAT_DD');data['dictionary']=dd
        require(dd,['Version','Variable / Field Name','Field Label'],'Dictionary')
        # Only original field definitions enter retrieval, not hypotheses/review answers.
        for x in dd.to_dict('records'):
            field=str(x['Variable / Field Name']);version=str(x['Version'])
            if field not in raw or version not in set(records.version) or not re.fullmatch(r'hcs_orc2(?:09|10)[a-z]*_1(?:_(?:mo|yr))?',field):continue
            text='\n'.join(f'{k}: {x.get(k) or ""}' for k in ['Version','Variable / Field Name','Field Label','Choices, Calculations, OR Slider Labels','Branching Logic (Show field only if...)'])
            documents.append({'chunk_id':f'dd:{version}:{field}','source_id':d.name,'source_file':d.name,
                'page':None,'line_start':int(x['_source_row']),'line_end':int(x['_source_row']),
                'locator':f"ORCCAT_DD row {x['_source_row']}; field {field}; version {version}",
                'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest(),'classification':'private_local',
                'evidence_kind':'original_excerpt','version':version,'field':field})
        try:
            tidy,_=tracker_workbooks_to_tidy(sheets,dd,f.name)
            from .completion import identify_issues
            identify_issues(tidy)
            data['tracker']=tidy;data['completion_status']='Candidate OEND/MOUD status mapping available; semantic review required.'
        except AuditError as exc:data['completion_status']='Status completion blocked: '+str(exc)
    for role in ('monthly','annual'):
        if supplied.get(role) is not None:
            file=supplied[role];data[role]=normalize_mortality(read_table(file.getvalue(),file.name))
    if supplied.get('coalition_process') is not None:
        file=supplied['coalition_process'];process=read_table(file.getvalue(),file.name)
        require(process,['county','month','measure','value','source_ref'],'Coalition process table')
        process['month']=process.month.map(as_month)
        if any(process[c].map(empty).any() for c in ['county','measure','source_ref']) or process.duplicated(['county','month','measure']).any():raise AuditError('Coalition rows need unique county-month-measure keys and source references.')
        if not set(process.county)<=set(records.county):raise AuditError('Coalition county identifiers do not match the Tracker.')
        data['coalition_process']=process
    evidence=None
    if supplied.get('completion_evidence') is not None:
        file=supplied['completion_evidence'];evidence=read_table(file.getvalue(),file.name)
        from .completion import prepare_evidence
        prepare_evidence(evidence)
    for file in evidence_files:documents.extend(extract_document(file.getvalue(),file.name))
    files=[v for v in supplied.values() if v is not None]+evidence_files
    if len({f.name for f in files})!=len(files):raise AuditError('Source filenames must be unique.')
    manifest={k:{'source_file':v.name,'sha256':hashlib.sha256(v.getvalue()).hexdigest()} for k,v in supplied.items() if v is not None}
    manifest['auxiliary_documents']=[{'source_file':f.name,'sha256':hashlib.sha256(f.getvalue()).hexdigest()} for f in evidence_files]
    return data,documents,evidence,manifest


def sample(variant='Unresolved status'):
    from .demo import demo_data
    from .workflow_ui import synthetic_project_inputs
    data=demo_data();tidy=data['tracker'].copy()
    if variant=='Changed input':tidy.loc[0,'started']=0
    records=tidy[['county','month']].drop_duplicates().copy();records['version']='SYNTHETIC_V1'
    records['source_file']='synthetic_study_tracker.csv';records['source_row']=range(2,len(records)+2)
    raw=records.rename(columns={'version':'hcs_orc2version'}).copy()
    raw['hcs_orc209l_1']=1;raw['hcs_orc209v_1']=0
    raw['hcs_orc210m_1']=1;raw['hcs_orc210v_1']=0;raw.loc[0,'hcs_orc210v_1']=None
    if variant=='Changed input':raw.loc[0,'hcs_orc209l_1']=0
    doc={'chunk_id':'synthetic-study-definition','source_id':'orcca_audit/H3_DEMO_PROTOCOL.txt','source_file':'orcca_audit/H3_DEMO_PROTOCOL.txt',
        'page':None,'line_start':1,'line_end':4,'locator':'Bundled H3_DEMO_PROTOCOL.txt, lines 1–4',
        'text':Path(__file__).with_name('H3_DEMO_PROTOCOL.txt').read_text(),
        'classification':'synthetic_public','evidence_kind':'original_excerpt'}
    doc['sha256']=hashlib.sha256(doc['text'].encode()).hexdigest()
    _,_,evidence,_=synthetic_project_inputs()
    from .exports import safe_csv
    evidence=read_table(safe_csv(evidence.drop(columns=['source_file','_source_row']).to_dict('records')),'synthetic_completion_evidence.csv')
    path=Path(__file__).with_name('H3_DEMO_DICTIONARY.csv')
    dd=read_table(path.read_bytes(),'orcca_audit/H3_DEMO_DICTIONARY.csv')
    docs=[doc]
    for x in dd.to_dict('records'):
        text='\n'.join(f'{k}: {x.get(k) or ""}' for k in ['Version','Variable / Field Name','Field Label','Choices, Calculations, OR Slider Labels'])
        docs.append({'chunk_id':'synthetic-dd:'+x['Variable / Field Name'],'source_id':'orcca_audit/H3_DEMO_DICTIONARY.csv',
            'source_file':'orcca_audit/H3_DEMO_DICTIONARY.csv','page':None,'line_start':x['_source_row'],'line_end':x['_source_row'],
            'locator':f"Synthetic dictionary CSV row {x['_source_row']}; field {x['Variable / Field Name']}",
            'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest(),'classification':'synthetic_public',
            'evidence_kind':'original_excerpt','version':'SYNTHETIC_V1','field':x['Variable / Field Name']})
    return {'_study_kind':'h3','raw_tracker':raw,'study_records':records,'tracker':tidy,'dictionary':dd,
        'completion_status':'Synthetic candidate status mapping; source review is still required.'},docs,evidence,{'synthetic_case':variant,'synthetic':True,'synthetic_tracker_file':{'source_file':'synthetic_study_tracker.csv','sha256':hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()}}


def build_findings(project):
    r=project['results'];s=r['structure'];findings=[];docs=[]
    def add(section,title,statement,limits,level,value):
        doc=_receipt(project,section,value);docs.append(doc)
        findings.append({'finding_id':'S'+str(len(findings)+1),'title':title,'statement':statement,
            'verification_method':'Current-version source inventory and deterministic checks; no fitted model.',
            'limitations':limits,'level':level,'evidence_ids':[doc['chunk_id']],'review_status':'pending'})
    add('structure','Tracker coverage',f"The Tracker contains {s['records']} records from {s['counties']} counties across {s['event_months']} event months ({s['first_event_month']} to {s['last_event_month']}).",
        'County-month rows are repeated observations, not independent counties. Entry dates and event time are different.', 'verified_source_check',s)
    missing=sum(not x['definition_found'] for x in r['field_review'])
    add('field_review','Version-specific measurement review',f"{len(r['field_review'])} representative slot-1 field-version definitions were screened; {missing} were not found in the supplied dictionary.",
        'Missing definitions can reflect instrument versions, metadata or incomplete documentation. Raw blanks are not automatically missing; branching is not evaluated.', 'method_review',r['field_review'])
    add('h3_readiness','H3 remains a research question','The proposed path is coalition process → EBP implementation or reach → later overdose outcome. Temporal ordering and causal identification remain unestablished.',
        'These materials do not establish a mediation effect. Dictionary definitions and retrieved passages cannot supply unobserved process measures or outcomes.', 'insufficient_material',r['h3_readiness'])
    for i,record in enumerate(project['data'].get('implementation_evidence',[]),1):
        findings.append({'finding_id':'I'+str(i),'title':'Linked implementation interpretation',
            'statement':'Candidate ENVIO codes: '+record['envio_codes']+'. Applicability note: '+record['applicability_reason'],
            'verification_method':'Exact quotation linked after explicit researcher applicability review; codebook validity is not established.',
            'limitations':'Not a validated H3 exposure, mediator or outcome. County/time applicability is human assigned.',
            'level':'method_review','evidence_ids':[record['source_chunk_id']],'review_status':'pending'})
    for d in project['documents']:
        d=deepcopy(d);d.setdefault('source_id',d['source_file']);d.setdefault('evidence_kind','original_excerpt');d.setdefault('line_end',d['line_start']);docs.append(d)
    sources=[{'source_id':name,'source_file':name} for name in dict.fromkeys(d['source_id'] for d in docs)]
    hypotheses=[{'hypothesis':'H3','definition':'Coalition processes influence overdose outcomes through EBP deployment and reach.',
        'available':'Tracker implementation candidates; optional supplied source tables.',
        'missing':'Validated X and M, later Y, confounders and defensible temporal ordering.',
        'next_step':'Review one prespecified pathway before estimating effects.'}]
    bundle={'schema_version':'1.0','classification':project['classification'],'review_date':datetime.fromisoformat(project['created_at_utc']).astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat(),
        'findings':findings,'documents':docs,'sources':sources,'questions':[],'hypotheses':hypotheses,
        'evaluation':{'human_labelled_cases':0,'improvement_measured':False},
        'audit_run':{'run_id':project['run_id'],'revision':project['revision'],'inputs':project['manifest'],
            'parent_run_id':project['parent_run_id'],'external_llm_calls':0,'effect_models_fitted':0}}
    validate_bundle(bundle);project['bundle']=bundle;project['finding_decisions']={};return bundle


def search(query,documents,backend='bm25',version=None):
    from .retrieval import retrieve
    docs=[d for d in documents if not version or d.get('version')==version or not d.get('version')]
    field=re.search(r'\bhcs_orc\w+\b',query)
    if field:
        exact=[d for d in docs if d.get('field')==field.group()]
        return {'status':'exact_field_definition' if exact else 'field_definition_not_found','backend':'deterministic_version_field_lookup','citations':exact}
    return retrieve(query,docs,backend)
