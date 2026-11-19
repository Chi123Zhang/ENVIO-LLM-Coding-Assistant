"""Retrospective validation of delivery counts using linked Tracker reports."""
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
import re
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

from .exports import safe_csv
from .site_estimates import extract, estimate

MATCH = ('county', 'reach_id', 'menu', 'slot', 'strategy', 'sector', 'venue', 'planned', 'version')
PROMPT = 'Copy the shared integer from these linked records. If different, output unknown. Output only the integer or unknown.'


def month_number(month):
    year, number = map(int, month.split('-'))
    return 12 * year + number


def generate(payload):
    from .h3_ai_runtime import load_model, _LOCK
    import torch
    with _LOCK:
        tokenizer, model, _ = load_model()
        prompt = tokenizer.apply_chat_template([
            {'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': payload}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt')
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with torch.inference_mode():
                ids = model.generate(**inputs, do_sample=False, max_new_tokens=10,
                                     pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        finally:
            torch.set_num_threads(threads)


def compare(rows, infer=generate):
    index = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in MATCH)
        if row['recorded'] is not None and all(v not in ('', None) for v in key):
            index[key].append(row)
    cache, results = {}, []
    for row in estimate(rows):
        target = month_number(row['month'])
        matches = [(month_number(d['month']) - target, d) for d in index.get(tuple(row[k] for k in MATCH), ())]
        donors = [d for gap, d in sorted(matches, key=lambda pair: (abs(pair[0]), pair[1]['target_cell']))
                  if 0 < abs(gap) <= 12]
        values = [str(d['recorded']) for d in donors]
        consensus = candidate = None
        response = ''
        reason = 'No linked record'
        if values:
            reason = 'Conflicting source values'
            if len(set(values)) == 1:
                consensus = int(values[0])
                payload = json.dumps(values)
                if payload not in cache:
                    cache[payload] = infer(payload)
                response = cache[payload]
                reason = 'Model abstained or disagreed'
                if re.fullmatch(r'[0-9]+(?:\.0+)?', response) and int(response.split('.')[0]) == consensus:
                    candidate = consensus
                    reason = 'Unverified retrospective estimate'
        observed = row['recorded']
        results.append(dict(cell=row['target_cell'], county=row['county'], month=row['month'],
            menu=row['menu'], reach_id=row['reach_id'], observed=observed, planned=row['planned'],
            earlier=row['estimate'], consensus=consensus, estimate=candidate,
            derived_site_count=observed if observed is not None else candidate, reason=reason,
            model_response=response, later_only=bool(donors) and all(d['month'] > row['month'] for d in donors),
            sources=' | '.join(f"{d['month']} {d['target_cell']}={d['recorded']}" for d in donors)))
    observed = [r for r in results if r['observed'] is not None]
    missing = [r for r in results if r['observed'] is None]
    metrics = []
    for method in ('earlier', 'consensus', 'estimate'):
        paired = [r for r in observed if r[method] is not None]
        metrics.append(dict(method=method, masked_answered=len(paired),
            exact=sum(r[method] == r['observed'] for r in paired),
            paired_plan_exact=sum(r['planned'] == r['observed'] for r in paired),
            missing_estimated=sum(r[method] is not None for r in missing)))
    return dict(observed=len(observed), missing=len(missing), calls=len(cache), metrics=metrics,
                later_only=sum(r['later_only'] and r['estimate'] is not None for r in missing)), results


def report(tracker, dictionary):
    from .h3_ai_runtime import MODEL_ID, MODEL_REVISION
    summary, rows = compare(extract(tracker, dictionary, include_reach_id=True))
    summary.update(model=MODEL_ID, revision=MODEL_REVISION, tracker_sha256=sha256(tracker[1]).hexdigest(),
                   dictionary_sha256=sha256(dictionary[1]).hexdigest())
    archive = BytesIO()
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        z.writestr('summary.json', json.dumps(summary, indent=2))
        z.writestr('site_review.csv', safe_csv(pd.DataFrame(rows)))
        z.writestr('README.txt', 'Earlier uses prior reports only. Consensus and estimate use linked records within 12 months on either side, excluding the target month. Estimate requires the on-server model to reproduce the consensus. Sources refer to orcca_tracker_v2 worksheet cells. Original observations are preserved. Missing-cell estimates are unconfirmed and excluded from H3 models.')
    return summary, archive.getvalue()


def render(files, fingerprint):
    import streamlit as st
    key = 'site_retrospective'
    previous = st.session_state.get(key)
    if previous and previous[0] != fingerprint:
        del st.session_state[key]
    with st.expander('Retrospective retrieval for missing delivery-site counts'):
        st.write('Compare the earlier-only method with linked reports up to 12 months before or after each target. The server runs Qwen on numeric source counts. Estimates stay outside H3 models.')
        if st.button('Run retrospective site completion', disabled=not {'tracker', 'dictionary'} <= files.keys()):
            st.session_state.pop(key, None)
            try:
                with st.spinner('Linking reports and checking counts with Qwen…'):
                    st.session_state[key] = (fingerprint, report(files['tracker'], files['dictionary']))
            except Exception as error:
                st.error(f'Unable to complete the comparison ({type(error).__name__}). Check input files and server model availability.')
        if key in st.session_state:
            summary, data = st.session_state[key][1]
            st.write(f"{summary['observed']} recorded counts held out by month · {summary['missing']} missing · "
                     f"{summary['calls']} model calls · {summary['later_only']} estimates rely only on later reports")
            st.table(pd.DataFrame(summary['metrics']).replace({'method': {
                'earlier': 'Earlier reports', 'consensus': 'Linked reports', 'estimate': 'Linked reports + Qwen'}}))
            st.download_button('Download retrospective site report', data, 'retrospective_sites.zip', 'application/zip')
