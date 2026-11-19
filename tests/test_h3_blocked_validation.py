from hashlib import sha256
import pandas as pd

from orcca_audit.h3_attendance_validation import blocked_validation, predictions
from orcca_audit.h3_evidence_memory import cached_search


def observed_fixture():
    months = pd.period_range('2020-01', '2022-12', freq='M').astype(str)
    return pd.DataFrame([dict(county=c, attendance_month=m, attendance=10 + i % 7)
        for c in ['Demo A', 'Demo B'] for i, m in enumerate(months)])


def test_outer_test_answers_do_not_affect_method_selection():
    observed = observed_fixture()
    predictions_a, scores_a, protocol_a = blocked_validation(observed)
    held = predictions_a[predictions_a.split.eq('held_out_test')]
    changed = observed.copy()
    test_keys = set(zip(held.county, held.month))
    mask = [key in test_keys for key in zip(changed.county, changed.attendance_month)]
    changed.loc[mask, 'attendance'] += 1000
    predictions_b, scores_b, protocol_b = blocked_validation(changed)
    pd.testing.assert_frame_equal(scores_a[scores_a.split.eq('method_selection')],
                                  scores_b[scores_b.split.eq('method_selection')])
    pd.testing.assert_frame_equal(predictions_a.drop(columns='observed'),
                                  predictions_b.drop(columns='observed'))
    assert protocol_a['selected_method'] == protocol_b['selected_method']
    assert not scores_a[scores_a.split.eq('held_out_test')].equals(
        scores_b[scores_b.split.eq('held_out_test')])


def test_entire_target_block_and_outer_tests_are_excluded_from_donors():
    frame, scores, _ = blocked_validation(observed_fixture())
    for row in frame.itertuples():
        for month in filter(None, row.donor_months.split(' | ')):
            block = pd.Period(month, freq='M').ordinal // 2
            assert block % 5 != 0
            assert block != row.block
    assert scores.groupby('split').evaluated_rows.nunique().eq(1).all()
    assert scores.groupby('split').candidate_rows.nunique().eq(1).all()


def test_weighted_donors_use_only_same_county_nearby_records():
    obs = pd.Series([100., 10., 10., 999.], index=['2021-01','2021-05','2021-07','2022-10'])
    result = predictions(obs, '2021-06')
    assert result['local_donors'] == 40
    assert result['time_weighted_donors'] < 20
    assert result['interpolation'] == 10
    _, metrics, protocol = blocked_validation(observed_fixture().iloc[:1])
    assert protocol['selection_status'] == 'predeclared_fallback_no_validation'
    assert metrics.mae.isna().all()


def document():
    text = 'The coalition attendance was twelve people.'
    return dict(chunk_id='source:p1:w0', source_file='source.txt', text=text,
        sha256=sha256(text.encode()).hexdigest(), line_start=1,
        role='original_source', county='Demo A', document_date='2021-01-01')


def test_cache_invalidates_changed_provenance_and_cannot_be_mutated_by_caller():
    cache = {}
    doc = document()
    first = cached_search('attendance', [doc], cache)
    first['citations'][0]['county'] = 'Incorrect caller mutation'
    repeat = cached_search('attendance', [doc], cache)
    assert repeat['memory_reused'] and repeat['citations'][0]['county'] == 'Demo A'
    changed = dict(doc, county='Demo B', document_date='2022-02-02')
    next_result = cached_search('attendance', [changed], cache)
    assert not next_result['memory_reused']
    assert next_result['citations'][0]['county'] == 'Demo B'


def test_transient_model_failure_is_not_cached(monkeypatch):
    from orcca_audit import hybrid
    calls = []
    def rank(*args):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('Synthetic temporary failure')
        return [(args[1][0], .8)], False
    monkeypatch.setattr(hybrid, 'rrf_retrieve', rank)
    cache = {}
    assert cached_search('attendance', [document()], cache, backend='hybrid_rrf')['status'] == 'hybrid_unavailable_no_fallback'
    result = cached_search('attendance', [document()], cache, backend='hybrid_rrf')
    assert len(calls) == 2 and result['citations']


def test_empty_scopes_do_not_evict_useful_memory():
    cache = {}
    cached_search('attendance', [document()], cache)
    for i in range(150):
        cached_search(str(i), [], cache)
    assert cached_search('attendance', [document()], cache)['memory_reused']
