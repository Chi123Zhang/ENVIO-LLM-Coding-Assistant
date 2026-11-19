from orcca_audit.retrospective_sites import compare


def row(cell, month, count, reach='a'):
    return dict(county='demo', month=month, menu='x', slot=1, version='v', strategy='s',
                sector='x', venue='y', planned=0, recorded=count, reach_id=reach, target_cell=cell)


def test_masked_month_identity_zero_and_memo():
    rows = [row('A1', '2021-01', None), row('A2', '2021-02', 0),
            row('A3', '2021-03', None), row('A4', '2021-01', 8, 'b'),
            row('A5', '2024-01', 5), row('A6', '2021-02', None, '')]
    calls = []
    def infer(payload):
        calls.append(payload)
        return '0'
    stats, detail = compare(rows, infer)
    by_cell = {r['cell']: r for r in detail}
    assert calls == ['["0"]']
    assert by_cell['A1']['derived_site_count'] == 0 and by_cell['A1']['later_only']
    assert by_cell['A2']['estimate'] is None
    assert by_cell['A6']['derived_site_count'] is None
    assert stats['metrics'][-1]['missing_estimated'] == 2


def test_disagreement_and_conflict():
    rows = [row('A1', '2020-01', 1), row('A2', '2020-02', None), row('A3', '2020-03', 3)]
    _, results = compare(rows, lambda payload: 'unknown')
    assert all(r['estimate'] is None for r in results)
    assert results[1]['reason'] == 'Conflicting source values'
