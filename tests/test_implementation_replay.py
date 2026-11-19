import pandas as pd
import pytest

from orcca_audit.implementation_replay import first_link, SearchIndex


def test_previous_month_and_cluster_inference():
    rows = []
    for c in range(5):
        for m in range(2, 9):
            a = (m*c + m*m) % 13
            rows.append(dict(input_version='original', menu='demo', county=str(c),
                             mediator_month=f'2024-{m:02}', attendance_month=f'2024-{m-1:02}',
                             attendance=a, active_snapshot=c + m + a/5))
    frame = pd.DataFrame(rows)
    assert first_link(frame).iloc[0].coefficient == pytest.approx(2.0)
    frame.loc[0, 'attendance_month'] = frame.loc[0, 'mediator_month']
    assert 'precede' in first_link(frame).iloc[0].reason


def test_empty_scope_never_returns_another_county():
    assert SearchIndex([]).search('meeting', 5) == []
