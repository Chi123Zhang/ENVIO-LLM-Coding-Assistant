from orcca_audit.h3_public_comparison import full_demo_sources
from orcca_audit.site_estimates import estimate, report


def test_historical_evaluation_and_missing_cells():
    files, _ = full_demo_sources()
    stats, detail, _ = report(files['tracker'], files['dictionary'])
    assert stats['estimated'] == stats['missing'] == 1
    assert stats['metrics'][0]['paired_cells'] > 0
    assert detail[detail.month == '2024-01'].estimate.isna().all()
    for row in detail.itertuples():
        if row.donor_months:
            assert max(row.donor_months.split(' | ')) < row.month


def test_never_reuse_predictions_as_observed_donors():
    base = dict(county='A', month='2024-01', menu='X', slot=1, version='v',
                planned=3, strategy='a', sector='s', venue='b', recorded=2, target_cell='A1')
    rows = [base, dict(base, county='B', month='2024-02', recorded=None),
            dict(base, county='C', month='2024-03', recorded=None)]
    assert all(r['estimate'] is None for r in estimate(rows))
