import pandas as pd

from orcca_audit.h3_results_ui import statistical_summary


def test_significance_handles_threshold_and_unavailable_fits():
    paths = pd.DataFrame({'p_approx': [.049, .05, .337, None, -1, 2],
                          'menu': ['OEND'] * 6})
    result = statistical_summary(paths)
    assert result['Significant (p < 0.05)'].tolist() == [
        'Yes', 'No', 'No', 'Unavailable', 'Unavailable', 'Unavailable']
    assert result['Approx. p-value'].notna().sum() == 3
    assert paths.p_approx.iloc[0] == .049


def test_all_failed_fits_remain_visible_without_significance():
    result = statistical_summary(pd.DataFrame([
        {'menu': 'OEND', 'mediator': 'active_snapshot', 'diagnostic': 'Insufficient rows'}]))
    assert len(result) == 1
    assert result['Intermediate measure'].iloc[0] == 'Active strategies'
    assert result['Significant (p < 0.05)'].iloc[0] == 'Unavailable'
    assert pd.isna(result['Approx. p-value'].iloc[0])
