from io import BytesIO
import json
from zipfile import ZipFile

import pytest

from orcca_audit.site_path import compare


def pack(**entries):
    buf = BytesIO()
    with ZipFile(buf, 'w') as z:
        for name, content in entries.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_only_complete_groups_with_matching_source_versions():
    panel = 'county,menu,outcome_month,mediator_month,attendance,outcome_available\nX,OEND,2021-03,2021-02,12,True\n'
    source = pack(**{'county_month.csv': panel, 'run_metadata.json': json.dumps({
        'inputs': [dict(role='tracker', sha256='t'), dict(role='dictionary', sha256='d')]})})
    def site_zip(tracker_hash):
        return pack(**{'summary.json': json.dumps(dict(tracker_sha256=tracker_hash, dictionary_sha256='d')),
                       'site_review.csv': 'cell,county,month,menu,observed,derived_site_count\nA1,X,2021-02,OEND,0,0\nB1,X,2021-02,OEND,,4\n'})
    with pytest.raises(ValueError, match='different input'):
        compare(source, site_zip('other'))
    checks, paths, archive = compare(source, site_zip('t'))
    assert checks[checks.menu == 'OEND'].county_months.tolist() == [0, 1]
    assert paths.empty
    with ZipFile(BytesIO(archive)) as z:
        assert b'sites_per_slot' in z.read('model_rows.csv')
        assert b'county_months' in z.read('path_results.csv')
