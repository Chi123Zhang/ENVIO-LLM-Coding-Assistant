from orcca_audit.research_access import authorized_claims, cloud_research_allowed


def fixture():
    return ({'enabled': True, 'allowed_emails': ['reviewer@example.org'],
             'issuer': 'https://accounts.google.com'},
            {'iss': 'https://accounts.google.com', 'sub': 'fictional-subject',
             'email': 'reviewer@example.org', 'email_verified': True, 'exp': 2000})


def test_access_requires_verified_allowlisted_unexpired_provider_identity():
    settings, claims = fixture()
    assert authorized_claims(settings, claims, True, now=1000)
    for changed in [{'email': 'outsider@example.org'}, {'email_verified': False},
                    {'email_verified': 'true'}, {'exp': 999}, {'exp': '2000'},
                    {'exp': float('nan')}, {'exp': float('inf')}, {'sub': ''},
                    {'iss': 'https://untrusted.example.org'}]:
        assert not authorized_claims(settings, dict(claims, **changed), True, now=1000)
    for changed in [{'enabled': False}, {'enabled': 'true'}, {'allowed_emails': []},
                    {'allowed_emails': 'reviewer@example.org'}, {'issuer': ''}]:
        assert not authorized_claims(dict(settings, **changed), claims, True, now=1000)
    assert not authorized_claims(settings, claims, False, now=1000)


def test_cloud_default_is_closed_even_with_fake_session_flag(monkeypatch):
    import streamlit as st
    from orcca_audit import research_access
    monkeypatch.setattr(research_access, 'cloud_settings', lambda: {})
    st.session_state['research_authenticated'] = True
    assert not cloud_research_allowed()
    del st.session_state['research_authenticated']


def test_policy_uses_provider_gate_and_preserves_local_mode(monkeypatch):
    from orcca_audit import research_access
    from orcca_audit.policy import private_uploads_allowed
    monkeypatch.delenv('ORCCA_DEPLOYMENT_MODE', raising=False)
    monkeypatch.setattr(research_access, 'cloud_research_allowed', lambda: False)
    assert not private_uploads_allowed()
    monkeypatch.setattr(research_access, 'cloud_research_allowed', lambda: True)
    assert private_uploads_allowed()
    monkeypatch.setenv('ORCCA_DEPLOYMENT_MODE', 'local_private')
    monkeypatch.delenv('ORCCA_ALLOW_PRIVATE_UPLOADS', raising=False)
    assert not private_uploads_allowed()
    monkeypatch.setenv('ORCCA_ALLOW_PRIVATE_UPLOADS', '1')
    assert private_uploads_allowed()
