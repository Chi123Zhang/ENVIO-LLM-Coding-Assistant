"""Opt-in, identity-provider authenticated access to cloud research uploads.

Default is closed. An upload checkbox, email text input or session-state flag
cannot grant access. No credentials, study files or identity claims are persisted.
"""
import math
import time


def authorized_claims(settings, claims, logged_in, now=None):
    if settings.get('enabled') is not True or logged_in is not True:
        return False
    emails = settings.get('allowed_emails', [])
    if not isinstance(emails, (list, tuple)) or not emails:
        return False
    issuer = settings.get('issuer', '')
    if not issuer or claims.get('iss') != issuer or not claims.get('sub'):
        return False
    if claims.get('email_verified') is not True:
        return False
    email = claims.get('email', '')
    if not isinstance(email, str) or email.casefold() not in {
            e.casefold() for e in emails if isinstance(e, str)}:
        return False
    expiry = claims.get('exp')
    if isinstance(expiry, bool) or not isinstance(expiry, (int, float)):
        return False
    return math.isfinite(expiry) and expiry > (time.time() if now is None else now)


def cloud_settings():
    import streamlit as st
    try:
        return dict(st.secrets.get('orcca_research', {}))
    except (FileNotFoundError, KeyError, RuntimeError):
        return {}


def cloud_research_allowed():
    import streamlit as st
    settings = cloud_settings()
    if settings.get('enabled') is not True:
        return False
    user = getattr(st, 'user', None)
    if user is None or not getattr(user, 'is_logged_in', False):
        return False
    return authorized_claims(settings, dict(user), user.is_logged_in)


def render_access():
    import streamlit as st
    settings = cloud_settings()
    with st.sidebar.expander('Real research access'):
        if settings.get('enabled') is not True:
            st.caption('Research uploads are closed. An operator must configure '
                       'authenticated research access before real files can be used here.')
            return
        if not hasattr(st, 'login'):
            st.error('Research login requires a Streamlit version with native OIDC support.')
            return
        st.caption('Only approved, signed-in researchers can upload study files. '
                   'The public demonstration remains synthetic. Research text is not sent to an external LLM.')
        if cloud_research_allowed():
            st.success('Research access active for this signed-in session.')
        else:
            st.info('Research access requires a verified allowlisted identity and an unexpired identity token.')
        if st.button('Sign in for research access', key='research_access_login'):
            st.login()
        if getattr(getattr(st, 'user', None), 'is_logged_in', False):
            if st.button('Sign out and clear this session', key='research_access_logout'):
                st.session_state.clear()
                st.logout()
