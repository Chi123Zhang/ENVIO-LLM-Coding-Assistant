"""Question-first browser controls, separate from source evidence and labels."""
import streamlit as st
from .research_plan import DEFAULT, MODES, save
from .engine import AuditError


def render():
    saved = st.session_state.get('research_plan_saved', DEFAULT)
    with st.expander('0 · Research question and data requirements', expanded=True):
        st.write('Name the question first. The workspace will inventory the supplied sources and show what still needs definition or evidence. It will not select a causal model or invent missing measures.')
        st.caption('Public practice: use a fictional question. Do not enter private study details. Planning text stays outside the evidence index and never goes to a model API.')
        with st.form('research_question_form'):
            p = {}
            p['mode'] = st.selectbox('Planning route', MODES, index=MODES.index(saved['mode']), key='question_mode')
            p['hypothesis'] = st.text_input('Hypothesis or question label', saved['hypothesis'], key='question_label')
            p['question'] = st.text_area('Research question (English)', saved['question'], key='question_text')
            p['unit'] = st.text_input('Analysis unit', saved['unit'], key='question_unit')
            a, b = st.columns(2)
            p['start_month'] = a.text_input('First applicable month (optional, YYYY-MM)', saved['start_month'], key='question_start')
            p['end_month'] = b.text_input('Last applicable month (optional, YYYY-MM)', saved['end_month'], key='question_end')
            p['exposure'] = st.text_input('Selected coalition-process / exposure measure (optional)', saved['exposure'], key='question_exposure')
            p['mediator'] = st.text_input('Selected implementation / reach measure (optional)', saved['mediator'], key='question_mediator')
            p['outcome'] = st.text_input('Selected later outcome measure (optional)', saved['outcome'], key='question_outcome')
            p['time_order'] = st.text_input('Proposed time order (optional)', saved['time_order'], key='question_order')
            p['confounders'] = st.text_input('Confounders to consider (optional)', saved['confounders'], key='question_confounders')
            p['definition_source'] = st.text_input('Protocol / definition reference (optional)', saved['definition_source'], key='question_source')
            submitted = st.form_submit_button('Save research question', key='question_save')
        if submitted:
            try:
                save(st.session_state, p)
                st.success('Question saved. Any changed question creates a new project version and clears prior finding approvals. The underlying data are preserved.')
            except AuditError as exc: st.error(str(exc))
        st.caption('Results refer to the saved question. Save edits before interpreting them. Other hypothesis labels are planning metadata; H4/H5 or custom effects are not automatically implemented.')
