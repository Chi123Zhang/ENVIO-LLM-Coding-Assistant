import streamlit as st

from orcca_audit.completion_ui import render

st.set_page_config(page_title="ORCCA Evidence Completion", layout="wide")
render()
