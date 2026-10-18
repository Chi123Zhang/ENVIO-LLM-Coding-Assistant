import streamlit as st

from orcca_audit.findings_ui import render

st.set_page_config(page_title="ORCCA Research Findings | ENVIO", layout="wide")
render()
