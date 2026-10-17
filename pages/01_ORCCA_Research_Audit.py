"""Additive Streamlit page; the original app.py remains the main entrypoint."""
import streamlit as st
from orcca_audit.ui import render

st.set_page_config(page_title="ORCCA Research Audit | ENVIO", layout="wide")
render()
