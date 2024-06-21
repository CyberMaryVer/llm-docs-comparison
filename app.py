from st_on_hover_tabs import on_hover_tabs
import streamlit as st

from app_pages.BASE import main as base_page
from app_pages.RAG import main as rag_page
from app_pages.SCAN import main as scan_page
from config.constants import PAGE_CONFIG, STYLES

st.set_page_config(**PAGE_CONFIG)
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Base Approach', 'RAG Approach', 'Fast Comparison'],
                         iconName=['apps', 'polyline', 'compare'],  # https://fonts.google.com/icons
                         default_choice=0,
                         styles=STYLES,
                         key="1")

if tabs == 'Base Approach':
    base_page()

elif tabs == 'RAG Approach':
    rag_page()

elif tabs == 'Fast Comparison':
    scan_page()