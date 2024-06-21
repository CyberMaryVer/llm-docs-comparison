import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd
import os

from app_utils.fast_comparison import pipeline
from app_utils.general import plot_horizontal_bar_with_strips, save_files_in_temp_dir


def main():
    st.title("Fast Comparison")
    doc1 = None
    doc2 = None

    col1, col2 = st.columns((4, 2))

    with col1:
        files = st.file_uploader("Upload two documents", type=["docx", ], accept_multiple_files=True)
        api_key = st.text_input("Enter your OpenAI API key")
        llm_model = st.selectbox("Select LLM model", ["gpt-4o", "gpt-3.5-turbo"])

        if files:
            # Check if two files are uploaded
            if len(files) != 2:
                # Show select two files message
                st.warning("Please select two files to compare")
            else:
                doc1, doc2 = save_files_in_temp_dir(files)
                st.session_state["files"] = [doc1, doc2]

        compare_button = ui.button(text="Compare",
                                   variant="primary",
                                   class_name="bg-indigo-600 hover:bg-cyan-600",
                                   key="compare_button")
        if 'files' in st.session_state and compare_button:
            with st.spinner("Comparing documents... It may take a while... (up to 5 minutes)"):
                result = pipeline(doc1, doc2, api_key, llm_model, save_intermediate_steps=True)
                st.session_state["scan_result"] = result

        if 'scan_result' in st.session_state:
            result = st.session_state["scan_result"]
            comparison, similarities = result
            fig = plot_horizontal_bar_with_strips(similarities)
            st.pyplot(fig)

            page = st.slider("Select a page to view comparison", 1, len(comparison) // 4, 1)

            list_for_iterating = list(zip(comparison, similarities))
            for txt_blocks, txt_rank in list_for_iterating[(page - 1) * 4:page * 4]:
                bg_color = f"RGB({int(255 - 255 * txt_rank)}, {int(255 - 255 * txt_rank)}, 255)"
                col1t, col2t = st.columns((1, 1))
                with col1t:
                    st.markdown(format_text_block(txt_rank, txt_blocks[0], bg_color), unsafe_allow_html=True)
                with col2t:
                    st.markdown(format_text_block(txt_rank, txt_blocks[1], bg_color), unsafe_allow_html=True)


def format_text_block(txt_rank, txt_block, bg_color):
    html = f'''
    <div style="background-color:{bg_color}; padding:10px; margin:10px;">
    {txt_rank}<br>{txt_block}
    </div>'''
    return html
