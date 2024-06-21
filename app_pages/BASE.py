import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd
import os

from app_utils.base_comparison import pipeline
from app_utils.general import merge_final, df_to_structured_html, save_files_in_temp_dir


def main():
    st.title("Base Approach")
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
            with st.spinner("Comparing documents... It may take a while... (up to 30 minutes)"):
                result = pipeline(doc1, doc2, api_key, llm_model, save_intermediate_steps=True)
                st.session_state["base_result"] = result

        if 'base_result' in st.session_state:
            result = st.session_state["base_result"]
            df_data, missed_ids1 = merge_final(result)
            df = pd.DataFrame(df_data)
            text_md = df_to_structured_html(df)
            st.markdown(text_md, unsafe_allow_html=True)

            # Show the result
            st.write(df)

    with col2:
        st.markdown("""
        ### This is the base approach for comparing two documents.
        The documents are compared using the LLM model from OpenAI.
        - The documents are split into pages and the LLM model is used to summarize the pages.
        - The summaries are then compared using the LLM model to generate a final comparison result.
        - The final result is displayed as a table showing the comparison between the two documents.
                         """)

        if 'result' in st.session_state:
            # try to open the intermediate steps
            try:
                st.write("### Intermediate steps:")
                for i in range(1, 4):
                    with open(f"temp/base_step_{i}.txt", "r", encoding="utf-8") as f:
                        st.write(f.read())
            except FileNotFoundError:
                st.write("Intermediate steps not found")
