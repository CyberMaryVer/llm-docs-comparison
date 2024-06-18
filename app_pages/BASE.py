import streamlit as st
import streamlit_shadcn_ui as ui

import os

from app_utils.base_comparison import pipeline


def save_files_in_temp_dir(files):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    for i, file in enumerate(files):
        with open(os.path.join(temp_dir, f"file_{i}.docx"), "wb") as f:
            f.write(file.getvalue())
    return os.path.join(temp_dir, "file_0.docx"), os.path.join(temp_dir, "file_1.docx")


def main():
    st.title("Base Approach")
    doc1 = None
    doc2 = None

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

        # Show the result
        st.write(result)
