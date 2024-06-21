from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
import os

from app_utils.general import (
    ask_openai,
    get_embedding,
    save_step_in_temp_dir,
    save_step_df_in_temp_dir,
    split_document_to_pages
)
from app_utils.rag_comparison import build_faiss_index, calculate_similarities


def pipeline(doc1_path, doc2_path, api_key, model_name="gpt-3.5-turbo", save_intermediate_steps=False):
    # set environment variable
    os.environ["OPENAI_API_KEY"] = api_key
    pages1 = split_document_to_pages(doc1_path, chunk_size=1000, chunk_overlap=200)
    pages2 = split_document_to_pages(doc2_path, chunk_size=1000, chunk_overlap=200)
    faiss_index = build_faiss_index(pages2)
    comparisons, similarities = calculate_similarities(pages1, faiss_index)
    return comparisons, similarities
