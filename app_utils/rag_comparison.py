from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import pandas as pd
import os

from app_utils.general import (
    ask_openai,
    get_embedding,
    merge_jsons,
    split_document_to_pages,
    save_step_in_temp_dir,
    save_step_df_in_temp_dir
)

COMPARE_PROMPT_1 = """
You are a legal assistant. Your task is to analyze two documents excerpts and identify key differences.
If one document contains some information and another does not, mention it in the output, e.g.: "Document 2": "no information"
Focus on clauses, conditions, obligations, restrictions, dates, and other relevant details.
Mention only KEY differences from the legal assistant perspective. If there are no such KEY differences, mention it in the output, e.g.: "Legal Assistant comment" : "no KEY differences"

Return following structure in the JSON format:
{"key_difference1": {"Document 1": "...", "Document 2": "...", "Legal Assistant comment" : "..."},
"key_difference2": {"Document 1": "...", "Document 2": "...", "Legal Assistant comment" : "..."},
...
}

Return only JSON string! Do not return anything else.

DOCUMENT 1 SUMMARY:
{chunk1_1}

DOCUMENT 2 SUMMARY:
{chunk2_1}

{chunk2_2}

Helpful Answer:"""

COMPARE_PROMPT_2 = """
You are a legal assistant. Your task is to analyze two documents excerpts and identify key differences.
Focus on clauses, conditions, obligations, restrictions, dates, and other relevant details.
Mention only KEY differences from the legal assistant perspective. 
Feel free to ignore irrelevant differences, such as broader lessons or norms that can be applied across cases.

Return following structure in the JSON format:
{"key_difference1": {"Document 1": "...", "Document 2": "...", "Legal Assistant comment" : "..."},
"key_difference2": {"Document 1": "...", "Document 2": "...", "Legal Assistant comment" : "..."},
...
}

Return only JSON string! Do not return anything else.

DOCUMENT 1 EXCERPTS:
{doc1}

DOCUMENT 2 EXCERPTS:
{doc2}

ADDITIONAL_INFO:
{additional_info}

Helpful Answer:"""


# def initialize_openai(api_key):
#     import openai
#     openai.api_key = api_key


def build_faiss_index(pages):
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    return faiss_index


def compare_chunks(pages1, faiss_index, compare_prompt1, model_name="gpt-3.5-turbo"):
    comparisons = []
    similarities = []
    for page in tqdm(pages1):
        chunk = page.page_content
        chunk_embedding = get_embedding(chunk)
        similar_chunks = faiss_index.similarity_search_with_score_by_vector(chunk_embedding, k=2)
        chunk2_1, score1 = similar_chunks[0][0].page_content, similar_chunks[0][1]
        chunk2_2, score2 = similar_chunks[1][0].page_content, similar_chunks[1][1]
        formatted_compare_prompt = compare_prompt1.replace("{chunk1_1}", chunk). \
            replace("{chunk2_1}", chunk2_1). \
            replace("{chunk2_2}", chunk2_2)
        final_result = ask_openai(formatted_compare_prompt, model_name=model_name, max_tokens=1000, verbose=False)
        comparisons.append(final_result)
        similarities.append(score1)

    return comparisons, similarities


def calculate_similarities(pages1, faiss_index, threshold=0.5):
    similarities = []
    comparisons = []
    for page in tqdm(pages1):
        chunk = page.page_content
        chunk_embedding = get_embedding(chunk)
        similar_chunks = faiss_index.similarity_search_with_score_by_vector(chunk_embedding, k=4)
        similar_chunks = [x for x in similar_chunks if x[1] < threshold]
        closest_score = similar_chunks[0][1]
        closest_chunk = similar_chunks[0][0].page_content
        num_similar_chunks = len(similar_chunks)
        similarity_rank = closest_score + (1 - num_similar_chunks / 4)
        similarities.append(similarity_rank)
        comparisons.append((chunk, closest_chunk))

    return comparisons, similarities


def create_df_from_comparison(comparisons):
    merged_comparisons, missed_ids = merge_jsons(comparisons)
    additional_info = "\n".join(str(comparisons[i]) for i in missed_ids)
    df = pd.DataFrame(merged_comparisons)
    return df, additional_info


def convert_column_to_text(df, column_name, llm_placeholder='no information'):
    df_doc = "\n".join(df.loc[df[column_name].apply(
        lambda x: str(x).lower() != llm_placeholder.lower()
    ), column_name].tolist())
    return df_doc


def fill_compare_prompt_1(chunk1_1, chunk2_1, chunk2_2):
    return COMPARE_PROMPT_1.replace("{chunk1_1}", chunk1_1)\
        .replace("{chunk2_1}", chunk2_1)\
        .replace("{chunk2_2}", chunk2_2)


def fill_compare_prompt_2(doc1, doc2, additional_info):
    return COMPARE_PROMPT_2.replace("{doc1}", doc1)\
        .replace("{doc2}", doc2)\
        .replace("{additional_info}", additional_info)


def pipeline(doc1_path, doc2_path, api_key, model_name="gpt-3.5-turbo", save_intermediate_steps=False):
    # set environment variable
    os.environ["OPENAI_API_KEY"] = api_key
    pages1 = split_document_to_pages(doc1_path)
    pages2 = split_document_to_pages(doc2_path)
    faiss_index = build_faiss_index(pages2)
    comparisons, similarities = compare_chunks(pages1, faiss_index, COMPARE_PROMPT_1, model_name)
    df, additional_info = create_df_from_comparison(comparisons)
    save_step_df_in_temp_dir(df, 1, 'rag_step') if save_intermediate_steps else None
    doc1 = convert_column_to_text(df, "Document 1")
    doc2 = convert_column_to_text(df, "Document 2")
    compare_prompt = fill_compare_prompt_2(doc1, doc2, additional_info)
    final_model_name = "gpt-4o" if model_name == "gpt-3.5-turbo" else model_name
    final_result = ask_openai(compare_prompt, model_name=final_model_name, max_tokens=4000, verbose=False)
    save_step_in_temp_dir(final_result, 2, 'rag_step') if save_intermediate_steps else None
    return final_result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    doc1_path = "../examples/doc1.docx"
    doc2_path = "../examples/doc2.docx"
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-3.5-turbo"

    final_result = pipeline(doc1_path, doc2_path, api_key, model_name)
    print(final_result)
