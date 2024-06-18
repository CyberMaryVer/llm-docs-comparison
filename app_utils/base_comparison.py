import os
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate

from app_utils.general import (
    split_document_to_pages,
    ask_openai,
    save_step_in_temp_dir
)

text_splitter = CharacterTextSplitter()

MAP_PROMPT = """
You are a legal assistant. Your task is to analyze the document and identify key points and meaningful information. 
Focus on clauses, conditions, obligations, restrictions, dates, and other relevant details. 
Provide a numbered list of key points.

DOCUMENT:

{docs}


Helpful Answer:"""

REDUCE_PROMPT = """The following is set of summaries:
{docs}
You are a legal assistant. Take these and distill it into a final, consolidated summary of the main themes. 
Focus on clauses, conditions, obligations, restrictions, dates, and other relevant details.

Helpful Answer:"""

COMPARE_PROMPT = """
You are a legal assistant. Your task is to analyze two documents summaries and identify key differences.
Focus on clauses, conditions, obligations, restrictions, dates, and other relevant details.

Return following structure in the JSON format:
{"key_difference1": {"Document 1": "...", "Document 2": "...", "Legal Assistant comment" : "..."},
"key_difference2": {"Document 1": "...", "Document 2": "...", "Legal Assistant comment" : "..."},
...
}

DOCUMENT 1 SUMMARY:
{doc1}

DOCUMENT 2 SUMMARY:
{doc2}

Helpful Answer:"""


def create_llm(api_key, model_name="gpt-4o"):
    llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key)
    return llm


def create_map_reduce_chain(llm,
                            map_template=MAP_PROMPT,
                            reduce_template=REDUCE_PROMPT):
    map_prompt = PromptTemplate.from_template(map_template)
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )
    return map_reduce_chain


def fill_compare_prompt(doc1, doc2):
    return COMPARE_PROMPT.replace("{doc1}", doc1).replace("{doc2}", doc2)


def summarize_document(doc_path, api_key, model_name="gpt-4o"):
    pages = split_document_to_pages(doc_path)
    llm = create_llm(api_key, model_name)
    map_reduce_chain = create_map_reduce_chain(llm)
    result = map_reduce_chain.invoke(pages)
    with open("temp/debug.txt", "w") as f:
        f.write(str(result))
    result = str(result)
    return result


def pipeline(doc1_path, doc2_path, api_key, model_name="gpt-4o", save_intermediate_steps=False):
    result1 = summarize_document(doc1_path, api_key, model_name)
    save_step_in_temp_dir(result1, 1, 'base_step') if save_intermediate_steps else None
    result2 = summarize_document(doc2_path, api_key, model_name)
    save_step_in_temp_dir(result2, 2, 'base_step') if save_intermediate_steps else None
    compare_prompt = fill_compare_prompt(result1, result2)
    final_result = ask_openai(compare_prompt, model_name=model_name, max_tokens=4000)
    save_step_in_temp_dir(final_result, 3, 'base_step') if save_intermediate_steps else None
    return final_result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    doc1_path = "../examples/doc1.docx"
    doc2_path = "../examples/doc2.docx"
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-3.5-turbo"

    result = pipeline(doc1_path, doc2_path, api_key, model_name)
    print(result)
