import requests
from openai import OpenAI
import openai
import os
import re
import json
import matplotlib.pyplot as plt
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader

text_splitter = CharacterTextSplitter()


def ask_openai(prompt,
               model_name="gpt-4o",
               verbose=True,
               max_tokens=300):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response.json()
    print(result) if verbose else None
    message = result['choices'][0]['message']['content']
    print(message) if verbose else None

    return message


def process_result(response):
    result = response.json()
    result = json.loads(result)
    return result['choices'][0]['message']['content']


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    r = openai.embeddings.create(input=[text], model=model)
    return r.dict()['data'][0]['embedding']


def merge_jsons(jsons):
    merged_df_data = []
    missed_indexes = []
    for idx, json_str in enumerate(jsons):
        json_str = json_str.replace("```json", "").replace("```", "")
        try:
            json_data = json.loads(json_str)
        except ValueError:
            # print(f"Invalid JSON [{idx}]: {json_str[:1000]}")
            missed_indexes.append(idx)
            continue
        for key, value in json_data.items():
            if isinstance(value, dict):
                merged_df_data.append(value)

    return merged_df_data, missed_indexes


def merge_final(final_result):
    df_data = []
    missed_ids = []
    # clean json string
    final_result = re.sub(r'```json\n', '', final_result)
    final_result = re.sub(r'\n```', '', final_result)
    final_result = re.sub(r'```', '', final_result)
    final_result = re.sub(r'\n', '', final_result)
    try:
        json_data = json.loads(final_result)
        for key, value in json_data.items():
            if isinstance(value, dict):
                df_data.append(value)
            else:
                missed_ids.append(key)
        return df_data, missed_ids
    except ValueError:
        print("Invalid JSON")
        return [], []


def plot_vertical_bar_with_strips(values):
    fig, ax = plt.subplots(figsize=(3, 4))
    bar_width = 2
    for idx, value in enumerate(values):
        color = "red" if value > 0.18 else "green"
        # color = (value, 0, value)  # RGB, with blue channel set by the value
        ax.bar(0, 1, bottom=idx, width=bar_width, color=color, edgecolor='none')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, len(values))
    ax.axis('off')
    plt.show()


def split_document_to_pages(doc_path):
    text_loader = Docx2txtLoader(file_path=doc_path)
    pages = text_loader.load_and_split(text_splitter)
    return pages


def save_step_in_temp_dir(step, step_number, prefix="step_base"):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, f"{prefix}_{step_number}.txt"), "w") as f:
        f.write(step)
    return os.path.join(temp_dir, f"{prefix}_{step_number}.txt")


def save_step_df_in_temp_dir(df, step_number, prefix="step_base"):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    df.to_csv(os.path.join(temp_dir, f"{prefix}_{step_number}.csv"), index=False)
    return os.path.join(temp_dir, f"{prefix}_{step_number}.csv")
