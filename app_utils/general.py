import requests
import openai
import os
import re
import json
import matplotlib.pyplot as plt
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader


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
    fig, ax = plt.subplots(figsize=(1, 6))
    bar_width = 1
    for idx, value in enumerate(values):
        color = "red" if value > 0.18 else "green"
        # color = (value, 0, value)  # RGB, with blue channel set by the value
        ax.bar(0, 1, bottom=idx, width=bar_width, color=color, edgecolor='none')

    # rotate bar to horizontal
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, len(values))
    ax.axis('off')
    return fig


def plot_horizontal_bar_with_strips(values):
    fig, ax = plt.subplots(figsize=(6, 1))
    bar_height = 1
    left = 0  # starting position for the first segment

    for value in values:
        color = "gray" if value > 0.18 else "lightgray"
        ax.barh(0, value, height=bar_height, left=left, color=color, edgecolor='none')
        left += value  # update the starting position for the next segment

    ax.set_xlim(0, sum(values))
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    return fig


def split_document_to_pages(doc_path, chunk_size=2000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_loader = Docx2txtLoader(file_path=doc_path)
    pages = text_loader.load_and_split(text_splitter)
    return pages


def save_files_in_temp_dir(files):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    for i, file in enumerate(files):
        with open(os.path.join(temp_dir, f"file_{i}.docx"), "wb") as f:
            f.write(file.getvalue())
    return os.path.join(temp_dir, "file_0.docx"), os.path.join(temp_dir, "file_1.docx")


def save_step_in_temp_dir(step, step_number, prefix="step_base"):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, f"{prefix}_{step_number}.txt"), "w", encoding="utf-8") as f:
        f.write(step)
    return os.path.join(temp_dir, f"{prefix}_{step_number}.txt")


def save_step_df_in_temp_dir(df, step_number, prefix="step_base"):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    df.to_csv(os.path.join(temp_dir, f"{prefix}_{step_number}.csv"), index=False)
    return os.path.join(temp_dir, f"{prefix}_{step_number}.csv")


def df_to_structured_text(df):
    txt = "### LEGAL DOCUMENTS COMPARISON\n\n"
    for idx, row in df.iterrows():
        txt += f"#### Difference {idx + 1}\n\n"
        txt_line = "".join([f"{k}:\n{v}\n\n" if i != 2
                            else f"📌 **:blue[{k}:\n{v}]**\n\n"
                            for i, (k, v) in enumerate(row.items())])
        txt += txt_line + "\n\n" + "----\n\n"
    return txt


def df_to_structured_html(df):
    txt = "<h3>L<strong>EGAL DOCUMENTS COMPARISON</strong></h3>"
    for idx, row in df.iterrows():
        txt += f"<h4>Difference {idx + 1}</h4>"
        txt_line = "".join([f"<p><strong>{k}</strong>:<br>{v}</p>" if i != 2
                            else
                            f"""<p><strong style='color:RGB(79,70,229);font-size:1.2em;'>
                            {v}
                            </strong></p>"""
                            for i, (k, v) in enumerate(row.items())])
        txt += txt_line + "<hr>"
    return txt
