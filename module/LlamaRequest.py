import logging
import requests
import json
import os
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    filename="AAL.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s -\n%(message)s"
)

client = OpenAI(
    # 从环境变量中读取您的方舟API Key
    api_key=os.environ.get("ARK_API_KEY"), 
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

# 豆包
def llm_ask(message, mode='high', remark=None):
    model = "doubao-seed-1-6-thinking-250715"

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": message}
        ]
    )

    thinking = '<None>'
    if hasattr(completion.choices[0].message, 'reasoning_content'):
        thinking = completion.choices[0].message.reasoning_content
    
    remark_ = 'Remark:' + remark + '\n' if remark != None else ''

    logging.info(
        '----\n'
        f'<llm_ask_{model}>\n'
        f'{remark_}'
        'Question:\n'
        f'{message}\n'
        'Thinking:\n'
        f'{thinking}\n'
        'Answer:\n'
        f'{completion.choices[0].message.content}\n'
        '----\n'
    )
    return completion.choices[0].message.content

def llm_embedding(text, useCPU=False):
    """
    调用 Ollama 本地 embedding 接口，获取文本向量。
    text: 需要生成 embedding 的文本
    model: embedding 模型（默认 bge-m3，需要本地先拉取 ollama pull bge-m3）
    返回 embedding 向量 (list[float])
    """
    model = "bge-m3-cpu" if useCPU else "bge-m3"

    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }

    response = requests.post(url, json=payload)
    data = response.json()
    return data["embedding"]
