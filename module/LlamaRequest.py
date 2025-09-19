import logging
import requests
import json

logging.basicConfig(
    level=logging.INFO,
    filename="AAL.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s -\n%(message)s"
)

def llm_ask_high(message, hideThinking=True, model="glm-4.5", temperature=0.6, api_key="5bd9da9f9bb84796b4b49ab33a7545bc.yJHCbrOdVnrJVEWf"):
    """
    向智谱AI接口发送请求，只需要提供 message。
    返回完整回复字符串。
    """
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API调用失败: {response.status_code}, {response.text}")

    data = response.json()
    x = data['choices'][0]['message']['content']
    logging.info(f'''
----
<llm_ask>
QUESTION:
{message}
ANSWER:
{x}
----
''')

    return x


def llm_ask_low(message, hideThinking=True, model="qwen3:8b-q4_K_M"):
    """
    向本地 Ollama 发送请求，只需要提供 message。
    返回完整回复字符串。
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}]
    }

    response = requests.post(url, json=payload, stream=True)
    answer = []
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "message" in data:
                answer.append(data["message"]["content"])

    x = "".join(answer)
    if(hideThinking):
        x2 = x.split("</think>")
        if(len(x2) > 1):
            x2 = ''.join(x2[1:])
        else:
            x2 = x2[0]

    logging.info(f'''
----
<llm_ask>
QUESTION:
{message}
ANSWER:
{x}
----
''')
    
    return x2

def llm_ask(message, mode='low'):
    if mode == 'low':
        return llm_ask_low(message)
    else:
        return llm_ask_high(message)

def llm_embedding(text, model="bge-m3-cpu"):
    """
    调用 Ollama 本地 embedding 接口，获取文本向量。
    text: 需要生成 embedding 的文本
    model: embedding 模型（默认 bge-m3，需要本地先拉取 ollama pull bge-m3）
    返回 embedding 向量 (list[float])
    """
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }

    response = requests.post(url, json=payload)
    data = response.json()
    return data["embedding"]
