import requests
import json

def llm_ask(message, hideThinking=True, model="qwen3:8b-q4_K_M"):
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
        x = x.split("</think>")
        if(len(x) > 1):
            x = ''.join(x[1:])
        else:
            x = x[0]

    # print('---')
    # print(message)
    # print(x)
    return x

def llm_embedding(text, model="bge-m3"):
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

