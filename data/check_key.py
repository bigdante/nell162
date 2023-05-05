import json
import random
import requests
import time
from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime
from tqdm import tqdm


def make_chat_request(prompt, key, max_length=1024):
    """
    For `gpt-3.5-turbo` api: $0.002 per 1k tokens.
    Introduction: https://platform.openai.com/docs/guides/chat/introduction
    Parameters: https://platform.openai.com/docs/api-reference/completions
    """
    # local_ip_address = "198.23.146.182"
    with requests.post(url=f"https://api.openai.com/v1/chat/completions",
                       headers={"Authorization": f"Bearer {key}"},
                       json={
                           "model": "gpt-3.5-turbo",
                           "temperature": 1.0,
                           "messages": [{'role': 'user', 'content': prompt}],
                           "max_tokens": max_length,
                           "top_p": 1.0,
                       }) as resp:
        return json.loads(resp.content)


if __name__ == '__main__':
    key = json.load(open("./keys.json"))
    # key = json.load(open("./old_key.json"))
    for k, v in key.items():
        prompt = "hi"
        start = time.time()
        a = make_chat_request(prompt, key=k)
        print(type(a), a)
        print(k, v, a['choices'][0]['message']['content'], time.time() - start)
