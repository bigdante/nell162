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
    key = json.load(open("./120_key1.json"))
    # key = json.load(open("./old_key.json"))
    for k, v in key.items():
        system_prompt = "Response Format: \n" \
                        "Your respond must be a dict format, for example: \n" \
                        "    {\n" \
                        "      'alias1':'template1'\n" \
                        "      'alias2':'template2'\n" \
                        "      'alias3':'template3'\n" \
                        "    }" \
                        "\n" \
                        "Ensure the response can be parsed by Python eval().\n"
        example_prompt = "For example: given relation alias [\"country of citizenship\",\"subject of (country)\",\"subject of\",\"citizenship\",\"citizen of\",\"national of\"], you should give me relation alias templates as following:\n" \
                         "{" \
                         "\"country of citizenship\": \"Given the sentence: '{s[sentence]}', determine the country of citizenship for {s[head]}.\"," \
                         "\"subject of\": \"Given the sentence: '{s[sentence]}', identify the country for which {s[head]} is a subject.\"," \
                         "\"citizenship\": \"Given the sentence: '{s[sentence]}', determine the citizenship of {s[head]}.\"," \
                         "\"citizen of\": \"Given the sentence: '{s[sentence]}', identify the country of which {s[head]} is a citizen.\"," \
                         "\"national of\": \"Given the sentence: '{s[sentence]}', determine the nationality of {s[head]}.\"," \
                         "}\n" \
                         "the keys are relation alias, and the values are templates. You should give me relation alias template according to the relation description.\n" \
                         "{s[head]} must appear in the template, do not have {s[tail]} or {s[object]} in the template."
        prompt = "Make alias template for me with the given relation alias: ['country','nation','birthday'].\n" + example_prompt + system_prompt
        # prompt = "hi"
        start = time.time()
        a = make_chat_request(prompt, key="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")
        print(type(a), a)
        try:
            print(k, v, a['choices'][0]['message']['content'], time.time() - start)
        except:
            continue
