import json
import random
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from template import *

proxies = {
    'http': '127.0.0.1:9898',
    'https': '127.0.0.1:9898',
}

ori_keys = json.load(open("../../data/120_key1.json"))
keys = [key for key, v in ori_keys.items() if v]
unused_keys = keys.copy()
used_keys = []
overload_keys = []
invalid_keys = []


def get_valid_key():
    global unused_keys, used_keys, overload_keys
    current_time = time.time()
    new_overload_keys = []
    for key, timestamp in overload_keys:
        if current_time - timestamp >= 60:
            unused_keys.append(key)
        else:
            new_overload_keys.append((key, timestamp))
    overload_keys = new_overload_keys
    while not unused_keys:
        time.sleep(5)
    key = random.choice(unused_keys)
    unused_keys.remove(key)
    used_keys.append(key)
    return key


def make_chat_request(prompt, max_length=1024, timeout=10, logit_bias=None, max_retries=5):
    global unused_keys, used_keys, overload_keys
    for index in range(max_retries):
        key = get_valid_key()
        try:
            with requests.post(
                    url=f"https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": "gpt-3.5-turbo",
                        "temperature": 1.0,
                        "messages": [{'role': 'user', 'content': prompt}],
                        "max_tokens": max_length,
                        "top_p": 1.0,
                        "logit_bias": logit_bias,
                    },
                    # proxies=proxies,
                    timeout=timeout
            ) as resp:
                if resp.status_code == 200:
                    used_keys.remove(key)
                    unused_keys.append(key)
                    return json.loads(resp.content)
                elif json.loads(resp.content).get('error'):
                    print(json.loads(resp.content).get('error'))
                    if json.loads(resp.content).get('error')['message'] == "You exceeded your current quota, please check your plan and billing details.":
                        invalid_keys.append(key)
                    else:
                        overload_keys.append((key, time.time()))
        except requests.exceptions.RequestException as e:
            used_keys.remove(key)
            unused_keys.append(key)
            timeout += 5
            if timeout >= 20:
                logit_bias = {"13": -100, "4083": -100}
                print(f"Error with key {key}: {e}")
            else:
                logit_bias = dict(list(logit_bias.items())[:int(len(logit_bias) / 2)])


def get_uncompleted_data(file_path, out_path):
    all_uuids = {json.loads(line)["uuid"] for line in open(file_path)}
    completed_uuids = {json.loads(line)['input']["uuid"] for line in open(out_path) if json.loads(line)["output"] != ["network error"]}
    completed_data = [json.loads(line) for line in open(out_path) if json.loads(line)['input']["uuid"] in completed_uuids]
    uncompleted_uuids = all_uuids - completed_uuids
    if uncompleted_uuids:
        with open(out_path, "w") as f:
            for item in completed_data:
                f.write(json.dumps(item) + "\n")
    data = [json.loads(line) for line in open(file_path) if json.loads(line)["uuid"] in uncompleted_uuids]
    return data


def pross_answer(input_string):
    if input_string.startswith("yes"):
        return "yes"
    if input_string.startswith("no"):
        return "no"
    if input_string.startswith("unknown"):
        return "unknown"
    return input_string


def process_one_data(args):
    data, relation, mode = args
    try:
        data = eval(data)
    except:
        data = data
    prompt, logit_bias = data['query']["prompt"], data['query']["logit_bias"]
    answer = make_chat_request(prompt, logit_bias=logit_bias)
    try:
        answer = answer['choices'][0]['message']['content']
        answer = pross_answer(answer.strip().lower())
    except:
        answer = ["network error"]
    item = {
        "input": data,
        "output": answer
    }
    with open(f"./data/{mode}/query_result/{relation}.json", "a") as f:
        f.write(json.dumps(item) + "\n")
    return "success"


def process_all_data(data_list, relation, mode):
    results = []
    max_threads = min(os.cpu_count(), len(keys) - len(invalid_keys))
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_one_data, (data, relation, mode)): data for data in data_list}
        with tqdm(total=len(data_list), desc=f"{relation, relation_list.index(relation)}") as progress_bar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error occurred while processing data: {e}")
                progress_bar.update(1)
    for id in invalid_keys:
        ori_keys[id] = False
    # 将更改后的数据写回到 JSON 文件中
    with open("../../data/120_key1.json", 'w') as file:
        json.dump(ori_keys, file)
