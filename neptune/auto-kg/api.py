import copy
import json
import os
import random
import re
import time
import threading
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt import *
import requests
from tqdm import tqdm
import sys
import shutil

sys.path.append('..')
from transformers import GPT2Tokenizer

ori_keys = json.load(open("../data/base_data_for_all/120_key1.json"))
keys = [key for key, v in ori_keys.items() if v]
unused_keys = keys.copy()
used_keys = []
overload_keys = []
invalid_keys = []

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
rel_info = json.load(open("../data/base_data_for_all/rel_info.json"))
entity_relation_tail = json.load(open("../data/base_data_for_all/head_relation_tail_constrain_dev.json"))
data = json.load(open("../data/base_data_for_all/dev_revised.json"))

proxies = {
    'http': '127.0.0.1:9898',
    'https': '127.0.0.1:9898',
}


def thinking_animation(stop_event: threading.Event):
    start_time = time.time()
    animation_chars = ['-', '\\', '|', '/']
    idx = 0
    while not stop_event.is_set():
        elapsed_time = int(time.time() - start_time)
        print(f"\rThinking {animation_chars[idx % len(animation_chars)]}... Elapsed time: {elapsed_time}s ", end="")
        idx += 1
        time.sleep(0.5)


def make_chat_request_with_thinking(message, func: Callable):
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as executor:
        thinking_thread = executor.submit(thinking_animation, stop_event)
        answer_future = executor.submit(func, message)
        answer = answer_future.result()
        stop_event.set()
    print("\r", end="")
    sys.stdout.flush()
    return answer


def get_entity_choose(item, vertex_i, tail_constrain):
    tail_choose = {}
    tail_choose_ids = {}
    entity_list = copy.deepcopy(item['vertexSet'])
    for index, i in enumerate(entity_list):
        if i == vertex_i:
            continue
        tail_types = str(sorted(list(set([tail['type'] for tail in i]))))
        if tail_types in tail_constrain:
            entity_name = [tail['name'] for tail in i]
            sentence_ids = [tail['sent_id'] for tail in i]
            entity_name = sorted(list(set(entity_name)))
            sentence_ids = list(set(sentence_ids))
            tail_choose[index] = max(entity_name, key=len)
            tail_choose_ids[index] = sentence_ids
        else:
            continue
    sentence_id = sorted(list(set([x for lst in tail_choose_ids.values() for x in lst])))
    return tail_choose, sentence_id


def get_entity_sentence_id(vertex):
    sentence_id = []
    for entity in vertex:
        sentence_id.append(entity['sent_id'])
    return list(set(sentence_id))


def get_queries(item, vertex_i, relation):
    '''
        获取当前vertex_i的全部query
    '''
    head_sentence_id = get_entity_sentence_id(vertex_i)
    head = max([i['name'] for i in vertex_i], key=len)
    head_types = sorted(list(set([i['type'] for i in vertex_i])))
    relations_constrains = entity_relation_tail.get(str(head_types), [])
    if not relations_constrains:
        return None
    tail_constrain = relations_constrains.get(relation, [])
    if not tail_constrain:
        return None
    tail_choose, tail_sentence_ids = get_entity_choose(item, vertex_i, tail_constrain)
    if not tail_choose:
        return None
    sentence_ids = sorted(list(set(head_sentence_id + tail_sentence_ids)))
    text = " ".join([" ".join(sen) for index, sen in enumerate(item.get("sents")) if index in sentence_ids])
    s_h = {
        "sentence": text,
        "head": head,
        "tail_choose": tail_choose
    }
    return s_h


def get_ground_tail_truth(item):
    h_r = {}
    for label in item.get("labels"):
        head = str(label['h'])
        relation = rel_info[label['r']]
        tail = label['t']
        if head + "_" + relation in h_r:
            h_r[head + "_" + relation].append(tail)
        else:
            h_r[head + "_" + relation] = [tail]
    return h_r


def get_logit_bias(tail_list):
    logit_bias = {"13": -100, "4083": -100}
    for t in tail_list:
        ids = tokenizer.encode_plus(t)['input_ids']
        for tid in ids:
            if str(tid) not in logit_bias:
                logit_bias[str(tid)] = 10
    return logit_bias


def get_sentence(relation):
    template = json.load(open("./alias_template.json"))
    relation_alias_templates = template[relation]
    print("making query")
    with open(f"./data/query_data/{relation}.json", "w") as file:
        uuid = 0
        query_id = 0
        for page_id, item in enumerate(tqdm(data, desc=relation)):
            h_r = get_ground_tail_truth(item)
            for head_id, vertex_i in enumerate(item.get("vertexSet")):
                ground_truth_tail_id_list = h_r.get(str(head_id) + "_" + relation, [])
                s_h_c = get_queries(item, vertex_i, relation)
                if not s_h_c:
                    continue
                true_tail = [(tail_id, tail_name) for tail_id, tail_name in s_h_c['tail_choose'].items() if tail_id in ground_truth_tail_id_list]
                for t_name, alias_template in relation_alias_templates.items():
                    suffix_template = " Possible answer:"
                    s = {
                        "uuid": uuid,
                        "query_id": query_id,
                        "sentence": s_h_c['sentence'],
                        "head": s_h_c['head'],
                        "tail_choose": list(s_h_c['tail_choose'].values()),
                        "true_tail": true_tail if true_tail else [],
                        "query": {
                            "prompt": alias_template.format(s={
                                "sentence": s_h_c['sentence'],
                                "head": s_h_c['head'],
                                "tail_choose": list(s_h_c['tail_choose'].values())
                            }) + suffix_template,
                            "logit_bias": get_logit_bias(list(s_h_c['tail_choose'].values())),
                        },
                        "template": t_name
                    }
                    json.dump(s, file)
                    file.write('\n')
                    uuid += 1
                query_id += 1
            # break
    return f"{relation} done"


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


def make_chat_request(message, max_length=1024, timeout=10, logit_bias=None, max_retries=5):
    global unused_keys, used_keys, overload_keys
    for index in range(max_retries):
        key = get_valid_key()
        try:
            with requests.post(
                    url=f"https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.0,
                        "messages": message,
                        "max_tokens": max_length,
                        "top_p": 1.0,
                        # "logit_bias": logit_bias
                    },
                    proxies=proxies,
                    # timeout=timeout
            ) as resp:
                if resp.status_code == 200:
                    used_keys.remove(key)
                    unused_keys.append(key)
                    return json.loads(resp.content)
                elif json.loads(resp.content).get('error'):
                    if json.loads(resp.content).get('error')['message'] == "You exceeded your current quota, please check your plan and billing details.":
                        invalid_keys.append(key)
                    else:
                        overload_keys.append((key, time.time()))
        except requests.exceptions.RequestException as e:
            used_keys.remove(key)
            unused_keys.append(key)
            timeout += 5
            if logit_bias:
                if timeout >= 20:
                    logit_bias = {"13": -100, "4083": -100}
                    print(f"Error with key {key}: {e}")
                else:
                    logit_bias = dict(list(logit_bias.items())[:int(len(logit_bias) / 2)])


def pross_answer(input_string):
    pattern = r"\[(.*?)\]"
    main_match = re.search(pattern, input_string)
    if not main_match:
        return [input_string]
    inner_string = main_match.group(1)
    inner_pattern = r"'(.*?)'|\"(.*?)\""
    matches = re.findall(inner_pattern, inner_string)
    result = [s[0] or s[1] for s in matches]
    return result


def process_one_data(args):
    data, relation = args
    try:
        data = eval(data)
    except:
        data = data
    prompt, logit_bias = data['query']["prompt"], data['query']["logit_bias"]
    message = [
        {"role": "system", "content": extract_prompt},
        {"role": "user", "content": prompt}
    ]
    answer = make_chat_request(logit_bias=logit_bias, message=message)
    try:
        answer = answer['choices'][0]['message']['content']
        answer = pross_answer(answer)
    except:
        answer = ["network error"]

    item = {
        "input": data,
        "output": answer
    }
    with open(f"./data/query_result/{relation}.json", "a") as f:
        f.write(json.dumps(item) + "\n")

    return "success"


def process_all_data(data_list, relation):
    results = []
    max_threads = min(os.cpu_count(), len(keys) - len(invalid_keys))
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_one_data, (data, relation)): data for data in data_list}
        with tqdm(total=len(data_list), desc=f"{relation}") as progress_bar:
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
    with open("../data/base_data_for_all/120_key1.json", 'w') as file:
        json.dump(ori_keys, file)


def get_gpt_result(relation):
    template = json.load(open("./alias_template.json"))
    relation_alias_templates = template[relation]
    print("\033[1;32m--------> the relation alias provided are following\033[0m ")
    for index, (k, v) in enumerate(relation_alias_templates.items()):
        print(f"{index + 1}】{k} : {v}")
    input_ = input(f"\033[1;32mDo you satisfy with these alias templates? Y(continue), N(regenerate new alias templates): \033[0m")
    while True:
        if input_.lower() == "n":
            reconstruct_alias(relation)
            break
        elif input_.lower() == "y":
            break
        else:
            print("please re_input")
            input_ = input(f"\033[1;32mDo you satisfy with these alias templates? Y(continue), N(regenerate new alias templates): \033[0m")
    base_file_path = "./data/query_data"
    result_file_path = "./data/query_result"
    file_path = os.path.join(base_file_path, relation + ".json")
    out_path = os.path.join(result_file_path, relation + ".json")
    try:
        data = get_uncompleted_data(file_path, out_path)
        if data:
            while True:
                input_ = input(f"\033[1;32m{relation} remain {len(data)} to query, do you want to continue?  Y(continue), N(re-generate):\033[0m ")
                if input_.lower() == "n":
                    os.remove(out_path)
                    data = open(file_path).readlines()
                    break
                elif input_.lower() == "y":
                    print("re-generate.........")
                    break
                else:
                    print("invalid input")
        else:
            print(f"{relation} query completed")
            while True:
                input_ = input("\033[1;32mDo you want to re-generate? Y(re-generate), N(check alias performance directly): \033[0m ")
                if input_.lower() == "y":
                    shutil.copy(out_path, os.path.join(result_file_path, relation + "_old.json"))
                    os.remove(out_path)
                    data = open(file_path).readlines()
                    break
                elif input_.lower() == "n":
                    return
                else:
                    print("invalid input")
    except FileNotFoundError:
        print(f"File '{out_path}' not found, query begin")
        get_sentence(relation)
        data = open(file_path).readlines()
    process_all_data(data, relation)


def vote(relation):
    relation_ground_truth_count = json.load(open("../data/base_data_for_all/relation_count.json"))
    result_data = []
    with open(f"./data/query_result/{relation}.json", "r") as f:
        for line in f:
            json_obj = json.loads(line)
            result_data.append(json_obj)
    vote_data = {}
    with open(f"./data/query_vote/{relation}.json", "w") as out:
        for query in result_data:
            if query['input']['query_id'] in vote_data:
                vote_data[query['input']['query_id']]["answer"].append(list(set(query['output'])))
            else:
                vote_data[query['input']['query_id']] = {"query_id": query['input']['query_id'], "sentence": query['input']['sentence'],
                                                         "exact query": query['input']['query']['prompt'],
                                                         "head": query['input']['head'],
                                                         "true_tail": [t[1] for t in query['input'][
                                                             'true_tail']], "answer": [list(set(query['output']))]}
        tp, fp, tn, fn = 0, 0, 0, 0
        for query_id, query_data in vote_data.items():
            answer_list = query_data['answer']
            for candidate in set([item for sublist in answer_list for item in sublist]):
                candidate_count = sum([1 for sublist in answer_list if candidate in sublist])
                copy_vote_data = query_data.copy()
                if candidate_count >= len(answer_list) / 2:
                    copy_vote_data["vote_answer"] = candidate
                else:
                    copy_vote_data["vote_answer"] = "unknown"
                if copy_vote_data["vote_answer"] != "unknown" and copy_vote_data["vote_answer"] in query_data["true_tail"]:
                    copy_vote_data["label"] = "yes"
                    tp += 1
                if copy_vote_data["vote_answer"] == "unknown" and not query_data["true_tail"]:
                    copy_vote_data["label"] = "tn"
                    tn += 1
                if copy_vote_data["vote_answer"] != "unknown" and copy_vote_data["vote_answer"] not in query_data["true_tail"]:
                    copy_vote_data["label"] = "no"
                    fp += 1
                if copy_vote_data["vote_answer"] == "unknown" and query_data["true_tail"]:
                    copy_vote_data["label"] = "fn"
                    fn += 1
                out.write(json.dumps(copy_vote_data) + "\n")
        print(f"overall: {relation} tp:{tp} fp:{fp} tn:{tn} fn:{fn}, recall:{tp / relation_ground_truth_count[relation]}")
        return tp / relation_ground_truth_count[relation]


def check_every_template(relation, old):
    result_data = []
    if not old:
        with open(f"./data/query_result/{relation}.json", "r") as f:
            for line in f:
                json_obj = json.loads(line)
                result_data.append(json_obj)
    else:
        template = json.load(open("./alias_template.json"))
        relation_alias_templates = template[relation + "_old"]
        print("\033[1;32m--------> the relation【old】 alias provided are following\033[0m")
        for index, (k, v) in enumerate(relation_alias_templates.items()):
            print(f"{index + 1}】{k} : {v}")
        with open(f"./data/query_result/{relation}_old.json", "r") as f:
            for line in f:
                json_obj = json.loads(line)
                result_data.append(json_obj)

    template = json.load(open("./alias_template.json"))
    alias = template[relation].keys() if not old else template[relation + "_old"].keys()
    alias_result_dict = {a: {"tp": 0, "fp": 0} for a in alias}
    try:
        for r in result_data:
            for o in list(set(r['output'])):
                if o in [t[1] for t in r['input']['true_tail']]:
                    alias_result_dict[r['input']['template']]['tp'] += 1
                else:
                    if o != "unknown":
                        alias_result_dict[r['input']['template']]['fp'] += 1
    except:
        if old:
            print(f"\033[1;32mthe {relation}_old.json is not generated by the old alias, you must choose Y(regenerate). \033[0m")
            print("=" * 50)
        return None, None
    sorted_alias_result = sorted(alias_result_dict.items(), key=lambda x: x[1]['tp'], reverse=True)
    recall = vote(relation)
    print("----------result detail----------")
    for k, v in sorted_alias_result:
        print(f"--->{k}, tp:{v['tp']}, fp:{v['fp']}")
    return sorted_alias_result, recall


used_alias_prompt = "The used alias templates are following:\n"


def reconstruct_alias(relation, sorted_alias_result="do not have scores yet"):
    global used_alias_prompt
    template = json.load(open("./alias_template.json"))
    old_alias_template = template[relation]
    relation_descript = json.load(open("../data/base_data_for_all/relation_descript.json"))[relation]
    s = {
        "template": old_alias_template,
        "relation": relation,
        "relation_descript": relation_descript,
        "score": sorted_alias_result,
    }
    while True:
        try:
            if len(used_alias_prompt) > 2048:
                a = used_alias_prompt.split("\n")[1:]
                used_alias_prompt = a[0] + random.sample(a, 2)
            message = [
                {"role": "user", "content": used_alias_prompt + str(old_alias_template) + "\n" + re_alias_prompt.format(s=s)}
            ]
            answer = make_chat_request_with_thinking(message, make_chat_request)
            new_alias = eval(answer['choices'][0]['message']['content'])
            for index, (k, v) in enumerate(new_alias.items()):
                print(f"{index + 1}】{k} : {v}")
            while True:
                input_ = input(f"\033[1;32mDo you want to use the new alias? Y(replace the old one), N(regenerate new alias): \033[0m")
                if input_.lower() == "y":
                    with open('alias_template.json', 'r') as f:
                        alias_dict = json.load(f)
                    alias_dict[relation + "_old"] = old_alias_template
                    alias_dict[relation] = new_alias
                    with open('alias_template.json', 'w') as f:
                        json.dump(alias_dict, f, indent=4)
                    shutil.copy(f"./data/query_result/{relation}.json", f"./data/query_result/{relation}_old.json")
                    os.remove(f"./data/query_result/{relation}.json")
                    # get_sentence(relation)
                    return
                elif input_.lower() == "n":
                    old_alias_template = new_alias
                    break
                else:
                    print("invalid input")
        except Exception as e:
            print(e)
