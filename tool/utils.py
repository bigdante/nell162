import pickle
import sys
import time
import requests
import os
import re
import importlib

from data_object import *
import numpy as np
import ast
import random
import datetime
import json
from bson.tz_util import utc
from bson import ObjectId
from requests import post
from flask import jsonify
from mongoengine.queryset.visitor import Q
from fuzzywuzzy import fuzz
import collections
import threading
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
from tool.model import *



relation_list = ['country of citizenship', 'date of birth', 'place of birth', 'participant of',
                 'located in the administrative territorial entity', 'contains administrative territorial entity',
                 'participant', 'location', 'followed by', 'country', 'educated at', 'date of death', 'sibling',
                 'head of government', 'legislative body', 'conflict', 'applies to jurisdiction', 'instance of',
                 'performer', 'publication date', 'creator', 'author', 'composer', 'lyrics by', 'member of',
                 'notable work', 'inception', 'part of', 'cast member', 'director', 'has part', 'production company',
                 'owned by', 'headquarters location', 'developer', 'manufacturer', 'country of origin', 'publisher',
                 'parent organization', 'subsidiary', 'capital of', 'capital', 'spouse', 'father', 'child', 'religion',
                 'mother', 'located in or next to body of water', 'located on terrain feature', 'basin country',
                 'member of political party', 'mouth of the watercourse', 'place of death', 'military branch',
                 'work location', 'start time', 'award received', 'point in time', 'founded by', 'employer',
                 'head of state', 'member of sports team', 'league', 'present in work', 'position held', 'chairperson',
                 'languages spoken, written or signed', 'location of formation', 'operator', 'producer', 'record label',
                 'follows', 'replaced by', 'replaces', 'end time', 'subclass of', 'residence', 'sister city',
                 'original network', 'ethnic group', 'separated from', 'screenwriter', 'continent', 'platform',
                 'product or material produced', 'genre', 'series', 'narrative location', 'parent taxon',
                 'original language of work', 'dissolved, abolished or demolished', 'territory claimed by',
                 'characters', 'influenced by', 'official language', 'unemployment rate']

ori_keys = json.load(open("data/keys.json"))
keys = [key for key, v in ori_keys.items() if v]
unused_keys = keys.copy()
used_keys = []
overload_keys = []
invalid_keys = []
proxies = {
    'http': '127.0.0.1:9898',
    'https': '127.0.0.1:9898',
}





def sort_dict_by_key(d, key):
    sorted_dict = collections.OrderedDict()
    for k in sorted(d.keys(), key=lambda x: len(d[x].get(key)), reverse=True):
        sorted_dict[k] = d[k]
    d.clear()
    d.update(sorted_dict)
    list_result = []
    for k, v in d.items():
        list_result.append({k: v})
    return list_result


def get_params(request):
    '''
        get parameters
    '''
    params = {}
    if request.method == 'POST':
        message = eval(request.data)
        params["query_name"] = message['query_name'] if "query_name" in message.keys() else ""
        # 分页
        params["page"] = message['page'] if "page" in message.keys() else 1
        params["size"] = message['size'] if "size" in message.keys() else 50
        # 刷新
        params["refresh"] = message['refresh'] if "refresh" in message.keys() else False
        # id作为查询entity
        params["id"] = message['id'] if "id" in message.keys() else ""
        # 查询entity
        params["text"] = message['text'] if "text" in message.keys() else ""
        # 点赞和点踩的类型
        params["type"] = message['type'] if "type" in message.keys() else ""
        # latest的start和end时间戳
        params['start'] = datetime.datetime.fromtimestamp(message['start'] / 1000,
                                                          tz=utc) if "start" in message.keys() else ""
        params['end'] = datetime.datetime.fromtimestamp(message['end'] / 1000,
                                                        tz=utc) if "end" in message.keys() else ""
        params['inPageId'] = message['inPageId'] if "inPageId" in message.keys() else ""
    else:
        return " 'it's not a POST operation! \n"
    # print(params)
    return params


def precess_db_data(db_document, need_span=True, need_time=False):
    '''
        formate the result for browser
    '''
    output = {}
    output['new'] = db_document.isNewFact
    output['triple_id'] = str(db_document.id)
    if db_document.headWikidataEntity:
        head_id = db_document.headWikidataEntity.id
        output['head_id'] = str(head_id)
    output["head_linked_entity"] = "????"
    if need_span:
        indexs = np.asarray(db_document.headSpan) - BaseSentence.objects.get(id=db_document.evidence.id).charSpan[0]
        output['headSpan'] = indexs.tolist()
    output['head_entity'] = db_document.head
    output['relation'] = db_document.relationLabel
    output['tail_entity'] = db_document.tail
    output['evidences'] = [{
        "up": db_document.upVote,
        "down": db_document.downVote,
        "text": db_document.evidenceText,
        "extractor": "GLM-2B/P-tuning",
        "confidence": random.random(),
        "filtered": True,
        "headSpan": indexs.tolist() if need_span else "",
        "evidenceID": str(db_document.evidence.id),
        "tripleID": str(db_document.id),
        "timestamp": db_document.timestamp if need_time else "",
        "ts": db_document.timestamp if need_time else "",
        "inPageId": str(db_document.evidence.refPage.id) if db_document.evidence.refPage else ""
    }]

    return output


def call_es(text):
    '''
        get the entity search result
    '''
    headers = {'Content-Type': 'application/json'}
    url = 'http://166.111.7.106:9200/wikipedia_entity/wikipedia_entity/_search'
    # url = 'http://166.111.7.106:9200/wikipedia_paragraph/wikipedia_paragraph/_search'
    # url = 'http://166.111.7.106:9200/wikipedia_sentence/wikipedia_sentence/_search'
    data = {
        "query": {"bool": {"should": [{"match": {"text": text}}]}}
    }
    with post(url=url, headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf8'),
              auth=("nekol", "kegGER123")) as resp:
        results = resp.json()
        s_r = results['hits']['hits']
    entity_names = [r['_source']['text'] for r in s_r]
    entity_ids = [r['_id'] for r in s_r]
    result_triples = get_entity_net(entity_ids, entity_names)
    return result_triples


def engine(text, mode="para"):
    head, tail = text
    headers = {'Content-Type': 'application/json'}
    url_para = 'http://166.111.7.106:9200/wikipedia_paragraph/wikipedia_paragraph/_search'
    url_sentence = 'http://166.111.7.106:9200/wikipedia_sentence/wikipedia_sentence/_search'
    url = {"sentence": url_sentence, "para": url_para}
    data = {
        "query": {"bool": {"must": [{"match": {"text": head}}, {"match": {"text": tail}}]}}
    }
    with post(url=url[mode], headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf8'),
              auth=("nekol", "kegGER123")) as resp:
        results = resp.json()
        s_r = results['hits']['hits']

        return " ".join([r['_source']['text'] for r in s_r[:2]])


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return str(o)
        return json.JSONEncoder.default(self, o)


def output_process(result):
    e = JSONEncoder()
    result = json.loads(e.encode(result))
    return jsonify(result)


def save_result_json(result_triples, path):
    with open(path, 'w') as f:
        json.dump(result_triples, f, indent=4)


def get_entity(query_name):
    result_all = {}
    result = call_es(query_name)
    result = sort_dict_by_key(result, 'tables')
    result_all["entity"] = result
    save_result_json(result_all, "tool/data/page_entity.json")
    return result_all


def get_pages(page_id):
    '''
        get the pages by page_ids
    '''
    result_triples = {}
    page_sentence = {}
    id_sentence = {}
    page = WikipediaPage.objects.get(id=ObjectId(page_id))
    sentences_ids = []
    for paragrah in page.paragraphs:
        for sentence in paragrah.sentences:
            sentences_ids.append(sentence.id)
            id_sentence[sentence.id] = sentence.text
        # 代表一个段落后的换行，连续换行的话，就只保留一个
        if not sentences_ids[-1] == 'enter':
            sentences_ids.append("enter")
    page_sentence[page.id] = sentences_ids

    for page_id, sentences_id in page_sentence.items():
        # 获取所有page所有sentence对应所有的三元组信息
        result_list = []
        for index, id in enumerate(sentences_id):
            # 遇到id=enter，代表需要回车
            if id == "enter":
                r = result_list[-1][0]['evidences'][0]['text']
                if not r.endswith("\n"):
                    result_list[-1][0]['evidences'][0]['text'] += '\n'
            else:
                result = []
                for triple in TripleFact.objects(evidence=id):
                    t = precess_db_data(triple)
                    # t['evidences'][0].pop("timestamp")
                    result.append(t)
                # 如果对应的id在triple表中能找到，则result不为空，加入result_list，否则将句子直接放入
                # 判断下是否文本内容就是个"\n""
                if not result:
                    if not id_sentence[id] == "\n":
                        result = [{
                            "_id": str(id),
                            "evidences": [{"text": id_sentence[id]}]
                        }]
                    else:
                        continue
                result_list.append(result)
        result_triples[str(page_id)] = result_list
    return result_triples


def get_entity_net(entity_ids, entity_names):
    '''
        根据id，获得head和tail为关键词的所有的信息
    '''
    result = {}
    for id, entity in zip(entity_ids, entity_names):
        tables = []
        for triple in TripleFact.objects(Q(headWikipediaEntity=ObjectId(id))):
            r = precess_db_data(triple, need_span=False)
            if fuzz.ratio(entity, r["head_entity"]) < 50:
                continue
            hrt = r["head_entity"] + r["relation"] + r["tail_entity"]
            flag = 1
            for index, triple_ in enumerate(tables):
                t = triple_["head_entity"] + triple_["relation"] + triple_["tail_entity"]
                if t == hrt:
                    flag = 0
                    tables[index]["evidences"].append(r["evidences"][0])
                    break
            if flag == 1:
                tables.append(r)
            result[entity] = {
                "tables": tables
            }
    return result


def get_latest_triple(params):
    '''
        查找指定时间范围内的triple数据，展示在latest中
    '''
    start, end = params['start'], params['end']
    result = []
    for index, triple in enumerate(TripleFact.objects((Q(timestamp__gte=start) & Q(timestamp__lte=end))).limit(300)):
        result.append(precess_db_data(triple, need_time=True))
    return result


def get_relation_alias():
    from tqdm import tqdm
    save = {}
    for relation in tqdm(relation_list):
        save[relation] = [relation]
        for triple in BaseRelation.objects(Q(text=relation)):
            save[relation].extend(triple['alias'])

    json.dump(save, open("./alias.json", "w"), indent=4)








def inference(input, history):
    pattern = r'【(.*?)】'
    while True:
        response, history = model.chat(tokenizer, input, history=history)
        print(response)
        if response.startswith("[Thought]"):
            return response, history, None
        match = re.search(pattern, response)
        if match:
            result = match.group(1)
            for f in get_api_functions()[0]:
                if f in result:
                    method_return = get_api(f)
                    print(method_return)
                    return response, history, method_return
            print("no method match")
        else:
            print("no method match")


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


def save_var(var_name, var_value):
    try:
        with open('my_vars.pkl', 'rb') as f:
            saved_vars = pickle.load(f)
    except:
        saved_vars = {}
    saved_vars[var_name] = var_value

    with open('my_vars.pkl', 'wb') as f:
        pickle.dump(saved_vars, f)


def load_var(var_name):
    with open('my_vars.pkl', 'rb') as f:
        saved_vars = pickle.load(f)
    return saved_vars[var_name] if var_name in saved_vars else None


def get_api_functions():
    def get_functions_from_file(file_path):
        with open(file_path, "r") as f:
            file_content = f.read()
        module_node = ast.parse(file_content)
        function_nodes = [node for node in module_node.body if isinstance(node, ast.FunctionDef)]
        function_names = [func.name for func in function_nodes]
        return function_names

    tool_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(tool_dir)
    sys.path.insert(0, parent_dir)

    file_path = "tool/api.py"
    functions = get_functions_from_file(file_path)
    api_module = importlib.import_module(file_path[:-3].replace("/", "."))
    function_objs = [getattr(api_module, name) for name in functions]

    return functions, function_objs


def get_api(api_name: str, *args, **kwargs):
    functions, function_objs = get_api_functions()
    for name, func in zip(functions, function_objs):
        if name == api_name:
            return func(*args, **kwargs)
    print(f"No such function {api_name}")
