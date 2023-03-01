from data_object import *
import numpy as np
import random
import datetime,time
import json
from bson.tz_util import utc
from bson import ObjectId
import uuid
from requests import post
from flask import jsonify
from mongoengine.queryset.visitor import Q
from tqdm import tqdm
import re
from fuzzywuzzy import fuzz
import collections

def sort_dict_by_key(d, key):
    sorted_dict = collections.OrderedDict()
    for k in sorted(d.keys(), key=lambda x: len(d[x].get(key)),reverse=True):
        sorted_dict[k] = d[k]
    d.clear()
    d.update(sorted_dict)
    list_result = []
    for k,v in d.items():
        list_result.append({k:v})
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
        params['start'] = datetime.datetime.fromtimestamp(message['start']/1000,tz=utc) if "start" in message.keys() else ""
        params['end'] = datetime.datetime.fromtimestamp(message['end']/1000,tz=utc) if "end" in message.keys() else ""
        params['inPageId'] = message['inPageId'] if "inPageId" in message.keys() else ""
    else:
        return " 'it's not a POST operation! \n"
    print(params)
    return params

def precess_db_data(db_document,need_span=True,need_time=False):
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
        indexs = np.asarray(db_document.headSpan)-BaseSentence.objects.get(id=db_document.evidence.id).charSpan[0]
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
        "headSpan" :indexs.tolist() if need_span else "",
        "evidenceID":str(db_document.evidence.id),
        "tripleID":str(db_document.id),
        "timestamp":db_document.timestamp if need_time else "",
        "inPageId":str(db_document.evidence.refPage.id) if db_document.evidence.refPage else ""
    }]

    return output


def call_es(text):
    '''
        get the entity search result
    '''
    headers = {'Content-Type': 'application/json'}
    url = 'http://166.111.7.106:9200/wikipedia_entity/wikipedia_entity/_search'
    data = {
        "query": {"bool": {"should": [{"match": {"text": text}}]}}
    }
    with post(url=url, headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf8'),auth=("nekol", "kegGER123")) as resp:
        results = resp.json()
        s_r =  results['hits']['hits']
    entity_names = [r['_source']['text'] for r in s_r]
    entity_ids = [r['_id'] for r in s_r]
    result_triples=get_entity_net(entity_ids,entity_names)
    return result_triples


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
    result = sort_dict_by_key(result,'tables')
    result_all["entity"] = result
    save_result_json(result_all,"./data/page_entity.json")
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
        if not sentences_ids[-1]=='enter':
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
                    if not id_sentence[id]=="\n":
                        result = [{
                            "_id": str(id),
                            "evidences": [{"text": id_sentence[id]}]
                        }]
                    else:
                        continue
                result_list.append(result)
        result_triples[str(page_id)] = result_list
    return result_triples


def get_entity_net(entity_ids,entity_names):
    '''
        根据id，获得head和tail为关键词的所有的信息
    '''
    result = {}
    for id, entity in zip(entity_ids,entity_names):
        tables=[]
        for triple in TripleFact.objects(Q(headWikipediaEntity=ObjectId(id))):
            r = precess_db_data(triple,need_span=False)
            if fuzz.ratio(entity,r["head_entity"]) <50:
                continue
            hrt = r["head_entity"] + r["relation"] + r["tail_entity"]
            flag = 1
            for index, triple_ in enumerate(tables):
                t = triple_["head_entity"] + triple_["relation"] + triple_["tail_entity"]
                if t == hrt:
                    flag = 0
                    tables[index]["evidences"].append(r["evidences"][0])
                    break
            if flag == 1 :
                tables.append(r)
            result[entity]={
                "tables":tables
            }
    return result


def get_latest_triple(params):
    '''
        查找指定时间范围内的triple数据，展示在latest中
    '''
    start,end = params['start'],params['end']
    result = []
    for index, triple in enumerate(TripleFact.objects((Q(timestamp__gte = start) & Q(timestamp__lte=end))).limit(300)):
        result.append(precess_db_data(triple,need_time=True))
    return result