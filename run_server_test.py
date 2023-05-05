from flask import Flask, abort, request
from datetime import datetime
import random
from mongoengine.queryset.visitor import Q
import json,time,math
from requests import post
from tqdm import tqdm
from flask_cors import cross_origin, CORS
from data_object import *
from tool.utils import *
import uuid
from bson import ObjectId
from fuzzywuzzy import fuzz


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)

# 程序开始运行的时间点
d1 = datetime.datetime.strptime('2022-12-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# 记录总的条数，防止每次计算
# total = WikidataEntity.objects.count()
total = 10630454
triple_total = 22958671
relation_total = 2400

@app.route('/latest', methods=['GET','POST'])
def latest():
    '''
        return the latest triples extracted
    '''
    start = time.time()
    params = get_params(request)
    result = get_latest_triple(params)
    print(f"latest show done, consume time {time.time()-start} s")
    return result[:300]


@app.route('/dashboard', methods=['GET'])
def dashboard():
    d2 = datetime.datetime.now()
    # return output_process(
    #     {
    #         "all_triples": TripleFact.objects.count(),
    #         "all_entities": WikidataEntity.objects.count(),
    #         "all_relations": BaseRelation.objects.count(),
    #         'running_days': (d2-d1).days
    #     }
    # )
    return output_process(
        {
            "all_triples": triple_total,
            "all_entities": total,
            "all_relations": relation_total,
            'running_days': (d2 - d1).days
        }
    )

@app.route('/get_page_by_id', methods=['GET','POST'])
def get_page_by_id():
    result = []
    params = get_params(request)
    result = get_pages(params['inPageId'])
    save_result_json(result,"./data/page_id.json")
    return output_process(result)


@app.route('/pps', methods=['GET', 'POST'])
def show_pps():
    '''
        get the page and entity search results
    '''
    params = get_params(request)
    print("search for",params["query_name"])
    if params['relation']:
        results = get_relation(params["query_name"])
    else:
        results = get_page_entity(params["query_name"])
    return output_process(results)


@app.route('/entity', methods=['GET', 'POST'])
def show_entity():
    entity_list = []
    start = time.time()
    params = get_params(request)
    # pages总共的页数
    # total总共的wikidata-entity的个数
    pages = math.ceil(total /params["size"])
    # 刷新和分页
    # while True:
    if params["refresh"]:
        params["page"] = random.randint(0,pages-1)
    start_item = params["size"]*(params["page"]-1)
    end_item = params["size"]*params["page"]
    for index, entity in enumerate(WikidataEntity.objects[start_item:end_item]):
        if TripleFact.objects(headWikidataEntity=ObjectId(entity.id)):
            entity_list.append({"id":str(entity.id),"text":entity.text})
    result ={
        "train_auto_glm_data":entity_list,
        "pages":pages,
        "total":total
    }
    if not entity_list:
        print("entity_list is empty")
    save_result_json(result,"./data/entity_list.json")
    print("entity bar show done, consume time {:.2f}s".format(time.time()-start)) 
    return result

@app.route('/entity_detail', methods=['GET','POST'])
def entity_detail():
    result = []
    start = time.time()
    params = get_params(request)
    for _, triple in enumerate(TripleFact.objects(headWikidataEntity=ObjectId(params["id"]))):
        r = precess_db_data(triple,need_span=True)
        r['evidences'][0].pop("timestamp")
        r["head_unified"] = params["text"]
        if fuzz.ratio(params['text'],r["head_entity"]) <50:
            continue
        hrt = r["head_entity"] + r["relation"] + r["tail_entity"]
        flag = 1
        for index, triple in enumerate(result):
            # 如果triple相同，则将evidence进行合并
            if hrt==triple["head_entity"] + triple["relation"] + triple["tail_entity"]:
                flag = 0
                result[index]["evidences"].append(r["evidences"][0])
                break
        # 如果triple不相同，才将r进行添加
        if flag == 1 :
            result.append(r)
    save_result_json(result,"./data/entity_deatil.json")
    print("entity-detail {} search done, consume time {:.2f}s".format(params["text"],time.time()-start)) 
    return output_process(result)


@app.route('/entity_table', methods=['GET', 'POST'])
def show_entity_table():
    entity_list = []
    start = time.time()
    params = get_params(request)
    # pages总共的页数
    # total总共的wikidata-entity的个数
    pages = math.ceil(total /params["size"])
    # 刷新和分页
    # while True:
    if params["refresh"]:
        params["page"] = random.randint(0,pages-1)
    start_item = params["size"]*(params["page"]-1)
    end_item = params["size"]*params["page"]
    for index, entity in enumerate(WikidataEntity.objects[start_item:end_item]):
        if TripleFact.objects(headWikidataEntity=ObjectId(entity.id)):
            entity_list.append({"id":str(entity.id),"text":entity.text})
    result ={
        "train_auto_glm_data":entity_list,
        "pages":pages,
        "total":total
    }
    if not entity_list:
        print("entity_list is empty")
    save_result_json(result,"./data/entity_list.json")
    print("entity bar show done, consume time {:.2f}s".format(time.time()-start)) 
    return result

@app.route('/thumb_up_down', methods=['GET','POST'])
def up_down():
    start = time.time()
    params = get_params(request)

    triple = TripleFact.objects.get(id=ObjectId(params["id"]))
    if type(triple.upVote)!=int:
        triple.upVote = 0
    if type(triple.downVote)!=int:
        triple.downVote = 0
    if params["type"] == "up":
        triple.upVote += 1
    elif params["type"] == "down":
        triple.downVote += 1
    result = triple.save()
    print("save done")
    if result:
        print("record done, consume time {:.2f}s".format(time.time()-start)) 
        return {"success":True}
    else:
        print("record failed, consume time {:.2f}s".format(time.time()-start)) 
        return {"success":False}

@app.route('/init', methods=['GET','POST'])
def init():
    from bson.tz_util import utc
    timestamp = datetime.datetime(2023, 6, 14, 12, 40, 38, tzinfo=utc)
    # 22958671

    # id=ObjectId(page_id))
    # 16893636 629825c697eaefbcc20b30b9 save done 
    start = 16893636
    for index, t in enumerate(TripleFact.objects[start:]):
        # if t.id == ObjectId("62aedf71c6a0977c31a5bc9a"):
        # print(t.timestamp)
        # if not t.timestamp:
        #     break
        # print(t)
        # t.upVote = 0
        # t.downVote = 0
        # t.isNewFact = 0
        timestamp = timestamp + datetime.timedelta(seconds=1)
        t.timestamp = timestamp
        t.save(validate=False)
        print(index+start, t.id,"save done ")
        # print(t.timestamp)
        # break
    print("init done")
    return "ok"

# @app.route('/initPage', methods=['GET','POST'])
# def initPara():
#     # 604 7494
#     # start = 1145130
#     # start = 0
#     # for index, page in enumerate(WikipediaPage.objects[start:]):
#     #     for paragrah in page.paragraphs:
#     #         # print(paragrah.id)
#     #         for sentence in paragrah.sentences:
#     #             # print(sentence)
#     for triple in TripleFact.objects():
#         print(triple.evidence.refPage.id)
#         exit()
#                     # triple.inPageId = page.id
#                     # triple.save(validate=False)
#                     # print(f"{index+start}, page_id: {page.id} save done")
#                     # exit()
#     print("init done")
#     return "ok"

@app.route('/initPage', methods=['GET','POST'])
def initPara():
    print("initing......")
    start_item = 0
    end_item = WikipediaEntity.objects.count()
    print(end_item)
    save_list = []
    with open("./entity_id_wiki_triple.json","w") as f:
        for index, entity in enumerate(WikipediaEntity.objects()):
            print(index)
            if TripleFact.objects(headWikipediaEntity=ObjectId(entity.id)):
                json.dump(str(entity.id),f)
                f.write('\n')
                # save_list.append(str(entity.id))
    # save_result_json(save_list,"./entity_id_wiki_triple.json")
    print("init done")
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8842)
    # app.run(host="0.0.0.0", port=8841)
    print("hhh")