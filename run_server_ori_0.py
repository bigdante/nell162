from flask import Flask, request, Response
import math
from flask_cors import CORS
from tool.utils import *
from bson import ObjectId
from fuzzywuzzy import fuzz

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)

# 程序开始运行的时间点
program_start_time = datetime.datetime.strptime('2022-12-01 00:00:00', '%Y-%m-%d %H:%M:%S')

# 记录总的条数，防止每次计算耗时太长
# WikipediaEntity_total = WikipediaEntity.objects.count()
# triple_total = TripleFact.objects.count()
# relation_total = BaseRelation.objects.count()
WikipediaEntity_total = 11827927
triple_total = 22958671
relation_total = 2400

with open('./data/entity_id_wiki_triple.json', 'r') as f:
    lines = f.readlines()
    wiki_in_triple = []
    for line in lines:
        wiki_in_triple.append(json.loads(line))
    random.shuffle(wiki_in_triple)
    total = len(wiki_in_triple)


@app.route('/latest', methods=['GET', 'POST'])
def latest():
    '''
        return the latest triples extracted
    '''
    start = time.time()
    params = get_params(request)
    result = get_latest_triple(params)
    print("latest entity show done, consume time {:.2f} s".format(time.time() - start))
    return result


@app.route('/dashboard', methods=['GET'])
def dashboard():
    today = datetime.datetime.now()
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
            "all_entities": WikipediaEntity_total,
            "all_relations": relation_total,
            'running_days': (today - program_start_time).days
        }
    )


@app.route('/get_page_by_id', methods=['GET', 'POST'])
def get_page_by_id():
    start = time.time()
    params = get_params(request)
    result = get_pages(params['inPageId'])
    save_result_json(result, "./data/page_id.json")
    print("page show done, consume time {:.2f} s".format(time.time() - start))
    return output_process(result)


@app.route('/pps', methods=['GET', 'POST'])
def show_pps():
    '''
        get entity search results
    '''
    start = time.time()
    params = get_params(request)
    print("search for", params["query_name"])
    results = get_entity(params["query_name"])
    print("search done, consume time {:.2f} s".format(time.time() - start))
    return output_process(results)


@app.route('/entity_detail', methods=['GET', 'POST'])
def entity_detail():
    result = []
    start = time.time()
    params = get_params(request)
    for _, triple in enumerate(TripleFact.objects(headWikipediaEntity=ObjectId(params["id"]))):
        r = precess_db_data(triple, need_span=True)
        r['evidences'][0].pop("timestamp")
        r["head_unified"] = params["text"]
        if fuzz.ratio(params['text'], r["head_entity"]) < 50:
            continue
        hrt = r["head_entity"] + r["relation"] + r["tail_entity"]
        flag = 1
        for index, triple in enumerate(result):
            if hrt == triple["head_entity"] + triple["relation"] + triple["tail_entity"]:
                flag = 0
                result[index]["evidences"].append(r["evidences"][0])
                break
        if flag == 1:
            result.append(r)
    save_result_json(result, "./data/entity_deatil.json")
    print("entity detail {} show done, consume time {:.2f} s".format(params["text"], time.time() - start))
    return output_process(result)


@app.route('/entity_table', methods=['GET', 'POST'])
def show_entity_table():
    result = []
    entity_list = []
    start = time.time()
    params = get_params(request)
    pages = math.ceil(total / params["size"])
    if params["refresh"]:
        params["page"] = random.randint(0, pages - 1)
    start_item = params["size"] * (params["page"] - 1)
    end_item = params["size"] * params["page"]
    wiki_in_triple_id_list = [id for id in wiki_in_triple[start_item:end_item]]
    for entity_id in wiki_in_triple_id_list:
        text = WikipediaEntity.objects.get(id=ObjectId(entity_id)).text
        for triple in TripleFact.objects(headWikipediaEntity=ObjectId(entity_id)):
            if fuzz.ratio(text, triple.head) < 50:
                continue
            entity_list.append({"id": str(entity_id), "text": text})
            break
    result_entity_table = {
        "data": entity_list,
        "pages": pages,
        "total": total
    }
    save_result_json(result_entity_table, "./data/result_entity_table.json")
    print("hot entity show done, consume time {:.2f} s".format(time.time() - start))
    return result_entity_table


@app.route('/thumb_up_down', methods=['GET', 'POST'])
def up_down():
    start = time.time()
    params = get_params(request)
    triple = TripleFact.objects.get(id=ObjectId(params["id"]))
    if type(triple.upVote) != int:
        triple.upVote = 0
    if type(triple.downVote) != int:
        triple.downVote = 0
    if params["type"] == "up":
        triple.upVote += 1
    elif params["type"] == "down":
        triple.downVote += 1
    result = triple.save()
    if result:
        print("save done, consume time {:.2f} s".format(time.time() - start))
        return {"success": True}
    else:
        print("save failed, consume time {:.2f} s".format(time.time() - start))
        return {"success": False}


@app.route('/get_relation', methods=['GET', 'POST'])
def get_relation():
    get_relation_alias()
    return "ok"


@app.route('/re', methods=['POST', "GET"])
def re():
    def generate():
        i = 0
        input = "[Thought] To be a fact extractor, I need to start by retrieving sentences from Wikipedia."
        history = []
        while True:
            i += 1
            yield f"data: ===========Thinking (^-^)==============\n\n\n\n"
            print("yield", i)
            print(">>>>>>>>>>>>>>>>>>>>>>>")
            response, history, method_return = inference(input, history)
            yield f"data: {response}\n"
            if method_return:
                yield f"data: {method_return}\n"
            yield f"data:   \n"
            print("<<<<<<<<<<<<<<<<<<<<<<<<\n")
            input = "what about next"
            if response == "[Return] EXIT" or method_return == "[Return] EXIT":
                yield "data: all done"
                yield f"data: \n"
                break
            if method_return:
                last_item = history[-1]
                new_last_item = (last_item[0], last_item[1] + "\n" + method_return)
                history[-1:] = [new_last_item]
            if i > 100:
                break
            time.sleep(1)

    return Response(generate(), content_type="text/event-stream")


@app.route('/re2', methods=['POST', "GET"])
def re2():
    def generate():
        input = "[Thought] To be a fact extractor, I need to start by retrieving sentences from Wikipedia."
        history = []
        i = 0
        while True:
            i += 1
            print(">>>>>>>>>>>>>>>>>>>>>>>")
            response, history, method_return = inference(input, history)
            yield f"data: {response}\n"
            yield f"data: {method_return}"
            yield f"data: \n"
            print("<<<<<<<<<<<<<<<<<<<<<<<<\n")
            input = "what about next"
            if response == "[Return] EXIT" or method_return == "[Return] EXIT":
                yield "data: all done"
                yield f"data: \n"
                print("="*30)
                break
            if method_return:
                last_item = history[-1]
                new_last_item = (last_item[0], last_item[1] + "\n" + method_return)
                history[-1:] = [new_last_item]
            if i > 100:
                break
            time.sleep(1)

    return Response(generate(), content_type="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8841, debug=True)
