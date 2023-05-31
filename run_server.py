import os

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


def vote_tail():
    relation_dict = load_var("relation_ALIA")
    count_dict = {}
    for key, values in relation_dict.items():
        for value in values:
            count_dict[value] = count_dict.get(value, 0) + 1
    vote_set = set()
    for value, count in count_dict.items():
        if count >= len(relation_dict) / 2:
            vote_set.add(value)
    vote_set = list(vote_set)
    return vote_set


@app.route('/re', methods=['POST', "GET"])
def re():
    # os.remove("./my_vars.pkl")
    def print_tree(name, level):
        yield '│   ' * (level - 1) + '├── ' + name + "\n"

    def process_hierarchy(input, history, var_name, level):
        new_history = history.copy()
        yield from print_tree(f"Start {var_name} loop", level)
        while True:
            response, new_history, method_return = inference(input, new_history)
            yield f"├{'─' * level} {response}\n"
            if method_return:
                for m in method_return.split("\n"):
                    yield f"├{'─' * level} {m}\n"
            if method_return:
                last_item = new_history[-1]
                new_last_item = (last_item[0], last_item[1] + "\n" + method_return)
                new_history[-1:] = [new_last_item]
            if "[Return] " + var_name.upper() in response:
                break
        return new_history

    def generate():
        i = 0
        input = "[Thought] Retrieve sentences from Wikipedia."
        history = []
        while True:
            i += 1
            yield f"===========Thinking (^-^)==============\n"
            response, history, method_return = inference(input, history)
            yield f"{response}\n"
            if method_return:
                yield f"{method_return}\n"
            input = "next"
            if method_return == "[Return] EXIT":
                yield "loop done"
                break
            if method_return:
                last_item = history[-1]
                new_last_item = (last_item[0], last_item[1] + "\n" + method_return)
                history[-1:] = [new_last_item]
            if method_return and "[Return] ENTITIES" in method_return:
                head_history_ori = history.copy()
                for head in load_var("ENTITIES_as_head"):
                    save_var("HEAD", head)
                    history = process_hierarchy(input, head_history_ori, "TYPES", 2)
                    type_history_ori = history.copy()
                    for type in load_var('TYPES'):
                        save_var("TYPE", type)
                        history = process_hierarchy(input, type_history_ori, "RELATIONS", 3)
                        relation_history_ori = history.copy()
                        for relation in load_var("RELATIONS"):
                            save_var("RELATION", relation)
                            history = process_hierarchy(input, relation_history_ori, "ALIA_TEMPLATES", 4)
                            alias_history_ori = history.copy()
                            for alia in load_var("ALIAS_TEMPLATES"):
                                save_var("RELATION_ALIA_TEMPLATE", alia)
                                history = process_hierarchy(input, alias_history_ori, "TAILS", 5)
                                tails_history_ori = history.copy()
                                for tail in load_var("TAILS"):
                                    save_var("TAIL", tail)
                                    history = process_hierarchy(input, tails_history_ori, "EXIT", 6)
                            yield f"├{'─' * 3} majority voting result\n"
                            result = vote_tail()
                            yield f"├{'─' * 3} result {result}\n"
                            save_var("FINAL_TAIL", result)

    return Response(generate(), content_type="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8841, debug=True)
