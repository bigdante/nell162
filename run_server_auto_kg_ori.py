import json
from flask import Flask, request, Response
import math
from flask_cors import CORS
from tool.utils import *
from bson import ObjectId
from fuzzywuzzy import fuzz
from tool.chat_gpt_prompt import *
from tool.vicuna_inference import *

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)

# 程序开始运行的时间点
program_start_time = datetime.datetime.strptime('2022-12-01 00:00:00', '%Y-%m-%d %H:%M:%S')

# 记录总的条数，防止每次统计耗时太长
# WikipediaEntity_total = WikipediaEntity.objects.count()
# triple_total = TripleFact.objects.count()
# relation_total = BaseRelation.objects.count()
WikipediaEntity_total = 11827927
triple_total = 22958671
relation_total = 2400

with open('./data/wikidata/entity_id_wiki_triple.json', 'r') as f:
    '''
        randomly select a wiki entity which has triple in neptune
    '''
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
    save_result_json(result, "./tool/data/page_id.json")
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


@app.route('/re', methods=['POST', "GET"])
def re():
    text = request.args.get('text', '')
    id = request.args.get('id', '')

    def generate():
        yield f"The sentence to be extract :\n {text}\n"
        yield f"{'=' * 200}\n"
        while True:
            try:
                entities = get_entities(text)
                break
            except:
                continue
        yield f"The entities in the sentence are:\n {entities}\n"
        if len(entities) > 1:
            yield f"{'=' * 200}\n"
            yield f"The given relations are:\n {relation_list}\n"
            yield f"{'=' * 200}\n"
            relations = get_relations(text, entities)
            if type(relations) == str:
                yield relations
            else:
                yield f"before checking..........\n"
                if "no answer" not in relations:
                    for r in relations:
                        yield str(r) + "\n"
                yield f"{'=' * 200}\n"
                if "no answer" not in relations or 'network error' in relations:
                    yield f"now checking..........\n"
                    results = []
                    for index, relation in enumerate(relations):
                        relation_label = get_relation_label(relation[1])
                        try:
                            descript = BaseRelation.objects.get(text=relation_label)
                        except:
                            print(f"{relation_label} not in database")
                            continue
                        prompt = f"Given the sentence: '{text}', and the relation '{relation[1]}' description is : {descript}.\n" \
                                 f"Is fact '{relation[0]} {relation[1]} {relation[2]}' ?\n"
                        sys_prompt = "You are a fact checker. You should use your power to help me carefully check if the fact is correct.\n" \
                                     "You should make your decision considering the relation description given by user.\n" \
                                     f"If correct, you must output: 【CORRECT】\n" \
                                     f"If wrong, you must output: 【WRONG】\n" \
                                     f"And give me wrong reason and if you know the correct answer, output true fact as:\n" \
                                     f"MODIFIED: ('head','relation','tail')"
                        result = process_one_data((prompt, sys_prompt), fact_check=True)
                        print("check.....", result)
                        if "【CORRECT】" in result:
                            yield str(str(relation) + " \n correct checking: " + result) + "\n"
                            results.append(relation)
                        else:
                            yield str(str(relation) + " \n wrong checking: " + result) + "\n"
                            try:
                                vf_fact = extract_triple_fact(result)
                                if type(vf_fact) == tuple:
                                    results.append(vf_fact)
                                print("vf_fact", vf_fact)
                            except:
                                continue
                        yield f"************************************ the {index + 1} fact verified ************************************\n"
                    print(results)
                    if results:
                        # for r in results:
                        #     if r[1] in relation_list and r[0] in entities and r[2] in entities:
                        #         satisfied_result.append(r)
                        satisfied_result = results
                        if satisfied_result:
                            yield f"{'=' * 200}\n"
                            yield "all fact checked done.\n"
                            yield f"The relations between entities are:\n"
                            for r in satisfied_result:
                                yield str(r) + "\n"
                            yield f"{'=' * 200}\n"
                            yield f"saving {len(satisfied_result)} new fact to dataset.......\n"
                            for relation in tqdm(satisfied_result, desc="saving"):
                                new_fact = TripleFact_new(
                                    head=relation[0],
                                    relationLabel=relation[1],
                                    tail=relation[2],
                                    headSpan=find_word_indices(text, relation[0]),
                                    relation=BaseRelation.objects.get(text=get_relation_label(relation[1])),
                                    tailSpan=find_word_indices(text, relation[2]),
                                    evidence=BaseSentence.objects.get(id=id),
                                    evidenceText=text,
                                    is_from_abstract='from chatgpt',
                                    upVote=1,
                                    downVote=0,
                                    isNewFact=1,
                                    timestamp=datetime.datetime.now(),
                                )
                                new_fact.save()
                                print(f"{relation} save: ", new_fact.id)
                            yield f"save {len(satisfied_result)} new fact done.......\n"
        else:
            yield "only one entity"

    return Response(generate(), content_type="text/event-stream")


model, tokenizer = get_model()
relation_desc = json.load(open("./data/wikidata/relation_desc.json"))


@app.route('/page_extract2', methods=['POST', "GET"])
def page_extract2():
    template_list = ["Identify entities in the sentence: \"{text}\"\n",
                     "Discover relations the entities may have according to the types.\n",
                     "Find relations between entities.\n",
                     ]
    vf_template = ["Check if {text} is correct, search information about HEAD and TAIL.\n",
                   "The \"{relation}\" description is : {desc}.\n",
                   "According to the information above, it is correct that {text} ? ANSWER: yes or no.\n",
                   ]
    pageid = request.args.get('id', '')
    try:
        result = json.load(open("./data/wikidata/text_extract_history.json"))
    except:
        result = []

    def generate():
        for para in WikipediaPage.objects.get(id=ObjectId(pageid)).paragraphs:
            for sentence in para.sentences:
                yield "=" * 80 + "\n"
                text = sentence['text']
                yield f"The sentence to be extract :\n {text}\n"
                gen_history = ""
                for index, template in enumerate(template_list):
                    prompt = gen_history + template.format(text=text)
                    out = inference(model, tokenizer, prompt)
                    print(out)
                    if index == 0:
                        out = list(set(eval(out)))
                    yield str(out) + "\n"
                    if index == 0:
                        entities = out
                        if len(entities) == 1:
                            result.append({
                                "text": text,
                                "entity": entities,
                                "fact_analyze": "no fact"
                            })
                            yield "only one entity\n"
                            break
                    if index == 1:
                        relations = eval(out)
                    gen_history = prompt + str(out) + "\n"
                if len(entities) > 1:
                    fact_list = eval(out)
                    fact_history = []
                    yield "now checking........\n"
                    for fact in fact_list:
                        vf_history = ""
                        if fact[2] in fact[0] or fact[0] in fact[2] or fact[0] not in text or fact[2] not in text or fact[1] not in relations or is_pronoun(fact[0]) or is_pronoun(
                                fact[2]):
                            yield str(fact) + " wrong \n"
                            yield "-" * 80 + "\n"
                            continue
                        for id, template in enumerate(vf_template):
                            if id == 0:
                                prompt = vf_history + template.format(text=fact)
                            if id == 1:
                                prompt = vf_history + template.format(relation=fact[1], desc=relation_desc.get(fact[1]))
                            if id == 2:
                                prompt = vf_history + template.format(text=fact)
                            out = inference(model, tokenizer, prompt)
                            yield out + "\n"
                            if "【engine()】" in out:
                                out = "【engine()】\n" + engine((fact[0], fact[2]))
                                yield out
                            vf_history = prompt + out
                        if out == "True":
                            try:
                                r = BaseRelation.objects.get(text=get_relation_label(fact[1]))
                            except:
                                continue
                            new_fact = TripleFact(
                                head=fact[0],
                                relationLabel=fact[1],
                                tail=fact[2],
                                headSpan=find_word_indices(text, fact[0]),
                                relation=r,
                                tailSpan=find_word_indices(text, fact[2]),
                                evidence=BaseSentence.objects.get(id=sentence.id),
                                evidenceText=text,
                                is_from_abstract='from auto_kg',
                                # headWikidataEntity='example_wikidata_entity_reference',
                                # headWikipediaEntity='example_wikipedia_entity_reference',
                                upVote=1,
                                downVote=0,
                                isNewFact=1,
                                timestamp=datetime.datetime.now(),
                            )
                            new_fact.save()
                            yield f"saving {str(fact)}......\n"
                            yield "-" * 80 + "\n"
                            print(f"{fact} save: ", new_fact.id)
                        history = gen_history + "\n" + vf_history
                        fact_history.append({
                            "fact": fact,
                            "history": history
                        })
                    if not fact_history:
                        fact_history.append({
                            "fact": "no fact",
                            "history": gen_history
                        })
                    result.append({
                        "text": text,
                        "entities": entities,
                        "fact_analyze": fact_history
                    })
        json.dump(result, open("./data/wikidata/text_extract_history.json", "w"), indent=4)

    return Response(generate(), content_type="text/event-stream")


@app.route('/page_extract', methods=['POST', "GET"])
def page_extract():
    '''
        entity -> fact
    :return:
    '''
    template_list = ["Identify entities in the sentence: \"{text}\"\n",
                     "Find relations between entities.\n",
                     ]
    vf_template = ["With given sentence: {sentence}, we get \"{fact}\". To check if {fact} is correct, retrieve information.\n",
                   "The \"{relation}\" description is : {description}.\n",
                   "According to the information above, it is correct that \"{fact}\" ? ANSWER: yes or no.\n"]
    pageid = request.args.get('id', '')

    def generate():
        for para in WikipediaPage.objects.get(id=ObjectId(pageid)).paragraphs:
            # for page in WikipediaPage.objects():
            #     for para in page.paragraphs:
            for sentence in para.sentences:
                yield "=" * 80 + "\n"
                text = sentence['text']
                yield f"The sentence to be extract :\n {text}\n"
                gen_history = ""
                for index, template in enumerate(template_list):
                    prompt = gen_history + template.format(text=text)
                    out = inference(model, tokenizer, prompt)
                    print(out)
                    if index == 0:
                        out = parse_string_to_list(out)
                        if not out:
                            break
                    yield str(out) + "\n"
                    if index == 0:
                        entities = out
                        if len(entities) == 1:
                            yield "only one entity\n"
                            break
                    gen_history = prompt + str(out) + "\n"
                if len(entities) > 1 and entities:
                    try:
                        fact_list = eval(out)
                    except:
                        continue
                    yield "now checking........\n"
                    for fact in fact_list:
                        if len(fact) != 3:
                            continue
                        if fact[2] in fact[0] or fact[0] in fact[2] or fact[0] not in text or fact[2] not in text or is_pronoun(fact[0]) or is_pronoun(fact[2]):
                            yield str(fact) + " wrong \n"
                            yield "-" * 80 + "\n"
                            continue
                        else:
                            vf_template[1].format(relation=fact[1], description=relation_desc.get(fact[1]))
                            vf_template[2].format(fact=fact)
                            prompt = vf_template[0].format(sentence=text, fact=fact) + "\n" + engine((fact[0], fact[2])) + vf_template[1].format(
                                relation=fact[1], description=relation_desc.get(fact[1])) + vf_template[2].format(fact=fact)
                            out = inference(model, tokenizer, prompt)
                            yield out + "\n"
                        if out == "yes":
                            try:
                                r = BaseRelation.objects.get(text=get_relation_label(fact[1]))
                            except:
                                continue
                            new_fact = TripleFact(
                                head=fact[0],
                                relationLabel=fact[1],
                                tail=fact[2],
                                headSpan=find_word_indices(text, fact[0]),
                                relation=r,
                                tailSpan=find_word_indices(text, fact[2]),
                                evidence=BaseSentence.objects.get(id=sentence.id),
                                evidenceText=text,
                                is_from_abstract='from auto_kg',
                                upVote=1,
                                downVote=0,
                                isNewFact=1,
                                timestamp=datetime.datetime.now(),
                            )
                            new_fact.save()
                            yield f"saving {str(fact)}......\n"
                            yield "-" * 80 + "\n"
                            print(f"{fact} save: ", new_fact.id)

    return Response(generate(), content_type="text/event-stream")


@app.route('/compare', methods=['POST', "GET"])
def compare():
    result = json.load(open("./data/wikidata/text_extract_history.json"))
    os.remove("./data/wikidata/text_extract_history.json")
    return result


@app.route('/all_page', methods=['POST', "GET"])
def all_page():
    template_list = ["Identify entities in the sentence: \"{text}\"\n",
                     "Discover relations the entities may have according to the types.\n",
                     "Find relations between entities.\n",
                     ]
    vf_template = ["Check if {text} is correct, search information about HEAD and TAIL.\n",
                   "The \"{relation}\" description is : {desc}.\n",
                   "According to the information above, it is correct that {text} ? ANSWER: yes or no.\n",
                   ]

    for page in WikipediaPage.objects.get(id=ObjectId("6257b648c20df149acb22501")):
        # page = WikipediaPage.objects.get(id=ObjectId("6257b648c20df149acb22501"))
        for para in page.paragraphs:
            for sentence in para.sentences:
                text = sentence['text']
                print(text)
                if text == " ":
                    continue
                gen_history = ""
                for index, template in enumerate(template_list):
                    prompt = gen_history + template.format(text=text)
                    out = inference(model, tokenizer, prompt)
                    print(out)
                    if index == 0:
                        try:
                            out = list(set(eval(out)))
                            entities = out
                        except:
                            break
                        if len(out) == 1:
                            break
                    if index == 1:
                        relations = eval(out)
                    gen_history = prompt + str(out) + "\n"
                try:
                    fact_list = list(set(eval(out)))
                except:
                    continue
                # for relation in tqdm(satisfied_result, desc="saving"):
                #     new_fact = TripleFact(
                #         head=relation[0],
                #         relationLabel=relation[1],
                #         tail=relation[2],
                #         headSpan=find_word_indices(text, relation[0]),
                #         relation=BaseRelation.objects.get(text=get_relation_label(relation[1])),
                #         # relation=relation[1],
                #         tailSpan=find_word_indices(text, relation[2]),
                #         evidence=BaseSentence.objects.get(id=sentence.id),
                #         evidenceText=text,
                #         is_from_abstract='from auto_kg',
                #         # headWikidataEntity='example_wikidata_entity_reference',
                #         # headWikipediaEntity='example_wikipedia_entity_reference',
                #         upVote=1,
                #         downVote=0,
                #         isNewFact=1,
                #         timestamp=datetime.datetime.now(),
                #     )
                #     new_fact.save()
                #     print(f"{relation} save: ", new_fact.id)
                if len(entities) > 1:
                    for fact in fact_list:
                        vf_history = ""
                        print("fact:", fact)
                        if len(fact) != 3:
                            continue
                        if fact[2] in fact[0] or fact[0] in fact[2] or fact[0] not in text or fact[2] not in text or fact[1] not in relations:
                            continue
                        for id, template in enumerate(vf_template):
                            if id == 0:
                                prompt = vf_history + template.format(text=fact)
                                out = "【engine()】\n" + engine((fact[0], fact[2]))
                            if id == 1:
                                prompt = vf_history + template.format(relation=fact[1], desc=relation_desc.get(fact[1]))
                                out = ""
                            if id == 2:
                                prompt = vf_history + template.format(text=fact)
                                out = inference(model, tokenizer, prompt)
                                print("prompt: ", prompt)
                                print("out: ", out)
                                print("======================")
                            if out:
                                vf_history = prompt + out + "\n"
                        if out == "True":
                            try:
                                r = BaseRelation.objects.get(text=get_relation_label(fact[1]))
                            except:
                                continue
                            new_fact = TripleFact(
                                head=fact[0],
                                relationLabel=fact[1],
                                tail=fact[2],
                                headSpan=find_word_indices(text, fact[0]),
                                relation=r,
                                tailSpan=find_word_indices(text, fact[2]),
                                evidence=BaseSentence.objects.get(id=sentence.id),
                                evidenceText=text,
                                is_from_abstract='from auto_kg',
                                upVote=1,
                                downVote=0,
                                isNewFact=1,
                                timestamp=datetime.datetime.now(),
                            )
                            new_fact.save()
                            print(f"{fact} save: ", new_fact.id)

    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8841, debug=False)
