import os
# from mogo_object import *
from flask import Flask, request, Response
import math
from flask_cors import CORS
from tool.utils import *
from bson import ObjectId
from tool.vicuna_inference import *

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)

program_start_time = datetime.datetime.strptime('2023-07-01 00:00:00', '%Y-%m-%d %H:%M:%S')
try:
    WikipediaEntity_total = BaseEntity.objects.count()
except:
    WikipediaEntity_total = 0
try:
    triple_total = TripleFact.objects.count()
except:
    triple_total = 0
relation_total = BaseRelation.objects.count()
relation_desc = json.load(open("./data/auto_kg/relation_list.json"))
relation_template = "Given the passage : \"{sentences}\", find all relations in the passage. The relations are :\n"
fact_template = "Given the passage : \"{sentences}\", and the relation description is : \"{description}\".\nNow, find all triplet fact that satisfy the relation description. " \
                "Triplet facts are :\n "
explanation_template = "The relation description is \"{description}\", and the given passage : \"{sentences}\"\nOne of triplet fact is {fact}. explain if it is right or wrong " \
                       "and give the reason. The explanation : "

# model, tokenizer = get_model()


@app.route('/dashboard', methods=['GET'])
def dashboard():
    '''
        show dashboard statistic
    :return:
    '''
    today = datetime.datetime.now()
    return output_process(
        {
            "all_triples": triple_total,
            "all_entities": WikipediaEntity_total,
            "all_relations": relation_total,
            'running_days': (today - program_start_time).days
        }
    )


@app.route('/latest', methods=['GET', 'POST'])
def latest():
    """
        return the latest triples extracted, show on the scroll bar
    """
    start = time.time()
    result = get_latest_triple()
    print("latest entity show done, consume time {:.2f} s".format(time.time() - start))
    return result


@app.route('/entity_table', methods=['GET', 'POST'])
def show_entity_table():
    """
    显示随机的entity列表
    :return:
    """
    entity_list = []
    start = time.time()
    params = get_params(request)
    params["size"] = 10
    pages = math.ceil(triple_total / params["size"])
    if params["refresh"]:
        params["page"] = random.randint(0, pages - 1)
    start_item = params["size"] * params["page"]
    end_item = params["size"] * (params["page"] + 1)
    filter_head = []
    for triple in TripleFact.objects()[start_item:end_item]:
        if triple.head not in filter_head:
            filter_head.append(triple.head)
            flag = 0
            for t in TripleFact.objects(head=triple.head):
                if t.tail != "unknown":
                    flag = 1
                    break
            if flag:
                entity_list.append({"id": str(triple.id), "text": triple.head})
    result_entity_table = {
        "data": entity_list,
        "pages": pages,
        "total": triple_total
    }
    print("hot entity show done, consume time {:.2f} s".format(time.time() - start))
    return result_entity_table


@app.route('/entity_detail', methods=['GET', 'POST'])
def entity_detail():
    """
        show triples taking entity as head
    :return:
    """
    result = []
    start = time.time()
    params = get_params(request)
    head = TripleFact.objects.get(id=ObjectId(params["id"])).head
    for _, triple in enumerate(TripleFact.objects(head=head)):
        if triple.tail == "unknown":
            continue
        r = precess_triple_data(triple, params["id"])
        r["head_unified"] = params["text"]
        hrt = r["head_entity"] + r["relation"] + r["tail_entity"]
        flag = 1
        for index, t in enumerate(result):
            if hrt == t["head_entity"] + t["relation"] + t["tail_entity"]:
                flag = 0
                result[index]["evidences"].append(r["evidences"][0])
                break
        if flag == 1:
            result.append(r)
    json.dump(result, open("./data/entity_table.json", "w"), indent=4)
    print("entity detail {} show done, consume time {:.2f} s".format(params["text"], time.time() - start))
    return output_process(result)


@app.route('/get_page_by_id', methods=['GET', 'POST'])
def get_page_by_id():
    '''
        get page by pageid
    :return:
    '''
    start = time.time()
    params = get_params(request)
    result = get_pages(params['inPageId'])
    json.dump(result, open("./data/page.json", "w"), indent=4)
    print("page show done, consume time {:.2f} s".format(time.time() - start))
    return output_process(result)


@app.route('/pps', methods=['GET', 'POST'])
def search_entity():
    '''
        get entity search results
    '''
    start = time.time()
    params = get_params(request)
    results = {"entity": sort_dict_by_key(call_es(params["query_name"]), 'tables')}
    print("search {} done, consume time {:.2f} s".format(params["query_name"], time.time() - start))
    json.dump(results, open("./data/entity_search_result.json", "w"), indent=4)
    return output_process(results)


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


@app.route('/sentence_extract', methods=['POST', "GET"])
def sentence_extract():
    """
        extract a sentence
    :return:
    """

    text = request.args.get('text', '')
    id = request.args.get('id', '')

    def generate():
        yield f"The sentence to be extract :\n {text}\n"
        yield f"{'=' * 200}\n"
        print(text)
        relaiton_prompt = relation_template.format(sentences=text)
        relations = eval(inference(model, tokenizer, relaiton_prompt))
        yield str(relations)
        for relation in relations:
            try:
                relation_description = relation_desc.get(relation)
            except:
                print(f"no such relation: {relation}")
                continue
            fact_prompt = fact_template.format(sentences=text, description=relation_description)
            try:
                facts = eval(inference(model, tokenizer, fact_prompt))

            except:
                print("eval error")
                continue
            entity_list = []
            unique_facts = list(set(tuple(fact) for fact in facts))
            unique_facts = [list(fact) for fact in unique_facts]
            for fact in unique_facts:
                headspan = find_word_positions(text, fact[0])
                tailspan = find_word_positions(text, fact[2])
                explanation_prompt = explanation_template.format(description=relation_description, sentences=text, fact=fact)
                explanation = inference(model, tokenizer, explanation_prompt)
                print(fact, "\n" + explanation)
                yield str(fact + "\n" + explanation)
                if fact[1] not in relation_desc:
                    continue
                triple_mogo = TripleFact(
                    head=fact[0],
                    relationLabel=fact[1],
                    tail=fact[2],
                    sentenceText=text,
                    headSpan=headspan[0] if headspan else [],
                    tailSpan=tailspan[0] if tailspan else [],
                    relation=BaseRelation.objects.get(text=fact[1]),
                    refsentence=BaseSentence.objects.get(id=id),
                    upVote=0,
                    downVote=0,
                    createtime=datetime.datetime.now(),
                    explanation=explanation
                )
                triple_mogo.save()
                for entity in list(set(entity_list)):
                    entity_mogo = BaseEntity(
                        text=entity,
                        sentenceText=text,
                        refsentence=BaseSentence.objects.get(id=id),
                        refpara=BaseSentence.objects.get(id=id).refpara,
                        refpage=BaseSentence.objects.get(id=id).refpage,
                    )
                    entity_mogo.save()

    return Response(generate(), content_type="text/event-stream")


@app.route('/page_extract', methods=['POST', "GET"])
def page_extract():
    """
        extract a neptune page
    :return:
    """
    pageid = request.args.get('id', '')

    def generate():
        page = BasePage.objects(id=pageid)
        for para in page.paragraphs:
            for sentence in para.sentences:
                text = sentence['text']
                if len(text.split(" ")) < 5:
                    continue
                print(text)
                if text == " ":
                    continue
                relaiton_prompt = relation_template.format(sentences=text)
                try:
                    relations = eval(inference(model, tokenizer, relaiton_prompt))
                    yield str(relations)
                except:
                    continue
                for relation in relations:
                    try:
                        relation_description = relation_desc.get(relation)
                    except:
                        print(f"no such relation {relation}")
                        continue
                    fact_prompt = fact_template.format(sentences=text, description=relation_description)
                    try:
                        facts = eval(inference(model, tokenizer, fact_prompt))
                    except:
                        continue
                    entity_list = []
                    unique_facts = list(set(tuple(fact) for fact in facts))
                    unique_facts = [list(fact) for fact in unique_facts]
                    for fact in unique_facts:
                        entity_list.extend([fact[0], fact[2]])
                        headspan = find_word_positions(text, fact[0])
                        tailspan = find_word_positions(text, fact[2])
                        explanation_prompt = explanation_template.format(description=relation_description, sentences=text, fact=fact)
                        explanation = inference(model, tokenizer, explanation_prompt)
                        print(fact, "\n" + explanation)
                        yield str(fact + "\n" + explanation)
                        if fact[1] not in relation_desc:
                            continue
                        triple_mogo = TripleFact(
                            head=fact[0],
                            relationLabel=fact[1],
                            tail=fact[2],
                            sentenceText=text,
                            headSpan=headspan[0] if headspan else [],
                            tailSpan=tailspan[0] if tailspan else [],
                            relation=BaseRelation.objects.get(text=fact[1]),
                            refsentence=sentence,
                            refpara=para,
                            refpage=page,
                            upVote=0,
                            downVote=0,
                            createtime=datetime.datetime.now(),
                            explanation=explanation
                        )
                        triple_mogo.save()
                    for entity in list(set(entity_list)):
                        entity_mogo = BaseEntity(
                            text=entity,
                            sentenceText=text,
                            refsentence=sentence,
                            refpara=para,
                            refpage=page,
                        )
                        entity_mogo.save()

    return Response(generate(), content_type="text/event-stream")


@app.route('/all_page', methods=['POST', "GET"])
def all_page():
    '''
        extract all page in neptune
    :return:
    '''
    for page in BasePage.objects():
        for para in page.paragraphs:
            for sentence in para.sentences:
                text = sentence['text']
                if len(text.split(" ")) < 5:
                    continue
                print(text)
                if text == " ":
                    continue
                relaiton_prompt = relation_template.format(sentences=text)
                try:
                    relations = eval(inference(model, tokenizer, relaiton_prompt))
                except:
                    continue
                for relation in relations:
                    try:
                        relation_description = relation_desc.get(relation)
                    except:
                        print(f"no such relation {relation}")
                        continue
                    fact_prompt = fact_template.format(sentences=text, description=relation_description)
                    try:
                        facts = eval(inference(model, tokenizer, fact_prompt))
                    except:
                        continue
                    entity_list = []
                    unique_facts = list(set(tuple(fact) for fact in facts))
                    unique_facts = [list(fact) for fact in unique_facts]
                    for fact in unique_facts:
                        if fact[1] not in relation_desc:
                            continue
                        entity_list.extend([fact[0], fact[2]])
                        headspan = find_word_positions(text, fact[0])
                        tailspan = find_word_positions(text, fact[2])
                        explanation_prompt = explanation_template.format(description=relation_description, sentences=text, fact=fact)
                        explanation = inference(model, tokenizer, explanation_prompt)
                        print(fact, "\n" + explanation)
                        if fact[1] not in relation_desc:
                            continue
                        triple_mogo = TripleFact(
                            head=fact[0],
                            relationLabel=fact[1],
                            tail=fact[2],
                            sentenceText=text,
                            headSpan=headspan[0] if headspan else [],
                            tailSpan=tailspan[0] if tailspan else [],
                            relation=BaseRelation.objects.get(text=fact[1]),
                            refsentence=sentence,
                            refpara=para,
                            refpage=page,
                            upVote=0,
                            downVote=0,
                            createtime=datetime.datetime.now(),
                            explanation=explanation
                        )
                        triple_mogo.save()
                    for entity in list(set(entity_list)):
                        entity_mogo = BaseEntity(
                            text=entity,
                            sentenceText=text,
                            refsentence=sentence,
                            refpara=para,
                            refpage=page,
                        )
                        entity_mogo.save()
    return "ok"


@app.route('/evaluation', methods=['POST', "GET"])
def evaluation():
    """
        for evaluation (vicuna model)
    :return:
    """
    nell_fact_total_right = 0
    nell_fact_total = 0
    out = open("./data/evaluation/record.txt", "w")
    redocred_test_data = json.load(open("./data/auto_kg/redocred/redocred_test_explanation_filtered.json"))
    tp = 0
    fp = 0
    wrong_list = []
    right_list = []
    total_fact = 0
    for sample in redocred_test_data:
        wrong = []
        right = []
        sentence = sample['passage']
        print("=" * 100)
        print(sentence)
        print("-" * 100)
        print((f"true fact: {sample['same_list']}"))
        print("-" * 100)
        total_fact += len(sample['same_list'])
        relaiton_prompt = relation_template.format(sentences=sentence)
        try:
            relations = eval(inference(model, tokenizer, relaiton_prompt))
        except:
            continue
        for relation in relations:
            try:
                relation_description = relation_desc.get(relation)
            except:
                print(f"no such relation {relation}")
                continue
            fact_prompt = fact_template.format(sentences=sentence, description=relation_description)
            try:
                facts = eval(inference(model, tokenizer, fact_prompt))
            except:
                continue
            unique_facts = list(set(tuple(fact) for fact in facts))
            unique_facts = [list(fact) for fact in unique_facts]
            fact_index = []
            for fact in unique_facts:
                if fact[2] == "unknown":
                    continue
                flag = 0
                if fact[1] not in relation_desc:
                    continue
                explanation_prompt = explanation_template.format(description=relation_description, sentences=sentence, fact=fact)
                explanation = inference(model, tokenizer, explanation_prompt)
                for index, true_fact in enumerate(sample['same_list']):
                    if fact in true_fact:
                        right.append({"fact": fact, "explanation": explanation})
                        print("right:", fact, explanation)
                        flag = 1
                        if index not in fact_index:
                            tp += 1
                            fact_index.append(index)
                        else:
                            print("overlap fact")
                        # del sample['same_list'][index]
                        break
                if not flag:
                    print("wrong:", fact, explanation)
                    fp += 1
                    wrong.append({"fact": fact, "explanation": explanation})
        if wrong:
            wrong_list.append({
                "text": sentence,
                "wrong_fact_list": wrong,
                "true_fact_list": sample['same_list']
            })
        if right:
            right_list.append({
                "text": sentence,
                "right_fact_list": right,
                "true_fact_list": sample['same_list']

            })
    json.dump(wrong_list, open("./data/evaluation/wrong_redocred.json", "w"), indent=4)
    json.dump(right_list, open("./data/evaluation/right_redocred.json", "w"), indent=4)
    out.write(f"vicuna, tp: {tp}, fp: {fp}, total_fact:{total_fact}, recall:{tp / total_fact}, precison:{tp / (tp + fp)}\n")
    print(f"vicuna, tp: {tp}, fp: {fp}, total_fact:{total_fact}, recall:{tp / total_fact}, precison:{tp / (tp + fp)}")
    wrong_list = []
    right_list = []
    for page in BasePage.objects()[:50]:
        for para in page.paragraphs:
            for sentence in para.sentences:
                print("=" * 100)
                wrong = []
                right = []
                text = sentence['text']
                print(text)
                if len(text.split(" ")) < 5:
                    continue
                if text == " ":
                    continue
                relaiton_prompt = relation_template.format(sentences=text)
                try:
                    relations = eval(inference(model, tokenizer, relaiton_prompt))
                except:
                    continue
                for relation in relations:
                    try:
                        relation_description = relation_desc.get(relation)
                    except:
                        print(f"no such relation {relation}")
                        continue
                    fact_prompt = fact_template.format(sentences=text, description=relation_description)
                    try:
                        facts = eval(inference(model, tokenizer, fact_prompt))
                    except:
                        continue
                    unique_facts = list(set(tuple(fact) for fact in facts))
                    unique_facts = [list(fact) for fact in unique_facts]
                    for fact in unique_facts:
                        if fact[2] == "unknown":
                            continue
                        nell_fact_total += 1
                        if fact[1] not in relation_desc:
                            continue
                        explanation_prompt = explanation_template.format(description=relation_description, sentences=text, fact=fact)
                        explanation = inference(model, tokenizer, explanation_prompt)
                        if fact[1] not in relation_desc:
                            continue
                        prompt = f"You are a fact checker.\n" \
                                 f"I have passage : \"{text}\"\n" \
                                 f"One possible fact in the passage is: \"{fact}\"\n" \
                                 f"The relation description is: \"{relation_desc.get(fact[1])}\"\n" \
                                 "According to the passage and relation description, Is the fact right? yor answer must be \"【right】\"or \"【wrong】\"."
                        message = [
                            {"role": "user", "content": prompt}
                        ]
                        check = make_chat_request(message)['choices'][0]['message']['content']
                        if check.lower == "【right】" or check.lower == "right" or "【right】" in check or "【Right】" in check or "Right" in check:
                            print(f"right: {fact}, {explanation}")
                            right.append({"fact": fact, "explanation": explanation})
                            nell_fact_total_right += 1
                        elif check.lower == "【wrong】" or check.lower == "wrong" or "【wrong】" in check or "【Wrong】" in check or "Wrong" in check or "wrong" in check:
                            print(f"wrong: {fact}, {explanation}")
                            wrong.append({"fact": fact, "explanation": explanation})
                        else:
                            print("nono", check)
                if wrong:
                    wrong_list.append({
                        "text": text,
                        "fact_list": wrong
                    })
                if right:
                    right_list.append({
                        "text": sentence,
                        "fact_list": right
                    })
    if invalid_keys:
        for invalid_key in invalid_keys:
            ori_keys.get(invalid_key)['label'] = False
        json.dump(ori_keys, open("../auto-kg/keys.json", "w"), indent=4)
    json.dump(wrong_list, open("./data/evaluation/wrong_neptune.json", "w"), indent=4)
    json.dump(right_list, open("./data/evaluation/right_neptune.json", "w"), indent=4)
    out.write(f"nell_fact_total:{nell_fact_total}, nell_fact_total_right:{nell_fact_total_right}\n")
    print(nell_fact_total_right, nell_fact_total)
    return "ok"


@app.route('/chatglm_inference', methods=['POST', "GET"])
def chatglm_inference():
    """
        for evaluation (chatglm model)
    :return:
    """
    import torch
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
    )
    finetune_type = "full_finetune"
    model_name_or_path = "./ChatGLM2/ptuning/THUDM/chatglm2-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if finetune_type == "full_finetune":
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half()
    else:
        ptuning_checkpoint = "./ChatGLM2/ptuning/ckpt/checkpoint-4000"
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config.pre_seq_len = 64
        config.prefix_projection = False
        model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True).half()
        prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    cuda_device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    model.to(cuda_device)
    model = model.eval()
    relation_template = "Given the passage : \"{sentences}\", find all relations in the passage. The relations are :\n"
    fact_template = "Given the passage : \"{sentences}\", and the relation description is : \"{description}\".\n" \
                    "Now, find all triplet fact that satisfy the relation description. Triplet facts are :\n"
    explanation_template = "The relation description is \"{description}\", and the given passage : \"{sentences}\"\n" \
                           "One of triplet fact is {fact}. explain if it is right or wrong and give the reason. The explanation :"
    nell_fact_total_right = 0
    nell_fact_total = 0
    out = open("./data/evaluation/record.txt", "w")
    f = open("./data/evaluation/wrong_neptune.json", "w")
    wrong_list = []
    for page in BasePage.objects()[:50]:
        for para in page.paragraphs:
            for sentence in para.sentences:
                wrong = []
                text = sentence['text']
                if len(text.split(" ")) < 5:
                    continue
                if text == " ":
                    continue
                relaiton_prompt = relation_template.format(sentences=text)
                try:
                    relations = eval(model.chat(tokenizer, relaiton_prompt))
                except:
                    continue
                for relation in relations:
                    try:
                        relation_description = relation_desc.get(relation)
                    except:
                        print(f"no such relation {relation}")
                        continue
                    fact_prompt = fact_template.format(sentences=text, description=relation_description)
                    try:
                        facts = eval(model.chat(tokenizer, fact_prompt))
                    except:
                        continue
                    nell_fact_total += len(facts)
                    unique_facts = list(set(tuple(fact) for fact in facts))
                    unique_facts = [list(fact) for fact in unique_facts]
                    for fact in unique_facts:
                        if fact[1] not in relation_desc:
                            continue
                        explanation_prompt = explanation_template.format(description=relation_description, sentences=text, fact=fact)
                        explanation = model.chat(tokenizer, explanation_prompt)
                        if fact[1] not in relation_desc:
                            continue
                        prompt = f"You are a fact checker.\n" \
                                 f"I have passage : \"{text}\"\n" \
                                 f"One possible fact in the passage is: \"{fact}\"\n" \
                                 f"The relation description is: \"{relation_desc.get(fact[1])}\"\n" \
                                 "According to the passage and relation description, Is the fact right? yor answer must be \"【right】\"or \"【wrong】\"."
                        message = [
                            {"role": "user", "content": prompt}
                        ]
                        check = make_chat_request(message)['choices'][0]['message']['content']
                        if check.lower == "【right】" or check.lower == "right" or "【right】" in check or "【Right】" in check or "Right" in check:
                            nell_fact_total_right += 1
                        elif check.lower == "【wrong】" or check.lower == "wrong" or "【wrong】" in check or "【Wrong】" in check or "Wrong" in check:
                            wrong.append({"fact": fact, "explanation": explanation})
                        else:
                            print("nono", check)
                if wrong:
                    wrong_list.append({
                        "text": text,
                        "fact_list": wrong
                    })
    if invalid_keys:
        for invalid_key in invalid_keys:
            ori_keys.get(invalid_key)['label'] = False
        json.dump(ori_keys, open("../auto-kg/keys.json", "w"), indent=4)
    json.dump(wrong_list, f, indent=4)
    out.write(f"nell_fact_total:{nell_fact_total}\n")
    out.write(f"nell_fact_total_right:{nell_fact_total_right}\n")
    print(nell_fact_total_right, nell_fact_total)
    redocred_test_data = json.load(open("./data/auto_kg/redocred/redocred_test_explanation_filtered.json"))
    out = open("record.txt", "w")
    tp = 0
    fp = 0
    wrong_list = []
    for sample in redocred_test_data:
        wrong = []
        sentence = sample['passage']
        relaiton_prompt = relation_template.format(sentences=sentence)
        try:
            relations = eval(model.chat(tokenizer, relaiton_prompt))
        except:
            continue
        for relation in relations:
            try:
                relation_description = relation_desc.get(relation)
            except:
                print(f"no such relation {relation}")
                continue
            fact_prompt = fact_template.format(sentences=sentence, description=relation_description)
            try:
                facts = eval(model.chat(tokenizer, fact_prompt))
            except:
                continue
            unique_facts = list(set(tuple(fact) for fact in facts))
            unique_facts = [list(fact) for fact in unique_facts]
            for fact in unique_facts:
                flag = 0
                if fact[1] not in relation_desc:
                    continue
                explanation_prompt = explanation_template.format(description=relation_description, sentences=sentence, fact=fact)
                explanation = model.chat(tokenizer, explanation_prompt)
                for index, true_fact in enumerate(sample['same_list']):
                    if fact in true_fact:
                        print("right", fact, "\n" + explanation)
                        tp += 1
                        flag = 1
                        del sample['same_list'][index]
                        break
                if fact[2] != "unknown" and flag == 0:
                    print("wrong", fact, "\n" + explanation)
                    fp += 1
                    wrong.append({"fact": fact, "explanation": explanation})
        wrong_list.append({
            "text": sentence,
            "fact_list": wrong
        })
    f = open("./wrong_redocred.json", "w")
    json.dump(wrong_list, f, indent=4)
    out.write(f"vicuna, tp: {tp}, fp: {fp}")
    print(f"vicuna, tp: {tp}, fp: {fp}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8841)
