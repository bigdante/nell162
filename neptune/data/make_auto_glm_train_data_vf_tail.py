import random
from collections import Counter
from requests import post
from chatgpt_gen_tail.template import *
from tqdm import tqdm


def get_relations(h_type):
    relations = list(relation_constrains.get(str(h_type)).keys())
    return relations


def call_es(text, mode="para"):
    head, tail = text
    headers = {'Content-Type': 'application/json'}
    url_para = 'http://166.111.7.106:9200/wikipedia_paragraph/wikipedia_paragraph/_search'
    url_sentence = 'http://166.111.7.106:9200/wikipedia_sentence/wikipedia_sentence/_search'
    url = {"sentence": url_sentence, "para": url_para}
    data = {
        "query": {"bool": {"must": [{"match": {"text": head}}, {"match": {"text": tail}}]}}
    }
    with post(url=url[mode], headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf8'), auth=("nekol", "kegGER123")) as resp:
        s_r = resp.json()['hits']['hits']
        return " ".join([r['_source']['text'].strip() for r in s_r[:2]])


main_template = "[T] Begin, retrieve sentences.\n" \
                "[A]【get_sentences()】\n" \
                "[R] SENTENCE={s[sentence]}\n" \
                "\n" \
                "[T] Got sentence, identify ENTITIES in the SENTENCE.\n" \
                "[A]【get_entities()】\n" \
                "[R] ENTITIES={s[entities]}\n" \
                "\n" \
                "[T] With ENTITIES found, iterate ENTITIES as HEAD.\n" \
                "{s[head_template]}"

head_template = "[A] ^{s[head_id]}【choose_a_head()】\n" \
                "[R] ^{s[head_id]} HEAD={s[head]}\n" \
                "\n" \
                "[T] ^{s[head_id]} Chosen a HEAD, determine TYPES of HEAD.\n" \
                "[A] ^{s[head_id]}【get_types()】\n" \
                "[R] ^{s[head_id]} TYPES={s[types]}\n" \
                "\n" \
                "[T] ^{s[head_id]} Obtained HEAD TYPES, traverse TYPES.\n" \
                "{s[type_template]}"

type_template = "[A] ${s[type_index]}【choose_a_type()】\n" \
                "[R] ${s[type_index]} TYPE={s[type]}\n" \
                "\n" \
                "[T] ${s[type_index]} Fetch a TYPE, discover relations HEAD has.\n" \
                "[A] ${s[type_index]}【get_relations()】\n" \
                "[R] ${s[type_index]} RELATIONS={s[relations]}\n" \
                "\n" \
                "[T] ${s[type_index]} Acquired all RELATIONS, loop trough {s[num_relations]} RELATIONS.\n" \
                "{s[relation_template]}"

relation_template = "[A] &{s[relation_index]}【choose_a_relation()】\n" \
                    "[R] &{s[relation_index]} RELATION={s[relation]}\n" \
                    "\n" \
                    "[T] &{s[relation_index]} Now, I got TAILS={s[tails]}. I should abandon the tails retrieve history and verify {s[num_tail]} TAILS.\n" \
                    "{s[engine_template]}\n" \
                    "[T] &{s[relation_index]} All TAILS verification done.\n"

engine_template = "[A] #{s[tail_index]}【choose_a_tail()】\n" \
                  "[R] #{s[tail_index]} TAIL={s[tail]}\n" \
                  "\n" \
                  "[T] #{s[tail_index]} Grasped a tail, search engine.\n" \
                  "[A] #{s[tail_index]}【search_engine()】\n" \
                  "[R] #{s[tail_index]} SEARCH_INFO={s[context]}\n" \
                  "\n" \
                  "[T] #{s[tail_index]} utilize the SEARCH_INFO, finally verify the TAIL.\n" \
                  "[A] #{s[tail_index]}【verify()】\n" \
                  "[R] #{s[tail_index]} LABEL={s[label]}\n" \
                  "\n"

rel_info = json.load(open("../data/base_data_for_all/rel_info.json"))
relation_constrains = json.load(open("../data/base_data_for_all/head_relation_tail_constrain_dev.json"))
data = json.load(open("../data/base_data_for_all/dev_revised.json"))


def get_in_sentence_entity(sent_ids, vertexs):
    entities = []
    for v in vertexs:
        for vv in v:
            if vv['sent_id'] in sent_ids:
                entities.append(max([h['name'] for h in v], key=len))
                break
    return entities


def make_iter_data():
    with open(f"./train_auto_glm_data/{type}/ori_vf_tails.json", "w") as f:
        for sample in tqdm(data):
            labels = sample['labels']
            entities = []
            for entity in sample['vertexSet']:
                entity_names = max([e['name'] for e in entity], key=len)
                entities.append(entity_names)
            for l in labels:
                main_s = {}
                head_name = max([h['name'] for h in sample['vertexSet'][l['h']]], key=len)
                head_ids = sorted(list(set([h['sent_id'] for h in sample['vertexSet'][l['h']]])))
                tail_name = max([t['name'] for t in sample['vertexSet'][l['t']]], key=len)
                tail_ids = sorted(list(set([t['sent_id'] for t in sample['vertexSet'][l['t']]])))
                sent_ids = sorted(list(head_ids + tail_ids))
                sentence = " ".join([" ".join(sent) for sent in [s_ for index, s_ in enumerate(sample['sents']) if index in sent_ids]])
                main_s['sentence'] = sentence
                in_sentence_entity = get_in_sentence_entity(sent_ids, sample['vertexSet'])
                main_s['entities'] = in_sentence_entity
                heads = ""
                # 这里就考虑一个head，多个head，之后考虑
                for head_id, head in enumerate([head_name]):
                    # head_s = {"head_id": len(in_sentence_entity) - 1 - head_id, "head": head}
                    head_s = {"head_id": 0, "head": head}
                    head_types = sorted(list(set([h['type'] for h in sample['vertexSet'][entities.index(head)]])))
                    head_s['types'] = head_types
                    types = ""
                    for t_id, h_type in enumerate(head_types):
                        type_s = {"type": h_type}
                        relations = get_relations(head_types)
                        relations = list(set(random.sample(relations, min(2, len(relations))) + [rel_info[l['r']]]))
                        type_s['relations'] = relations
                        type_s['type_index'] = len(head_types) - t_id - 1
                        relation_templates = ""
                        for r_id, relation in enumerate(relations):
                            relation_s = {"relation": relation, 'relation_index': len(relations) - r_id - 1}
                            r_t = {k: save_gen_template[relation][k] for k in random.sample(list(save_gen_template[relation].keys()), 2)}
                            relation_s['relation_template'] = str(r_t)
                            vote = []
                            for a_id, alia in enumerate(r_t):
                                alias_s = {"alias_index": len(r_t) - a_id - 1, 'alia': str(alia)}
                                en = in_sentence_entity.copy()
                                en.remove(tail_name)
                                alias_s['tails'] = [tail_name] + random.sample(en, min(3, len(en))) if en else [tail_name]
                                vote.append(alias_s['tails'])
                            threshold = len(vote) * 0.8
                            answer_count = Counter(answer for inner_list in vote for answer in set(inner_list))
                            result = [answer for answer, count in answer_count.items() if count >= threshold]
                            relation_s['tails'] = result

                            engine_templates = ""
                            for t_id, tail in enumerate(result):
                                engine_s = {"tail_index": len(result) - t_id - 1, "context": call_es((head_s['head'], tail)).strip(), 'tail': tail,
                                            "label": "True" if tail == tail_name else "False"}
                                engine_templates += engine_template.format(s=engine_s)
                            relation_s['engine_template'] = engine_templates
                            relation_s['num_tail'] = len(result)
                            relation_templates += relation_template.format(s=relation_s)
                        type_s['relation_template'] = relation_templates
                        type_s['num_relations'] = len(relations)
                        types += type_template.format(s=type_s)
                    head_s['type_template'] = types
                    heads += head_template.format(s=head_s)
                main_s["head_template"] = heads
                s = main_template.format(s=main_s)
                f.write(json.dumps(s) + "\n")
                # print(s)
                # exit()


def split_json_file(input_file_path, train_output_file_path, valid_output_file_path, train_ratio=0.8):
    with open(input_file_path + "rounds_chat_gen_tails.json", 'r') as f:
        data1 = [json.loads(line) for line in f]
    with open(input_file_path + "rounds_chat_vf_tails.json", 'r') as f:
        data2 = [json.loads(line) for line in f]
    data = data1 + data2
    num_train = int(len(data) * train_ratio)
    train_data = data[:num_train]
    valid_data = data[num_train:]
    # train_data = data[:200000]
    # valid_data = data[200000:240000]
    # valid_data = [d for d in data if d not in train_data]

    with open(train_output_file_path, 'w') as f:
        for d in tqdm(train_data, desc="train"):
            f.write(json.dumps(d) + '\n')

    with open(valid_output_file_path, 'w') as f:
        for d in tqdm(valid_data, desc="valid"):
            f.write(json.dumps(d) + '\n')


def make_rounds_thought_chat(input_file_path):
    with open(f"./train_auto_glm_data/{type}/rounds_chat_vf_tails.json", "w") as outfile:
        with open(input_file_path, 'r') as f:
            datas = [json.loads(line) for line in f]
            for data in tqdm(datas):
                blocks = data.strip().replace("\n\n", "\n").split('\n')
                dataset = []
                history = []
                flag = 0
                for i, block in enumerate(blocks):
                    if i > len(blocks) - 3:
                        break
                    if "[A]" in block:
                        continue
                    block_dict = {
                        "prompt": block if i == 0 else "what about next",
                        "response": blocks[i + 1],
                        "history": history.copy()
                    }
                    if "[T]" in blocks[i + 1]:
                        history.append([block_dict["prompt"], block_dict["response"]])
                    else:
                        history.append([block_dict["prompt"], block_dict["response"] + "\n" + blocks[i + 2]])
                    if "I should abandon the tails retrieve history" in block:
                        flag = 1
                    if flag:
                        dataset.append(block_dict)

                for block in dataset:
                    json.dump(block, outfile)
                    outfile.write('\n')


if __name__ == '__main__':
    type = "step"
    make_iter_data()
    make_rounds_thought_chat(f"./train_auto_glm_data/{type}/ori_vf_tails.json")
    split_json_file(f"./train_auto_glm_data/{type}/", f"../ChatGLM-6B-main/ptuning/auto_kg/{type}/train.json",
                    f"../ChatGLM-6B-main/ptuning/auto_kg/{type}/dev.json")
