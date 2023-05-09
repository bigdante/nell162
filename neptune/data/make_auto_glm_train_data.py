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
        # 只返回2个para，减小句子的长度
        return " ".join([r['_source']['text'].strip() for r in s_r[:2]])


main_template = "[Thought] To be a fact extractor, I need to start by retrieving sentences from Wikipedia.\n" \
                  "[Action] 【get_sentences()】\n" \
                  "[Return] SENTENCES=\"{s[sentence]}\"\n" \
                  "\n" \
                  "[Thought] After obtaining sentences from Wikipedia, it's time to identify all ENTITIES in the SENTENCES.\n" \
                  "[Action] 【get_entities(sentences=SENTENCES)】\n" \
                  "[Return] ENTITIES=\"{s[entities]}\"\n" \
                  "\n" \
                  "[Thought] With all ENTITIES found, I must select an entity from ENTITIES as the head.\n" \
                  "[Action] 【choose_an_entity(entities=ENTITIES)】\n" \
                  "[Return] HEAD=\"{s[head]}\"\n" \
                  "\n" \
                  "[Thought] Having selected a HEAD from ENTITIES, I need to determine what TYPES the HEAD may have.\n" \
                  "[Action] 【get_types(head=HEAD)】\n" \
                  "[Return] TYPES=\"{s[types]}\"\n" \
                  "\n" \
                  "[Thought] After getting HEAD TYPES, it's time to choose a TYPE from TYPES.\n" \
                  "[Action] 【choose_a_type(types=TYPES)】\n" \
                  "[Return] TYPE=\"{s[type]}\"\n" \
                  "\n" \
                  "[Thought] With a TYPE selected from TYPES, I must discover what relations HEAD may have according to TYPE.\n" \
                  "[Action] 【get_relations(head=HEAD,type=TYPE)】\n" \
                  "[Return] RELATIONS=\"{s[relations]}\"\n" \
                  "\n" \
                  "[Thought] Having obtained all RELATIONS, I need to pick a relation from RELATIONS.\n" \
                  "[Action] 【choose_a_relation(relations=RELATIONS)】\n" \
                  "[Return] RELATION=\"{s[relation]}\"\n" \
                  "\n" \
                  "[Thought] With a RELATION chosen, I must attempt to get RELATION_ALIAS_TEMPLATE.\n" \
                  "[Action] 【get_relation_alias_template(relation=RELATION)】\n" \
                  "[Return] RELATION_ALIAS_TEMPLATE={s[relation_template]}\n" \
                  "\n" \
                  "[Thought] After obtaining RELATION_ALIAS_TEMPLATE, I need to use RELATION_ALIAS_TEMPLATE to get TAILS.\n" \
                  "[Action] 【get_tail(sentence=SENTENCE, relation_alias_template=RELATION_ALIAS_TEMPLATE, head=HEAD)】\n" \
                  "[Return] TAILS=\"{s[tails]}\"\n" \
                  "\n" \
                  "[Thought] After get all TAILS, I ought to choose a tail from TAILS.\n" \
                  "[Action] 【choose_a_tail(tails=TAILS)】\n" \
                  "[Return] TAIL=\"{s[tail]}\"\n" \
                  "\n" \
                  "[Thought] When a TAIL acquired, I can search for more information about HEAD and TAIL using a search engine.\n" \
                  "[Action] 【search_engine(head=HEAD, tail=TAIL)】\n" \
                  "[Return] CONTEXT=\"{s[context]}\"\n" \
                  "\n" \
                  "[Thought] After obtaining the CONTEXT, I can utilize the information to verify the fact.\n" \
                  "[Action] 【verify(context=CONTEXT, head=HEAD, tail=TAIL)】\n" \
                  "[Return] LABEL=\"{s[label]}\"\n" \
                  "\n" \
                  "[Thought] After verifying the fact, I should exit this system.\n" \
                  "[Action]【exit()】\n" \
                  "[Return] EXIT"
#
# main_template = "[T] Begin, retrieve sentences.\n" \
#                 "[A]【get_sentences()】\n" \
#                 "[R] SENTENCE={s[sentence]}\n" \
#                 "\n" \
#                 "[T] Got sentence, identify ENTITIES in the SENTENCE.\n" \
#                 "[A]【get_entities()】\n" \
#                 "[R] ENTITIES={s[entities]}\n" \
#                 "\n" \
#                 "[T] With ENTITIES found, iterate ENTITIES as HEAD.\n" \
#                 "[A]【iterate_entity(ENTITIES)】\n" \
#                 "[R] HEAD={s[head]}\n" \
#                 "\n" \
#                 "[T] Chosen a HEAD, determine TYPES of HEAD.\n" \
#                 "[A]【get_types(HEAD)】\n" \
#                 "[R] TYPES={s[types]}\n" \
#                 "\n" \
#                 "[T] Obtained HEAD TYPES, traverse TYPES.\n" \
#                 "{s[type_template]}" \
#                 "\n" \
#                 "[T] TYPES traverse done."
#
# type_template = "[A] ${s[type_index]}【iterate_types()】\n" \
#                 "[R] ${s[type_index]} TYPE={s[type]}\n" \
#                 "\n" \
#                 "[T] ${s[type_index]} Fetch a TYPE, discover relations HEAD has.\n" \
#                 "[A] ${s[type_index]}【get_relations()】\n" \
#                 "[R] ${s[type_index]} RELATIONS={s[relations]}\n" \
#                 "\n" \
#                 "[T] ${s[type_index]} Acquired all RELATIONS, loop trough RELATIONS.\n" \
#                 "{s[relation_template]}" \
#                 "\n" \
#                 "[T] ${s[type_index]} RELATIONS loop trough done."
#
# relation_template = "[A] &{s[relation_index]}【choose_a_relation()】\n" \
#                     "[R] &{s[relation_index]} RELATION={s[relation]}\n" \
#                     "\n" \
#                     "[T] &{s[relation_index]} Received a RELATION, get ALIAS_TEMPLATE.\n" \
#                     "[A] &{s[relation_index]}【alias_template()】\n" \
#                     "[R] &{s[relation_index]} ALIAS_TEMPLATE={s[relation_template]}\n" \
#                     "\n" \
#                     "[T] &{s[relation_index]} Extracted ALIAS_TEMPLATES, run through ALIAS_TEMPLATES.\n" \
#                     "{s[alias_template]}" \
#                     "[T] &{s[relation_index]} Alias run through done. Now vote.\n" \
#                     "{s[vote_template]}" \
#                     "[T] @{s[relation_index]} TAILS cycle through done.\n"
#
# alias_template = "[A] @{s[alias_index]}【get_a_alias()】\n" \
#                  "[R] @{s[alias_index]} ALIA=\"{s[alia]}\"\n" \
#                  "\n" \
#                  "[T] @{s[alias_index]} ALIA acquired, check TAILS in the SENTENCE.\n" \
#                  "[A] @{s[alias_index]}【check_tails()】\n" \
#                  "[R] @{s[alias_index]} TAILS={s[tails]}\n" \
#                  "\n"
#
# vote_template = "[R] &{s[relation_index]}【vote()】\n" \
#                 "[A] &{s[relation_index]} {s[vote]}\n" \
#                 "\n" \
#                 "[T] &{s[relation_index]} Voted TAILS, cycle through the TAILS.\n" \
#                 "{s[engine_template]}"
#
# engine_template = "[A] #{s[tail_index]}【choose_a_tail()】\n" \
#                   "[R] #{s[tail_index]} TAIL={s[tail]}\n" \
#                   "\n" \
#                   "[T] #{s[tail_index]} Grasped a tail, search engine.\n" \
#                   "[A] #{s[tail_index]}【search_engine()】\n" \
#                   "[R] #{s[tail_index]} SEARCH_INFO={s[context]}\n" \
#                   "\n" \
#                   "[T] #{s[tail_index]} utilize the SEARCH_INFO, finally verify the TAIL.\n" \
#                   "[A] #{s[tail_index]}【verify()】\n" \
#                   "[R] #{s[tail_index]} LABEL={s[label]}\n" \
#                   "\n"

rel_info = json.load(open("../data/base_data_for_all/rel_info.json"))
relation_constrains = json.load(open("../data/base_data_for_all/head_relation_tail_constrain_dev.json"))
data = json.load(open("../data/base_data_for_all/dev_revised.json"))


# to make the sentence smaller, sentence 3, types 3
def make_data():
    with open(f"./train_auto_glm_data/{type}/ori.json", "w") as f:
        for sample in tqdm(data):
            labels = sample['labels']
            main_s = {}
            sentence = " ".join([" ".join(sent) for sent in [s for s in sample['sents'][:3]]])
            main_s['sentence'] = sentence
            entities = []
            for entity in sample['vertexSet']:
                entity_names = max([e['name'] for e in entity], key=len)
                entities.append(entity_names)
            main_s['entities'] = entities[:5]
            for l in labels:
                head_name = max([h['name'] for h in sample['vertexSet'][l['h']]], key=len)
                head_type = sorted(list(set([h['type'] for h in sample['vertexSet'][l['h']]])))
                main_s['types'] = head_type[:3]
                main_s["type"] = random.sample(head_type, 1)[0]
                tail_name = max([t['name'] for t in sample['vertexSet'][l['t']]], key=len)
                main_s['head'] = head_name
                main_s['tails'] = [tail_name] + random.sample(main_s['entities'], 1)
                main_s['tail'] = tail_name
                main_s['relations'] = get_relations(head_type)[:2]
                relation = random.choice(main_s['relations'])
                main_s['relation'] = relation
                relation_template = {k: save_gen_template[relation][k] for k in random.sample(save_gen_template[relation].keys(), 2)}
                main_s["relation_template"] = relation_template
                main_s['context'] = call_es((main_s['head'], main_s['tail']))
                main_s['label'] = random.choice(["yes", "no"])
                s = main_template.format(s=main_s)
                f.write(json.dumps(s) + "\n")


# def make_iter_data():
#     with open(f"./train_auto_glm_data/{type}/ori.json", "w") as f:
#         for sample in tqdm(data):
#             labels = sample['labels']
#             entities = []
#             for entity in sample['vertexSet']:
#                 entity_names = max([e['name'] for e in entity], key=len)
#                 entities.append(entity_names)
#             for l in labels:
#                 main_s = {}
#                 head_name = max([h['name'] for h in sample['vertexSet'][l['h']]], key=len)
#                 head_types = sorted(list(set([h['type'] for h in sample['vertexSet'][l['h']]])))
#                 head_ids = sorted(list(set([h['sent_id'] for h in sample['vertexSet'][l['h']]])))
#                 tail_name = max([t['name'] for t in sample['vertexSet'][l['t']]], key=len)
#                 tail_ids = sorted(list(set([t['sent_id'] for t in sample['vertexSet'][l['t']]])))
#                 sent_ids = sorted(list(head_ids + tail_ids))
#                 sentence = " ".join([" ".join(sent) for sent in [s for index, s in enumerate(sample['sents']) if index in sent_ids]])
#                 main_s['sentence'] = sentence
#                 main_s['entities'] = entities
#                 main_s['types'] = head_types
#                 main_s['head'] = head_name
#                 types = ""
#                 for h_id, h_type in enumerate(head_types):
#                     type_s = {}
#                     relations = get_relations(head_types)
#                     rn = relations.copy()
#                     rn.remove(rel_info[l['r']])
#                     type_s["type"] = h_type
#                     relations = [rel_info[l['r']]] + random.sample(rn, 1) if rn else [rel_info[l['r']]]
#                     type_s['relations'] = relations
#                     type_s['type_index'] = len(head_types) - h_id - 1
#                     relation_templates = ""
#                     for r_id, relation in enumerate(relations):
#                         relation_s = {"relation": relation, 'relation_index': len(relations) - r_id - 1}
#                         r_t = {k: save_gen_template[relation][k] for k in random.sample(save_gen_template[relation].keys(), 2)}
#                         relation_s['relation_template'] = str(r_t)
#                         alias_templates = ""
#                         vote = []
#                         for a_id, alia in enumerate(r_t):
#                             alias_s = {"alias_index": len(r_t) - a_id - 1, 'alia': str(alia)}
#                             en = entities.copy()
#                             en.remove(tail_name)
#                             alias_s['tails'] = [tail_name] + random.sample(en, min(5, len(en))) if en else [tail_name]
#                             vote.append(alias_s['tails'])
#                             alias_templates += alias_template.format(s=alias_s)
#                         threshold = len(vote) * 0.8
#                         answer_count = Counter(answer for inner_list in vote for answer in set(inner_list))
#                         result = [answer for answer, count in answer_count.items() if count >= threshold]
#                         vote_s = {
#                             "vote": result,
#                             "relation_index": len(relations) - r_id - 1
#                         }
#                         engine_templates = ""
#                         for t_id, tail in enumerate(result):
#                             engine_s = {"tail_index": len(result) - t_id - 1, "context": call_es((main_s['head'], tail)).strip(), 'tail': tail,
#                                         "label": "True" if tail == tail_name else "False"}
#                             engine_templates += engine_template.format(s=engine_s)
#                         vote_s['engine_template'] = engine_templates
#                         vote_templates = vote_template.format(s=vote_s)
#                         relation_s['alias_template'] = alias_templates
#                         relation_s['vote_template'] = vote_templates
#                         relation_templates += relation_template.format(s=relation_s)
#                     type_s['relation_template'] = relation_templates
#                     types += type_template.format(s=type_s)
#                 main_s["type_template"] = types
#                 s = main_template.format(s=main_s)
#                 f.write(json.dumps(s) + "\n")
#                 # print(s)
#                 # lines = s.strip().split("\n")
#                 # for line in lines:
#                 #     line = line.replace("$", "|-").replace("&", "|----").replace("@", "|------").replace("#", "|--------")
#                 #     print(line)
#                 #
#                 # exit()


def split_json_file(input_file_path, train_output_file_path, valid_output_file_path, train_ratio=0.8):
    with open(input_file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    num_train = int(len(data) * train_ratio)
    # train_data = data[:num_train]
    # valid_data = data[num_train:]
    train_data = data[:200000]
    valid_data = data[200000:240000]
    # valid_data = [d for d in data if d not in train_data]

    with open(train_output_file_path, 'w') as f:
        for d in tqdm(train_data, desc="train"):
            f.write(json.dumps(d) + '\n')

    with open(valid_output_file_path, 'w') as f:
        for d in tqdm(valid_data, desc="valid"):
            f.write(json.dumps(d) + '\n')


def make_rounds_thought_chat(input_file_path):
    with open(f"./train_auto_glm_data/{type}/rounds_chat.json", "w") as outfile:
        with open(input_file_path, 'r') as f:
            datas = [json.loads(line) for line in f]
            for data in tqdm(datas):
                blocks = data.strip().replace("\n\n", "\n").split('\n')
                dataset = []
                history = []
                for i, block in enumerate(blocks):
                    if i > len(blocks) - 3:
                        break
                    if "[Action]" in block:
                        continue
                    block_dict = {
                        "prompt": block if i == 0 else "what about next",
                        "response": blocks[i + 1],
                        "history": history.copy()
                    }
                    if "[Thought]" in blocks[i + 1]:
                        history.append([block_dict["prompt"], block_dict["response"]])
                    else:
                        history.append([block_dict["prompt"], block_dict["response"] + "\n" + blocks[i + 2]])
                    dataset.append(block_dict)

                for block in dataset:
                    json.dump(block, outfile)
                    outfile.write('\n')


def make_rounds_no_thought_chat(input_file_path):
    with open(f"./train_auto_glm_data/{type}/rounds_chat.json", "w") as outfile:
        with open(input_file_path, 'r') as f:
            datas = [json.loads(line) for line in f]
            for data in tqdm(datas):
                blocks = data.strip().split('\n\n')
                dataset = []
                for i, block in enumerate(blocks):
                    lines = block.split('\n')
                    history = []
                    for j in range(i):
                        prev_lines = blocks[j].split('\n')
                        history.append([prev_lines[0], prev_lines[1] + "\n" + prev_lines[2]])
                    try:
                        block_dict = {
                            "prompt": lines[0],
                            "response": lines[1],
                            "history": history
                        }
                    except:
                        pass
                    dataset.append(block_dict)

                for block in dataset:
                    json.dump(block, outfile)
                    outfile.write('\n')


if __name__ == '__main__':
    # type = "one_no_thought"
    type = "thought"
    make_iter_data()
    # if type == "one_thought":
    #     make_rounds_thought_chat(f"./train_auto_glm_data/{type}/ori.json")
    # else:
    #     make_rounds_no_thought_chat(f"./train_auto_glm_data/{type}/ori.json")
    #
    # split_json_file(f"./train_auto_glm_data/{type}/rounds_chat.json", f"../ChatGLM-6B-main/ptuning/auto_kg/{type}/train.json",
    #                 f"../ChatGLM-6B-main/ptuning/auto_kg/{type}/dev.json")
