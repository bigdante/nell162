import random
from utils import *
from template import *


def get_a_sentence():
    # todo: get sentence by wikipedia
    docred_data = json.load(open("../../data/base_data_for_all/dev_revised.json"))
    one_doc = random.sample(docred_data, len(docred_data))[0]
    r = random.sample(one_doc['labels'], len(one_doc['labels']))[0]
    sentence, sent_ids = get_text_by_sent_id(one_doc['vertexSet'][r['h']], one_doc['vertexSet'][r['t']], one_doc)
    head_type = str(sorted(list(set([h['type'] for h in one_doc['vertexSet'][r['h']]]))))
    relations = head_relation_tail_constrain.get(head_type)
    t_names = sorted(list(set([t['name'] for t in one_doc['vertexSet'][r['t']]])))
    entity_list = []
    for vertex in one_doc['vertexSet']:
        for v in vertex:
            if v['sent_id'] in sent_ids:
                entity_list.append(v['name'])
                break
    entity_list = list(set(entity_list))
    head = {
        "sentence": sentence,
        "h_type": head_type,
        "h_name": one_doc['vertexSet'][r['h']][0]['name'],
        "false_relation": random.sample([key for key in relations if key != rel_info[r['r']]], 1),
        "true_relation": [rel_info[r['r']]],
        "true_tail": max(t_names, key=len),
        "false_tail": random.sample([n for n in entity_list if n != one_doc['vertexSet'][r['h']][0]['name']], 1)
    }
    save_var("HEAD", head)
    save_var("ENTITIES", entity_list)
    return sentence


def get_entities():
    # todo: get entity by GLM
    return load_var("ENTITIES")


def choose_an_entity_from_entity_list():
    # todo: get random entity
    entity = load_var("HEAD")
    return entity['h_name']


def get_relations():
    # todo: get all possible relations
    entity = load_var("HEAD")
    return entity['true_relation'] + entity["false_relation"]


flag = 1


def choose_an_relation():
    # todo: random relation
    global flag
    if flag:
        return load_var("HEAD")['true_relation']
    else:
        return load_var("HEAD")['false_relation']


def get_relation_alias_template():
    # todo
    global flag
    if flag:
        flag = 0
        alias_templates = save_gen_template[load_var("HEAD")['true_relation'][0]]
    else:
        alias_templates = save_gen_template[load_var("HEAD")['false_relation'][0]]
    relation_template = {k: alias_templates[k] for k in random.sample(list(alias_templates), 2)}
    return relation_template


tail_flag = 1


def get_tail():
    # todo: get tail by GLM
    global tail_flag
    if tail_flag:
        tail_flag = 0
        return load_var("HEAD")['true_tail']
    else:
        return load_var("HEAD")['false_tail']


def search_engine():
    # context = call_es((load_var("HEAD")['h_name'], load_var("tail")))
    context = "sentences......."
    return context


fact = 1


def verify():
    # todo: vf by vf model
    global fact
    if fact:
        fact = 0
        return True
    else:
        return False


if __name__ == '__main__':
    get_a_sentence()
    print("SENTENCE:", load_var("HEAD")['sentence'])
    print("ENTITIES:", get_entities())
    print("choose an entity:", choose_an_entity_from_entity_list())
    print("RELATIONS:", get_relations())
    for _ in range(len(get_relations())):
        print("     CHOOSE A RELATION:", choose_an_relation())
        print("     ALIAS TEMPLATEs:", get_relation_alias_template())
        for alias, template in get_relation_alias_template().items():
            s = {
                "head": load_var("HEAD")['h_name'],
                "sentence": load_var("HEAD")['sentence'],
                "tail_choose": load_var("ENTITIES"),
            }
            print("         THE query sentence:", template.format(s=s))
            save_var("QUERY", template.format(s=s))
            tail = get_tail()
            save_var("TAIL", tail)
            print("         THE tail predict is ", tail)
            print("         For further vf, using search engine, and the context: ", search_engine())
            print("         VF result: ", verify())
