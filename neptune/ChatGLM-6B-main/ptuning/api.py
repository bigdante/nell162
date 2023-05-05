import os
import random
from utils import *
from template import *
from data_object import *
from bson import ObjectId
import spacy

# 加载spaCy英语模型
nlp = spacy.load("en_core_web_sm")


def get_sentences():
    '''
        get a wikipedia sentence from neptune dataset.
        get entities in the sentence
        get relations constrains of entities
        get relation alias
        make fact extractor templates
    :return:
    '''
    data_dir = 'data'
    filename = 'wiki_sentence_id.json'
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        os.makedirs(data_dir, exist_ok=True)
        with open(filepath, "w") as f:
            for id, s in enumerate(BaseSentence.objects()):
                if len(s.text.split(" ")) < 10:
                    continue
                print(id, s.id)
                f.write(json.dumps(str(s.id)) + "\n")

    with open(filepath, 'r') as f:
        random_ids = f.readlines()
        while True:
            random_id = json.loads(random.sample(random_ids, 1)[0])
            sentence = BaseSentence.objects.get(id=ObjectId(random_id))
            if len(sentence.text.split(" ")) < 10:
                continue
            entities = sentence.mentions
            entities_names = []
            wiki_entity = {}
            all_entity = []
            for e in entities:
                # skip pron words
                if nlp(e.text)[0].pos_ != "PRON":
                    all_entity.append(e.text)
                else:
                    continue
                # check if e is in wikipedia entities list
                if e.entity:
                    entities_names.append(e.text)
                    # entities_wiki_entity.append(e.entity)
                    entity_type_relation_constrains = {}
                    types = e.entity.types
                    # if this wikipedia entity has no types, skip this one
                    if not types:
                        continue
                    for type in types[0]:
                        relations = type.asHeadConstraint
                        entity_type_relation_constrains[type.text] = [t.text for t in relations]
                    wiki_entity[e.text] = entity_type_relation_constrains
                else:
                    continue
                    # print(f"{e.text} is not in wikipedia entity")
            if wiki_entity:
                break
        save_var("SENTENCES", sentence.text)
        save_var("ENTITIES_as_head", wiki_entity)
        save_var("ALL_ENTITIES", list(set(all_entity)))

    return sentence.text


def get_entities():
    # todo: get entity by GLM
    return load_var("ALL_ENTITIES")


def choose_an_entity():
    # todo: get random entity
    head_entity = load_var("ENTITIES_as_head")
    head_entity = random.sample(head_entity.keys(), 1)[0]
    save_var("HEAD", head_entity)
    return head_entity


def get_types():
    entity = load_var("ENTITIES_as_head")
    head = load_var("HEAD")
    types = list(entity[head].keys())
    save_var("TYPES", types)
    return types


def choose_a_type():
    types = load_var("TYPES")
    type = random.sample(types, 1)[0]
    save_var("TYPE", type)
    return type


def get_relations():
    # todo: get all possible relations
    entity = load_var("ENTITIES_as_head")
    head = load_var("HEAD")
    type = load_var("TYPE")
    relations = entity[head][type]
    save_var("RELATIONS", relations)

    return relations


def get_a_relation():
    relations = load_var("RELATIONS")
    relation = random.sample(relations, 1)[0]
    save_var("RELATION", relation)

    return relation


def get_relation_alias_template():
    relation = BaseRelation.objects.get(text=load_var("RELATION"))
    alias = relation.alias
    descript = relation.description
    save_var("ALIAS", alias)
    save_var("DESCRIPT", descript)
    # todo: i should make relation alias tempaltes here, use which model

    return descript + "\n" + str(alias)


def get_tail():
    pass


def search_engine():
    # context = call_es((load_var("HEAD")['h_name'], load_var("tail")))
    context = "sentences......."
    return context


def verify():
    # todo: vf by vf model
    global fact
    if fact:
        fact = 0
        return True
    else:
        return False


def exit():
    return "EXIT"


if __name__ == '__main__':
    print("SENTENCE:", get_sentences())
    print("ENTITIES:", get_entities())
    print("choose an entity:", choose_an_entity())
    print("TYPES:", get_types())
    print("choose a type of entity:", choose_a_type())
    print("RELATIONS:", get_relations())
    print("choose a relation:", get_a_relation())
    print("RELATIONS_ALIAS_TEMPLATE:", get_relation_alias_template())
