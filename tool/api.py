import json
import os
import random
from tool.utils import save_var, load_var, engine
from tqdm import tqdm
# from tool.utils import *
from bson import ObjectId
import spacy
from data_object import *

wiki_template = json.load(open("./data/wiki_relation_templates.json"))
nlp = spacy.load("en_core_web_sm")
from_wiki = True
from_page = False


def get_sentences():
    if from_wiki:
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        data_dir = current_file_path + '/data'
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
                    if nlp(e.text)[0].pos_ != "PRON":
                        all_entity.append(e.text)
                    else:
                        continue
                    if e.entity:
                        entities_names.append(e.text)
                        entity_type_relation_constrains = {}
                        types = e.entity.types
                        if not types:
                            continue
                        for type in types[0]:
                            relations = type.asHeadConstraint
                            entity_type_relation_constrains[type.text] = [t.text for t in relations]
                        wiki_entity[e.text] = entity_type_relation_constrains
                    else:
                        continue
                if wiki_entity:
                    break
        save_var("SENTENCES", sentence.text)
        save_var("ENTITIES_as_head", wiki_entity)
        save_var("ALL_ENTITIES", list(set(all_entity)))
    elif from_page:
        while True:
            sentence_id = None
            sentence = BaseSentence.objects.get(id=ObjectId(sentence_id))
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
                if e.entity:
                    entities_names.append(e.text)
                    entity_type_relation_constrains = {}
                    types = e.entity.types
                    if not types:
                        continue
                    for type in types[0]:
                        relations = type.asHeadConstraint
                        entity_type_relation_constrains[type.text] = [t.text for t in relations]
                    wiki_entity[e.text] = entity_type_relation_constrains
                else:
                    continue
            if wiki_entity:
                break

    return "[Return] SENTENCES=" + "\"" + sentence.text + "\""


def get_entities():
    # todo: get entity by GLM
    return "[Return] ENTITIES=" + "\"" + str(load_var("ALL_ENTITIES")) + "\""


def choose_an_entity():
    # todo: get random entity
    head_entity = load_var("HEAD")
    return "[Return] HEAD=" + "\"" + head_entity + "\""


def get_types():
    entity = load_var("ENTITIES_as_head")
    head = load_var("HEAD")
    types = list(entity[head].keys())
    save_var("TYPES", types)
    return "[Return] TYPES=\"" + str(types) + "\""


def choose_a_type():
    type = load_var("TYPE")
    return "[Return] TYPE=\"" + type + "\""


def get_relations():
    entity = load_var("ENTITIES_as_head")
    head = load_var("HEAD")
    type = load_var("TYPE")
    relations = entity[head][type][:5]
    save_var("RELATIONS", relations)

    return "[Return] RELATIONS=\"" + str(relations) + "\""


def choose_a_relation():
    relation = load_var("RELATION")
    return "[Return] RELATION=\"" + relation + "\""


def choose_an_relation():
    relation = load_var("RELATION")
    return "[Return] RELATION=\"" + relation + "\""


def get_alia_templates():
    alias_template = wiki_template.get(load_var("RELATION"))
    save_var("ALIAS_TEMPLATES", alias_template)
    return "[Return] ALIA_TEMPLATES=" + str(alias_template)
#
#
# def get_alia_templates2():
#     relation = BaseRelation.objects.get(text=load_var("RELATION"))
#     alias = relation.alias
#     descript = relation.description
#     save_var("ALIAS", alias)
#     save_var("DESCRIPT", descript)
#     relations = alias + [load_var("RELATION")]
#     # todo: i should make relation alias tempaltes here, use which model, i use chatgpt for now.
#     system_prompt = "Response Format: \n" \
#                     "Your respond must be a dict format, for example: \n" \
#                     "    {\n" \
#                     "      'alias1':'template1'\n" \
#                     "      'alias2':'template2'\n" \
#                     "      'alias3':'template3'\n" \
#                     "    }" \
#                     "\n" \
#                     "Ensure the response can be parsed by Python eval().\n"
#     # todo: only consider 5 at most
#     s = {
#         "relations": relations[:5],
#         "relation": load_var("RELATION"),
#         "descript": descript
#     }
#     user_prompt = "The description of \"{s[relation]}\" is: \"{s[descript]}.\"\n" \
#                   "Make alias template for me with the given relation alias: {s[relations]}.\n"
#     example_prompt = "For example: given relation alias [\"country of citizenship\",\"subject of (country)\",\"subject of\",\"citizenship\",\"citizen of\",\"national of\"], you should give me relation alias templates as following:\n" \
#                      "{" \
#                      "\"country of citizenship\": \"Given the sentence: '{s[sentence]}', determine the country of citizenship for {s[head]}.\"," \
#                      "\"subject of\": \"Given the sentence: '{s[sentence]}', identify the country for which {s[head]} is a subject.\"," \
#                      "\"citizenship\": \"Given the sentence: '{s[sentence]}', determine the citizenship of {s[head]}.\"," \
#                      "\"citizen of\": \"Given the sentence: '{s[sentence]}', identify the country of which {s[head]} is a citizen.\"," \
#                      "\"national of\": \"Given the sentence: '{s[sentence]}', determine the nationality of {s[head]}.\"," \
#                      "}\n" \
#                      "the keys are relation alias, and the values are templates. You should give me relation alias template according to the relation description.\n" \
#                      "{s[head]} must appear in the template, do not have {s[tail]} or {s[object]} in the template."
#     user_prompt = user_prompt.format(s=s)
#     user_prompt += example_prompt
#     message = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]
#     while True:
#         # alias_template = make_chat_request_with_thinking(message, make_chat_request)
#         alias_template = make_chat_request(message)
#         # print(alias_template)
#         try:
#             print(alias_template)
#             alias_template = eval(alias_template['choices'][0]['message']['content'])
#         except:
#             print("try again")
#             continue
#         finally:
#             for id in invalid_keys:
#                 ori_keys[id] = False
#             # 将更改后的数据写回到 JSON 文件中
#             with open("data/keys.json", 'w') as file:
#                 json.dump(ori_keys, file, indent=4)
#         flag = 0
#         for k, v in alias_template.items():
#             if "s[head]" not in v:
#                 print("try again")
#                 break
#             flag = 1
#         if flag:
#             break
#     save_var("ALIAS_TEMPLATES", alias_template)
#
#     return "[Return] ALIA_TEMPLATES=" + str(alias_template)


def choose_an_alia_template():
    alias_template = load_var("RELATION_ALIA_TEMPLATE")
    head = load_var("HEAD")
    tails = load_var("ALL_ENTITIES")
    tails.remove(head)
    alia_template_ = list(alias_template.values())[0].format(s={
        "head": head,
        "tail_choose": tails,
        "sentence": load_var("SENTENCES")
    })
    return "[Return] ALIA_TEMPLATE=" + str(alias_template) + "\n" + "[Thought] " + alia_template_ + "\n" + "[Action] 【get_tails()】"


# def get_tails():
#     alia_template = load_var("RELATION_ALIA_TEMPLATE")
#     head = load_var("HEAD")
#     tails = load_var("ALL_ENTITIES")
#     tails.remove(head)
#     alia_template = list(alia_template.valuse())[0].format(s={
#         "head": head,
#         "tail_choose": tails,
#         "sentence": load_var("SENTENCES")
#     })
#     return "[API_Thought] " + alia_template


def choose_a_tail():
    tail = load_var("TAIL")
    return "[Return] TAIL=\"" + tail + "\""


def choose_an_tail():
    tail = load_var("TAIL")
    return "[Return] TAIL=\"" + tail + "\""


def search_engine():
    tail = load_var("TAIL")
    context = engine((load_var("HEAD"), tail))
    save_var("CONTEXT", context)
    a = "[Thought] Use the CONTEXT to verify the fact." + "\n" + "[Action] It is correct that {s[head]} {s[relation]} {s[tail]}? ANSWER: yes or no."
    head = load_var("HEAD")
    tail = load_var("TAIL")
    relation = load_var("RELATION")
    a = a.format(s={
        "head": head,
        "tail": tail,
        "relation": relation
    })
    return "[Return] CONTEXT=\"" + context + "\"" + "\n" + a


def exit():
    return "[Return] EXIT"


#
# def get_all_relation_template():
#     system_prompt = "Response Format: \n" \
#                     "Your respond must be a dict format, for example: \n" \
#                     "    {\n" \
#                     "      'alias1':'template1'\n" \
#                     "      'alias2':'template2'\n" \
#                     "      'alias3':'template3'\n" \
#                     "    }" \
#                     "\n" \
#                     "Ensure the response can be parsed by Python eval().\n" \
#                     "{s[head]} must appear in the template."
#     user_prompt = "The description of \"{s[relation]}\" is: \"{s[descript]}.\"\n" \
#                   "Make alias template for me with the given relation alias: {s[relations]}.\n"
#     example_prompt = "For example: given relation alias [\"country of citizenship\",\"subject of\",\"subject of\",\"citizenship\",\"citizen of\",\"national of\"], you should give me relation alias templates as following:\n" \
#                      "{" \
#                      "\"country of citizenship\": \"Given the sentence: '{s[sentence]}', determine the country of citizenship for {s[head]}.\"," \
#                      "\"subject of\": \"Given the sentence: '{s[sentence]}', identify the country for which {s[head]} is a subject.\"," \
#                      "\"citizenship\": \"Given the sentence: '{s[sentence]}', determine the citizenship of {s[head]}.\"," \
#                      "\"citizen of\": \"Given the sentence: '{s[sentence]}', identify the country of which {s[head]} is a citizen.\"," \
#                      "\"national of\": \"Given the sentence: '{s[sentence]}', determine the nationality of {s[head]}.\"," \
#                      "}\n" \
#                      "the keys are relation alias, and the values are templates. You should give me relation alias template according to the relation description, but do not use the description words.\n"
#     relations = BaseRelation.objects()
#     relaiton_dict = {}
#     for relation in relations:
#         relaiton_dict[relation.text] = relation.alias
#     json.dump(relaiton_dict, open("./relation_alias.json", "w"), indent=4)
#     try:
#         already_relaiton_alias = json.load(open("./already_relation_alias.json"))
#     except:
#         already_relaiton_alias = []
#     try:
#         relation_alias_template = json.load(open("./wiki_relation_alias_template.json"))
#     except:
#         relation_alias_template = []
#     try:
#         unsolve_alias_template = json.load(open("./unsolve_alias.json"))
#     except:
#         unsolve_alias_template = []
#
#     for relation in tqdm(relations):
#         print(f"======={relation.text}===========")
#         relation_text = relation.text
#         if relation_text in already_relaiton_alias:
#             print(f"{relation_text} already done")
#             continue
#         relation_alias = relation.alias
#         relation_descript = relation.description
#         all_relation = [relation_text] + relation_alias
#         s = {
#             "relations": all_relation[:20],
#             "relation": relation_text,
#             "descript": relation_descript
#         }
#         user_prompt_ = user_prompt.format(s=s)
#         user_prompt_ += example_prompt
#         message = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt_}
#         ]
#         count = 0
#         while True:
#             if count == 3:
#                 if relation_text not in unsolve_alias_template:
#                     unsolve_alias_template.append(relation_text)
#                 with open("./unsolve_alias.json", "w") as f:
#                     json.dump(unsolve_alias_template, f, indent=4)
#                 break
#             alias_template = make_chat_request(message)
#             try:
#                 alias_template = eval(alias_template['choices'][0]['message']['content'])
#                 print(alias_template)
#             except:
#                 count += 1
#                 print("eval error, try again")
#                 continue
#             finally:
#                 for id in invalid_keys:
#                     ori_keys[id] = False
#                 with open("data/keys.json", 'w') as file:
#                     json.dump(ori_keys, file, indent=4)
#             flag = 0
#             for k, v in alias_template.items():
#                 if "s[head]" not in v:
#                     count += 1
#                     print("no head, try again")
#                     break
#                 flag = 1
#             if flag:
#                 relation_alias_template.append({
#                     relation_text: alias_template
#                 })
#                 with open("./wiki_relation_alias_template.json", "w") as f:
#                     json.dump(relation_alias_template, f, indent=4)
#
#                 already_relaiton_alias.append(relation_text)
#                 with open("./already_relation_alias.json", "w") as out:
#                     json.dump(already_relaiton_alias, out, indent=4)
#                 break


# if __name__ == '__main__':
#     get_all_relation_template()
    # print("SENTENCE:", get_sentences())
    # print("ENTITIES:", get_entities())
    # print("choose an entity:", choose_an_entity())
    # print("TYPES:", get_types())
    # print("choose a type of entity:", choose_a_type())
    # print("RELATIONS:", get_relations())
    # print("choose a relation:", choose_a_relation())
    # print("RELATIONS_ALIAS_TEMPLATE:", get_alias_template())
    # print("choose a alia", choose_an_alia_template())
    # print("TAILS:", get_tails())
    # print("SEARCH ENGINE:", search_engine())
    # print("VF:", verify())
