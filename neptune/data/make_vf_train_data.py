import json
import random
import os
import sys

sys.path.append("..")
from requests import post
from tqdm import tqdm
from chatgpt_gen_tail.template import *
from utils_tools.get_ground_truth import *

entity_relation_tail = json.load(open("./base_data_for_all/head_relation_tail_constrain_train.json"))
relation_list = ['country of citizenship', 'date of birth', 'place of birth', 'participant of',
                 'located in the administrative territorial entity', 'contains administrative territorial entity',
                 'participant', 'location', 'followed by', 'country', 'educated at', 'date of death', 'sibling',
                 'head of government', 'legislative body', 'conflict', 'applies to jurisdiction', 'instance of',
                 'performer', 'publication date', 'creator', 'author', 'composer', 'lyrics by', 'member of',
                 'notable work', 'inception', 'part of', 'cast member', 'director', 'has part', 'production company',
                 'owned by', 'headquarters location', 'developer', 'manufacturer', 'country of origin', 'publisher',
                 'parent organization', 'subsidiary', 'capital of', 'capital', 'spouse', 'father', 'child', 'religion',
                 'mother', 'located in or next to body of water', 'located on terrain feature', 'basin country',
                 'member of political party', 'mouth of the watercourse', 'place of death', 'military branch',
                 'work location', 'start time', 'award received', 'point in time', 'founded by', 'employer',
                 'head of state', 'member of sports team', 'league', 'present in work', 'position held', 'chairperson',
                 'languages spoken, written or signed', 'location of formation', 'operator', 'producer', 'record label',
                 'follows', 'replaced by', 'replaces', 'end time', 'subclass of', 'residence', 'sister city',
                 'original network', 'ethnic group', 'separated from', 'screenwriter', 'continent', 'platform',
                 'product or material produced', 'genre', 'series', 'narrative location', 'parent taxon',
                 'original language of work', 'dissolved, abolished or demolished', 'territory claimed by',
                 'characters', 'influenced by', 'official language'
                 ]

relation_dict = {
    'country of citizenship': 0.15,
    'date of birth': 0.5,
    'located in the administrative territorial entity': 0.1,
    "contains administrative territorial entity": 0.15,
    "followed by": 0.5,
    "country": 0.12,
    "participant": 0.5,
    "date of death": 0.5,
    "participant of": 0.5,
    "member of": 0.3,
    "has part": 0.2,
    "headquarters location": 0.15,
    "capital of": 0.2,
    "capital": 0.2,
    "basin country": 0.2,
    "start time": 0.2,
    "follows": 0.2,
    "replaced by": 0.2,
    "replaces": 0.2,
    "influenced by": 0.5,
    "mouth of the watercourse": 0.3,
    "located on terrain feature": 0.3,
    "religion": 0.5,
    "point in time": 0.5,
    "founded by": 0.5,
    "part of": 0.2,
    'notable work': 0.2,
    "applies to jurisdiction": 0.5,
    "inception": 0.5,
    "located in or next to body of water": 0.5,
    "performer": 0.5,
    "owned by": 0.5,
    "child": 0.5,
    "official language": 0.5,
}


def call_es(head, tail, mode="para", match="should"):
    headers = {'Content-Type': 'application/json'}
    url_para = 'http://166.111.7.106:9200/wikipedia_paragraph/wikipedia_paragraph/_search'
    url_sentence = 'http://166.111.7.106:9200/wikipedia_sentence/wikipedia_sentence/_search'
    url = {"sentence": url_sentence, "para": url_para}
    data = {
        "query": {"bool": {match: [{"match": {"text": head}}, {"match": {"text": tail}}]}}
    }
    with post(url=url[mode], headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf8'), auth=("nekol", "kegGER123")) as resp:
        results = resp.json()
        s_r = results['hits']['hits']
        s_r_filter = [r['_source']['text'] for r in s_r[:5]] if mode == "sentence" else [r['_source']['text'] for r in s_r[:3]]
    if s_r_filter:
        return " ".join(s_r_filter)
    else:
        return None


def cal_ori_data(path, augment_trigger_num=700):
    all_data = {}
    augment_relation = []
    for relation in relation_list:
        with open(f"{path}/{relation}.json", "r") as f:
            data = json.load(f)
        y, n, _, _ = count_yes_no(data)
        if y < augment_trigger_num:
            augment_relation.append(relation)
        all_data[relation] = {"yes": y, "no": n}
    for k in ["yes", "no"]:
        print(f"======{k}=========")
        for relation in relation_list:
            print(all_data[relation][k])
    return augment_relation


def shrink_data(version):
    os.makedirs(f"./train_vf_model_data/{version}/shrink_data", exist_ok=True)
    for relation in relation_list:
        data = json.load(open(f"./train_vf_model_data/{version}/ori_data/{relation}.json"))
        no_data = [d for d in data if d["label"] == "no"]
        yes_data = [d for d in data if d["label"] == "yes"]
        try:
            keep_data = yes_data + random.sample(no_data, int(relation_dict[relation] * len(no_data)))
            print(int(relation_dict[relation] * len(no_data)))
        except:
            keep_data = yes_data + no_data
            print(len(no_data))
        keep_data = sorted(keep_data, key=lambda x: x['uuid'])
        json.dump(keep_data, open(f"./train_vf_model_data/{version}/shrink_data/{relation}.json", "w"), indent=4)


def process_json_files(folder_path):
    def traverse_folder(folder_path):
        files = os.listdir(folder_path)
        json_files = []
        for file in files:
            if file.endswith('.json') and file != "all.json":
                json_files.append(os.path.join(folder_path, file))
        return json_files

    def split_dataset(data, split_ratio=0.8):
        split_index = int(len(data) * split_ratio)
        train_data = data[:split_index]
        test_data = data[split_index:]
        return train_data, test_data

    file_paths = traverse_folder(folder_path)
    all_train_data = []
    all_test_data = []
    data_ratio = {}
    max = 0
    for file_path in tqdm(file_paths):
        print(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
            train_data, test_data = split_dataset(data, split_ratio=0.8)
            all_train_data.extend(train_data)
            all_test_data.extend(test_data)
            yes, no, _, _ = count_yes_no(data)
            data_ratio[file_path.split("/")[-1].split(".")[0]] = {"all": len(train_data + test_data), "yes": yes, "no": no}
            max = len(train_data + test_data) if len(train_data + test_data) > max else max
    all_data = all_train_data + all_test_data
    # all_count = len(all_data)
    for d in all_data:
        d["all_ratio"] = max / data_ratio[d['relation']]['all']
        d["no/yes"] = data_ratio[d['relation']]['no'] / data_ratio[d['relation']]['yes'] if d['label'] == "yes" else 1
        d["final_loss_weigh"] = d["all_ratio"] * d["no/yes"]
    with open(folder_path + "/all.json", 'w') as f:
        json.dump(all_data, f, indent=4)


def get_text_by_sent_id(h=None, t=None, sample=None, ids=None, evidence_ids=None):
    if evidence_ids:
        sent_ids = sorted(list(set([i['sent_id'] for i in h] + [i['sent_id'] for i in t] + evidence_ids)))
    else:
        if not ids:
            sent_ids = sorted(list(set([i['sent_id'] for i in h] + [i['sent_id'] for i in t])))
        else:
            sent_ids = sorted(list(set(ids)))
    text = process_data(" ".join([" ".join(sample["sents"][i]) for i in sent_ids]))
    return text, sent_ids


def count_yes_no(data):
    yes_count = sum([1 for d in data if d['label'] == 'yes'])
    no_count = sum([1 for d in data if d['label'] == 'no'])
    total_count = len(data)
    yes_ratio = yes_count / total_count
    no_ratio = no_count / total_count
    return yes_count, no_count, yes_ratio, no_ratio


def make_vf_train_data(relation, augment_relations):
    data = json.load(open("./base_data_for_all/train_revised.json"))
    result_save_path = f"./train_vf_model_data/v1/ori_data"
    os.makedirs(result_save_path, exist_ok=True)
    uuid = 0
    data_save = []
    for doc_id, sample in enumerate(tqdm(data, desc=f"{relation} {relation_list.index(relation)}")):
        h_t = get_ground_tail_truth(sample)
        flag = 0
        for r in sample['labels']:
            r_name = rel_info[r['r']]
            if r_name != relation:
                continue
            else:
                flag = 1
            h_names = sorted(list(set([h['name'] for h in sample['vertexSet'][r['h']]])))
            t_names = sorted(list(set([t['name'] for t in sample['vertexSet'][r['t']]])))
            for h_name in h_names:
                text, _ = get_text_by_sent_id(sample['vertexSet'][r['h']], sample['vertexSet'][r['t']], sample)
                for t_name in t_names:
                    v = {
                        "uuid": uuid,
                        "sentence": text,
                        "head": h_name,
                        "relation": r_name,
                        "tail": t_name,
                        "label": "yes",
                    }
                    data_save.append(v)
                    uuid += 1
                    if relation in augment_relations:
                        text_list = []
                        for mode in ["sentence", "para"]:
                            for match in ["should", "must"]:
                                text = call_es(h_name, t_name, mode=mode, match=match)
                                if text:
                                    text_list.append(text)
                        for text in list(set(text_list)):
                            v = {
                                "uuid": uuid,
                                "sentence": text,
                                "head": h_name,
                                "relation": relation,
                                "tail": t_name,
                                "label": "yes",
                                "es": True
                            }
                            data_save.append(v)
                            uuid += 1
        if flag != 0:
            for h_index, vertex_h in enumerate(sample['vertexSet']):
                h_names = sorted(list(set([h['name'] for h in vertex_h])))
                h_type = str(sorted(list(set([h['type'] for h in vertex_h]))))
                try:
                    tail_constrains = entity_relation_tail[h_type][relation]
                except:
                    continue
                for t_index, vertex_t in enumerate(sample['vertexSet']):
                    if vertex_t == vertex_h:
                        continue
                    t_type = str(sorted(list(set([t['type'] for t in vertex_t]))))
                    if t_type not in tail_constrains or t_index in h_t.get(str(h_index) + "_" + relation, []):
                        continue
                    t_names = list(set([t['name'] for t in vertex_t]))
                    for h_name in h_names:
                        text, _ = get_text_by_sent_id(vertex_h, vertex_t, sample)
                        for t_name in t_names:
                            v = {
                                "uuid": uuid,
                                "sentence": text,
                                "head": h_name,
                                "relation": relation,
                                "tail": t_name,
                                "label": "no",
                            }
                            data_save.append(v)
                            uuid += 1
                            if relation in augment_relations:
                                text = call_es(h_name, t_name, mode="para", match="must")
                                if not text:
                                    continue
                                else:
                                    v = {
                                        "uuid": uuid,
                                        "sentence": text,
                                        "head": h_name,
                                        "relation": relation,
                                        "tail": t_name,
                                        "label": "no",
                                        "es": True
                                    }
                                    data_save.append(v)
                                    uuid += 1
    json.dump(data_save, open(result_save_path + "/" + relation + ".json", "w"), indent=4)


# def furthuer_augment(version):
#     os.makedirs(f"./train_vf_model_data/{version}/shrink_data_augment/", exist_ok=True)
#     augment_relation = cal_ori_data(path=f"./train_vf_model_data/{version}/shrink_data/", augment_trigger_num=500)
#     for relation in tqdm(relation_list):
#         data = json.load(open(f"./train_vf_model_data/{version}/shrink_data/{relation}.json"))
#         if relation in augment_relation:
#             data_save = []
#             uuid = 0
#             for d in tqdm(data, desc=f"{relation} {relation_list.index(relation)}"):
#                 d['uuid'] = uuid
#                 data_save.append(d)
#                 h_name = d['head']
#                 t_name = d['tail']
#                 if d['label'] == "yes":
#                     text_list = []
#                     for mode in ["sentence", "para"]:
#                         for match in ["should", "must"]:
#                             text = call_es(t_name, h_name, mode=mode, match=match)
#                             if text:
#                                 text_list.append(text)
#                     for text in list(set(text_list)):
#                         uuid += 1
#                         v = {
#                             "uuid": uuid,
#                             "sentence": text,
#                             "head": h_name,
#                             "relation": relation,
#                             "tail": t_name,
#                             "label": "yes",
#                             "es": True
#                         }
#                         data_save.append(v)
#                 else:
#                     text = call_es(t_name, h_name, mode="para", match="should")
#                     if not text:
#                         continue
#                     else:
#                         uuid += 1
#                         v = {
#                             "uuid": uuid,
#                             "sentence": text,
#                             "head": h_name,
#                             "relation": relation,
#                             "tail": t_name,
#                             "label": "no",
#                             "es": True
#                         }
#                         data_save.append(v)
#             json.dump(data_save, open(f"./train_vf_model_data/{version}/shrink_data_augment/{relation}.json", "w"), indent=4)
#         else:
#             json.dump(data, open(f"./train_vf_model_data/{version}/shrink_data_augment/{relation}.json", "w"), indent=4)


if __name__ == '__main__':
    # augment_relation = cal_ori_data(path="./train_vf_model_data/v0")
    # for relaiton in relation_list[:]:
    #     make_vf_train_data(relaiton, augment_relation)
    # cal_ori_data(path="./train_vf_model_data/v1/ori_data")
    shrink_data(version="v1")
    # furthuer_augment(version="v1")
    cal_ori_data(path="./train_vf_model_data/v1/shrink_data")
    process_json_files('train_vf_model_data/v1/shrink_data')
