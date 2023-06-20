import json
import os
from requests import post
from tqdm import tqdm
# from template import *
# import wikipedia

# rel_info = json.load(open("../../data/rel_info.json"))

#
# def get_wikipedia_summary_containing_head_tail(head, tail, lang='en'):
#     wikipedia.set_lang(lang)
#
#     search_query = f"{head} {tail}"
#     search_results = wikipedia.search(search_query)
#
#     for result in search_results:
#         try:
#             page = wikipedia.page(result)
#         except wikipedia.exceptions.DisambiguationError as e:
#             continue
#         except wikipedia.exceptions.PageError as e:
#             continue
#
#         summary = page.summary
#         paragraphs = summary.split('\n')
#
#         for paragraph in paragraphs:
#             if head.lower() in paragraph.lower() and tail.lower() in paragraph.lower():
#                 return paragraph
#
#     return None


def call_es(head, tail, mode="para"):
    headers = {'Content-Type': 'application/json'}
    url_para = 'http://166.111.7.106:9200/wikipedia_paragraph/wikipedia_paragraph/_search'
    url_sentence = 'http://166.111.7.106:9200/wikipedia_sentence/wikipedia_sentence/_search'
    url = {"sentence": url_sentence, "para": url_para}
    data = {
        "query": {"bool": {"must": [{"match": {"text": head}}, {"match": {"text": tail}}]}}
    }
    with post(url=url[mode], headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf8'), auth=("nekol", "kegGER123")) as resp:
        results = resp.json()
        s_r = results['hits']['hits']
        s_r_filter = [r['_source']['text'] for r in s_r if head in r['_source']['text'] and tail in r['_source']['text']]
        if not s_r_filter:
            print("holy shit")
            return "holy shit"
        return " ".join(s_r_filter)


def get_ground_tail_truth(item):
    h_r = {}
    for label in item.get("labels"):
        head = str(label['h'])
        relation = rel_info[label['r']]
        tail = label['t']
        if head + "_" + relation in h_r:
            h_r[head + "_" + relation].append(tail)
        else:
            h_r[head + "_" + relation] = [tail]
    return h_r


def make_vf_train_data(relation, mode, tool="es"):
    entity_relation_tail = json.load(open("../../data/head_relation_tail_constrain_train.json"))
    data = json.load(open("../../data/train_revised.json"))
    result_save_path = f"./for_vf_train_data"
    os.makedirs(result_save_path, exist_ok=True)
    with open(result_save_path + "/" + relation + ".json", "w") as f:
        uuid = 0
        groupid = 0
        for doc_id, sample in enumerate(tqdm(data)):
            h_t = get_ground_tail_truth(sample)
            for r in sample['labels']:
                r_name = rel_info[r['r']]
                if r_name != relation:
                    continue
                h_names = list(set([h['name'] for h in sample['vertexSet'][r['h']]]))
                t_names = list(set([t['name'] for t in sample['vertexSet'][r['t']]]))
                h_name = max(h_names, key=len)
                t_name = max(t_names, key=len)
                text = call_es(h_name, t_name, mode) if tool == "es" else get_wikipedia_summary_containing_head_tail(h_name, t_name)
                if text != "holy shit":
                    for template in sentence_template[relation].values():
                        v = {
                            "uuid": uuid,
                            "groupid": groupid,
                            "input": template.format(s={
                                "sentence": text,
                                "head": h_name,
                                "tail": t_name,
                            }),
                            "head": h_name,
                            "tail": t_name,
                            "label": "Yes"
                        }
                        json.dump(v, f)
                        f.write("\n")
                        uuid += 1
                        break
                    groupid += 1

                h_type = str(sorted(list(set([h['type'] for h in sample['vertexSet'][r['h']]]))))
                try:
                    tail_constrains = entity_relation_tail[h_type][relation]
                except:
                    continue

                for t_index, vertex_t in enumerate(sample['vertexSet']):
                    if t_index == r['h']:
                        continue
                    t_type = str(sorted(list(set([t['type'] for t in vertex_t]))))
                    if t_type not in tail_constrains or t_index in h_t.get(str(r['h']) + "_" + relation, []):
                        continue
                    t_names = list(set([t['name'] for t in vertex_t]))
                    t_name = max(t_names, key=len)
                    text = call_es(h_name, t_name, mode) if tool == "es" else get_wikipedia_summary_containing_head_tail(h_name, t_name)
                    if text != "holy shit":
                        for template in sentence_template[relation].values():
                            v = {
                                "uuid": uuid,
                                "groupid": groupid,
                                "input": template.format(s={
                                    "sentence": text,
                                    "head": h_name,
                                    "tail": t_name,
                                }),
                                "head": h_name,
                                "tail": t_name,
                                "label": "No"
                            }
                            json.dump(v, f)
                            f.write("\n")
                            uuid += 1
                            break
                        groupid += 1


if __name__ == '__main__':
    # mode = "para"
    # tool = "es"
    # for relation in relation_list[:1]:
    #     make_vf_train_data(relation, mode, tool)
    a = call_es("head","tail")
    print(a)
    b = call_es("tail","head")
    print(b)


