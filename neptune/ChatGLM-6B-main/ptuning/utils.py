import json
import pickle

from requests import post

rel_info = json.load(open("./data/rel_info.json"))
head_relation_tail_constrain = json.load(open("./data/head_relation_tail_constrain_dev.json"))


def get_ground_tail_truth(item):
    '''
        获取某个doc的head+relation=[tail_ids,...]
    '''
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


def get_text_by_sent_id(h=None, t=None, sample=None, ids=None, evidence_ids=None):
    if evidence_ids:
        sent_ids = sorted(list(set([i['sent_id'] for i in h] + [i['sent_id'] for i in t] + evidence_ids)))
    else:
        if not ids:
            sent_ids = sorted(list(set([i['sent_id'] for i in h] + [i['sent_id'] for i in t])))
        else:
            sent_ids = sorted(list(set(ids)))
    text = " ".join([" ".join(sample["sents"][i]) for i in sent_ids])
    return text, sent_ids




def save_var(var_name, var_value):
    try:
        with open('my_vars.pkl', 'rb') as f:
            saved_vars = pickle.load(f)
    except EOFError:
        saved_vars = {}

    saved_vars[var_name] = var_value

    with open('my_vars.pkl', 'wb') as f:
        pickle.dump(saved_vars, f)

def load_var(var_name):
    with open('my_vars.pkl', 'rb') as f:
        saved_vars = pickle.load(f)
    return saved_vars[var_name] if var_name in saved_vars else None


def call_es(text, mode="para"):
    head, tail = text
    headers = {'Content-Type': 'application/json'}
    url_para = 'http://166.111.7.106:9200/wikipedia_paragraph/wikipedia_paragraph/_search'
    url_sentence = 'http://166.111.7.106:9200/wikipedia_sentence/wikipedia_sentence/_search'
    url = {"sentence": url_sentence, "para": url_para}
    data = {
        "query": {"bool": {"must": [{"match": {"text": head}}, {"match": {"text": tail}}]}}
    }
    with post(url=url[mode], headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf8'),
              auth=("nekol", "kegGER123")) as resp:
        results = resp.json()
        s_r = results['hits']['hits']

        return " ".join([r['_source']['text'] for r in s_r[:2]])