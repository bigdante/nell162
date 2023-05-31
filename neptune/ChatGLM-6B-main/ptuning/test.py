import json

from requests import post
from tqdm import tqdm

evaluation_data = json.load(open("./data/redocred/test_evaluation_data.json"))


def engine(text, mode="para"):
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


save = {}
for data in tqdm(evaluation_data):
    head = data['head']
    tails = data['entities']
    for tail in tails:
        result = engine((head, tail))
        save[head + tail] = result
json.dump(save, open("./es_data.json", "w"), indent=4)
