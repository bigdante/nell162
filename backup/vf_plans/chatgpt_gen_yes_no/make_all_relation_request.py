import json
from template import *
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def get_logit_bias(tail_list):
    logit_bias = {"13": -100, "4083": -100}
    for t in tail_list:
        ids = tokenizer.encode_plus(t)['input_ids']
        for tid in ids:
            if str(tid) not in logit_bias:
                logit_bias[str(tid)] = 50
    return logit_bias


def make_all_relation_query(relation, mode):
    relation_vf_templates = vf_template[relation]
    with open(f"./data/{mode}/query_data/{relation}.json", "w") as file:
        uuid = 0
        query_id = 0
        result_data = []
        suffix = " You must only answer yes, no or unknown. Answer:"
        with open(f"../../chatgpt_gen_tail/data/{mode}/query_result_vote/{relation}.json", "r") as f:
            for line in f:
                json_obj = json.loads(line)
                result_data.append(json_obj)
            for r in result_data:
                if r['vote_answer'] == "unknown":
                    continue
                for template in relation_vf_templates.values():
                    s = {
                        "uuid": uuid,
                        "query_id": query_id,
                        "head": r['head'],
                        "tail": r['vote_answer'],
                        "query": {
                            "prompt": template.format(s={
                                "sentence": r['sentence'],
                                "head": r['head'],
                                "tail": r['vote_answer'],
                            }),
                            "logit_bias": get_logit_bias(["yes", "no", "unknown"]),
                        },
                        "label": r['label']
                    }
                    file.write(json.dumps(s) + "\n")
                    uuid += 1
                query_id += 1
    return f"{relation} done"


if __name__ == '__main__':
    relation_list = ["country of citizenship"]
    for relation in relation_list:
        make_all_relation_query(relation, mode="alias")
