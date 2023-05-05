import json
from collections import Counter

from template import *

relation_ground_truth_count = json.load(open("../../data/relation_count.json"))
for relation in relation_list[:1]:
    result_data = []
    with open(f"./data/alias/query_result/{relation}.json", "r") as f:
        for line in f:
            json_obj = json.loads(line)
            result_data.append(json_obj)
    vote_data = {}
    with open(f"./data/alias/query_result_vote/{relation}.json", "w") as out:
        for query in result_data:
            if query['input']['query_id'] in vote_data:
                vote_data[query['input']['query_id']]["answer"].append(query['output'])
            else:
                vote_data[query['input']['query_id']] = {"sentence": query['input']['query']['prompt'], "head": query['input']['head'], "label": query['input'][
                    'label'], "answer": [query['output']]}
        tp, fp, tn, fn = 0, 0, 0, 0
        for query_id, query_data in vote_data.items():
            answer_list = query_data['answer']
            result = [a for a in answer_list]
            vote_answer = max(result, key=answer_list.count)
            copy_vote_data = query_data.copy()
            copy_vote_data['tail'] = vote_answer
            copy_vote_data['vote_answer'] = vote_answer
            copy_vote_data['head'] = query_data['head']
            if vote_answer == "yes" and copy_vote_data['label'] == "tp":
                tp += 1
                copy_vote_data['result'] = "true_p"
            if vote_answer == "yes" and copy_vote_data['label'] == "fp":
                fp += 1
                copy_vote_data['result'] = "false_p"
            if vote_answer == "no" or vote_answer == "unknown" and copy_vote_data['label'] == "tp":
                fn += 1
                copy_vote_data['result'] = "false_n"
            if vote_answer == "no" or vote_answer == "unknown" and copy_vote_data['label'] == "fp":
                tn += 1
                copy_vote_data['result'] = "true_n"
            out.write(json.dumps(copy_vote_data) + "\n")
        print(f"{relation} tp:{tp} fp:{fp} tn:{tn} fn:{fn}, recall:{tp / relation_ground_truth_count[relation]}")
