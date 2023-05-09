import random
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
        # 只返回两个para，减小句子的长度
        return " ".join([r['_source']['text'] for r in s_r[:2]])


main = "[Thought] To be a fact extractor, I need to start by retrieving sentences from Wikipedia.\n" \
       "[Action] 【get_sentences()】\n" \
       "[Return] SENTENCES=\"{s[sentence]}\"\n" \
       "\n" \
       "[Thought] After obtaining sentences from Wikipedia, it's time to identify all ENTITIES in the SENTENCES.\n" \
       "[Action] 【get_entities(sentences=SENTENCES)】\n" \
       "[Return] ENTITIES=\"{s[entities]}\", with all ENTITIES found, I should iterate all ENTITIES to get the facts.\n" \
       "\n" \
       "{entity}"

entity = "[Thought] When iterating all ENTITIES, I must select an entity from ENTITIES as the head.\n" \
         "[Action] 【choose_an_entity(entities=ENTITIES)】\n" \
         "[Return] HEAD=\"{s[head]}\"\n" \
         "\n" \
         "[Thought] Having selected a HEAD from ENTITIES, I need to determine what TYPES the HEAD may have.\n" \
         "[Action] 【get_types(head=HEAD)】\n" \
         "[Return] TYPES=\"{s[types]}\", found all TYPES, I should iterate all TYPES to check the RELATIONS HEAD may have\n" \
         "\n", \
         "{types}" \
         "\n" \
         "[Thought] Is ENTITIES all done? \n" \
         "[Action] 【check_entities()】\n" \
         "[Return] {s[is_entity_done]}"

types = "[Thought] When iterating all TYPES, it's time to choose a TYPE from TYPES.\n" \
        "[Action] 【choose_a_type(types=TYPES)】\n" \
        "[Return] TYPE=\"{s[type]}\"\n" \
        "\n" \
        "[Thought] With a TYPE selected from TYPES, I must discover what relations HEAD may have according to TYPE.\n" \
        "[Action] 【get_relations(head=HEAD,type=TYPE)】\n" \
        "[Return] RELATIONS=\"{s[relations]}\", after finding all RELATIONS, I should iterate them to get TAILS.\n" \
        "\n" \
        "{relations}\n" \
        "\n" \
        "[Thought] Is TYPES all done? \n" \
        "[Action] 【check_types()】\n" \
        "[Return] {s[is_type_done]}"

relations = "[Thought] Having obtained all RELATIONS, I need to pick a relation from RELATIONS.\n" \
            "[Action] 【choose_a_relation(relations=RELATIONS)】\n" \
            "[Return] RELATION=\"{s[relation]}\"\n" \
            "\n" \
            "{process_tails}"

process_tails = "[Thought] With a RELATION chosen, I must attempt to get RELATION_ALIAS_TEMPLATE.\n" \
                "[Action] 【get_relation_alias_template(relation=RELATION)】\n" \
                "[Return] RELATION_ALIAS_TEMPLATE={s[relation_template]}\n" \
                "\n" \
                "[Thought] After obtaining RELATION_ALIAS_TEMPLATE, I need to use RELATION_ALIAS_TEMPLATE to get TAILS.\n" \
                "[Action] 【get_tails(sentence=SENTENCE, relation_alias_template=RELATION_ALIAS_TEMPLATE, head=HEAD)】\n" \
                "[Return] TAILS=\"{s[tails]}\"\n" \
                "\n" \
                "{verify_each_tail}"

verify_each_tail = "[Thought] For each TAIL in TAILS, I need to search for more information about HEAD and TAIL using a search engine.\n" \
                   "[Action] 【search_engine(head=HEAD, tail=TAIL)】\n" \
                   "[Return] CONTEXT=\"{s[context]}\"\n" \
                   "\n" \
                   "[Thought] After obtaining the CONTEXT, I can utilize the information to verify the fact.\n" \
                   "[Action] 【verify(context=CONTEXT, head=HEAD, tail=TAIL)】\n" \
                   "[Return] \"{s[label]\"\n"

"[Thought] If there are more TAILS to process, repeat the process for the next TAIL. Otherwise, continue with the next RELATION, TYPE, or ENTITY, as needed.\n"

rel_info = json.load(open("../data/base_data_for_all/rel_info.json"))
relation_constrains = json.load(open("../data/base_data_for_all/head_relation_tail_constrain_dev.json"))
data = json.load(open("../data/base_data_for_all/dev_revised.json"))
all = []


def make_data():
    with open("./train_auto_glm_data/ori.json", "w") as f:
        for sample in tqdm(data):
            labels = sample['labels']
            main_s = {}
            sentence = " ".join([" ".join(sent) for sent in [s for s in sample['sents']]])
            main_s['sentence'] = sentence
            entities = []
            for entity in sample['vertexSet']:
                entity_names = max([e['name'] for e in entity], key=len)
                entities.append(entity_names)

            main_s['entities'] = entities[:5]
            for l in labels:
                head_name = max([h['name'] for h in sample['vertexSet'][l['h']]], key=len)
                head_type = sorted(list(set([h['type'] for h in sample['vertexSet'][l['h']]])))
                tail_name = max([t['name'] for t in sample['vertexSet'][l['t']]], key=len)
                main_s['head'] = head_name
                main_s['tail'] = tail_name

                inner_template_s = {}
                inner_template_s['head'] = main_s['head']
                inner_template_s['entities'] = main_s['entities']
                inner_template_s['relations'] = get_relations(head_type)[:2]
                all_inner_inner = ""
                for relation in inner_template_s['relations']:
                    inner_inner_template_s = {}
                    inner_inner_template_s['head'] = main_s['head']
                    inner_inner_template_s['sentence'] = main_s['sentence']
                    inner_inner_template_s["relation"] = relation
                    templates = save_gen_template[relation]
                    relation_template = {k: templates[k] for k in random.sample(templates.keys(), 2)}
                    inner_inner_template_s["relation_template"] = relation_template

                    inner_inner_template_s['tail'] = main_s['tail']

                    inner_inner_inner_template_s = {}
                    inner_inner_inner_template_s['head'] = main_s['head']
                    inner_inner_inner_template_s['tail'] = main_s['tail']
                    inner_inner_inner_template_s['context'] = call_es((main_s['head'], main_s['tail']))
                    inner_inner_inner_template_s['label'] = random.choice(["yes", "no"])

                    inner_inner_inner_template_ = inner_inner_inner_template.format(s=inner_inner_inner_template_s)
                    inner_inner_template_s["inner_inner_inner_template"] = inner_inner_inner_template_
                    all_inner_inner += inner_inner_template.format(s=inner_inner_template_s)

                inner_template_s['inner_inner_template'] = all_inner_inner
                inner_template_ = inner_template.format(s=inner_template_s)
                main_s['inner_template'] = inner_template_

                s = main_template.format(s=main_s)
                # all.append(s)
                s = s.replace("\n\n\n", "\n\n")
                f.write(json.dumps(s) + "\n")


def split_json_file(input_file_path, train_output_file_path, valid_output_file_path, train_ratio=0.9):
    with open(input_file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    num_train = int(len(data) * train_ratio)
    train_data = data[:num_train]
    valid_data = data[num_train:]
    # valid_data = [d for d in data if d not in train_data]

    with open(train_output_file_path, 'w') as f:
        for d in tqdm(train_data, desc="train"):
            f.write(json.dumps(d) + '\n')

    with open(valid_output_file_path, 'w') as f:
        for d in tqdm(valid_data, desc="valid"):
            f.write(json.dumps(d) + '\n')


def make_rounds_chat(input_file_path):
    with open("./rounds_chat.json", "w") as outfile:
        with open(input_file_path, 'r') as f:
            datas = [json.loads(line) for line in f]
            for data in tqdm(datas):
                blocks = data.strip().split('\n\n')
                # Initialize the final dataset
                dataset = []

                for i, block in enumerate(blocks):
                    lines = block.split('\n')

                    # Initialize the history list
                    history = []
                    # Add the previous histories
                    for j in range(i):
                        prev_lines = blocks[j].split('\n')
                        history.append([prev_lines[0], prev_lines[1] + "\n" + prev_lines[2]])

                    # Create a dictionary for the current block
                    try:
                        block_dict = {
                            "prompt": lines[0],
                            "response": lines[1],
                            "history": history
                        }
                    except:
                        pass
                    # Add the current block dictionary to the dataset
                    dataset.append(block_dict)

                for block in dataset:
                    json.dump(block, outfile)
                    outfile.write('\n')


if __name__ == '__main__':
    make_data()
    make_rounds_chat("./train_auto_glm_data/ori.json")
    split_json_file("./rounds_chat.json", "../ChatGLM-6B-main/ptuning/auto_kg/train.json", "../ChatGLM-6B-main/ptuning/auto_kg/dev.json")
