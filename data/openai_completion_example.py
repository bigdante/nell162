import json
import random
import requests
import time
from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime
from tqdm import tqdm
# from transformers import GPT2Tokenizer
#
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def make_request(prompt, key, max_length=1024):
    """
    For `text-davinci-003` api: $0.02 per 1k tokens
    Introduction: https://platform.openai.com/docs/guides/completion/introduction
    Parameters: https://platform.openai.com/docs/api-reference/chat
    """
    with requests.post(url=f"https://api.openai.com/v1/completions",
                       headers={"Authorization": f"Bearer {key}"},
                       json={
                           "model": "text-davinci-003",
                           "temperature": 1.0,
                           "prompt": prompt,
                           "max_tokens": max_length,
                           "top_p": 1.0
                       }) as resp:
        return json.loads(resp.content)


def make_chat_request(prompt, key, max_length=1024):
    """
    For `gpt-3.5-turbo` api: $0.002 per 1k tokens.
    Introduction: https://platform.openai.com/docs/guides/chat/introduction
    Parameters: https://platform.openai.com/docs/api-reference/completions
    """
    with requests.post(url=f"https://api.openai.com/v1/chat/completions",
                       headers={"Authorization": f"Bearer {key}"},
                       json={
                           "model": "gpt-3.5-turbo",
                           "temperature": 1.0,
                           "messages": [{'role': 'user', 'content': prompt}],
                           "max_tokens": max_length,
                           "top_p": 1.0,
                           "stop":"].",
                           # "logit_bias": {'13': -100, '50256': 50, '16010': 50, '28203': 50, '17940': 50, '16220': 50}
                           "logit_bias": {"13": -100, "11": -100}
                       }) as resp:
        return json.loads(resp.content)


def truncate(string):
    length = len(tokenizer.tokenize(string))
    if length > 4097 - 1024:
        length = 4097 - 2048
        return tokenizer.decode(tokenizer.encode(string)[- (4097 - 2048):]), length
    return string, length


def format_completion_request(query):
    return f"问题：{query}\n\n回答："


def single_worker(doc):
    global finish, all_num, alive
    req = format_completion_request(doc['query'])
    key = get_a_valid_key()
    wrong_trial = 0
    while wrong_trial < 5:
        try:
            req, input_length = truncate(req)
            resp = make_request(prompt=req, key=key, max_length=4097 - input_length)
        except KeyError as e:
            print(e)
            alive -= 1
            return False
        except Exception as e:
            print(str(e))
            alive -= 1
            return False
        if resp.get('error'):
            if resp['error']['message'] == 'You exceeded your current quota, please check your plan and billing details.':
                KEYS[key] = False
                continue
            elif resp['error']['message'].startswith('Rate limit reached for default-gpt-3.5-turbo in organization'):
                time.sleep(0.5)
            else:
                print(f"Key {key} error. Message: {resp}")
            CNTS[key] += 1
            wrong_trial += 1
            key = get_a_valid_key()
        else:
            CNTS[key] = 0
            break
    if wrong_trial >= 5:
        return False
    res = {"input": doc, "output": resp}
    with open(f"[OUT_PATH]/{filename.replace('.jsonl', '-out.jsonl')}", "a") as f:
        f.write(json.dumps(res, ensure_ascii=False) + '\n')
    finish += 1
    alive -= 1
    print(f"Finish {finish} / {all_num} completions.")
    return True


def get_a_valid_key():
    while True:
        candidate = random.choice(list(KEYS.items()))
        if CNTS[candidate[0]] >= 5:
            KEYS[candidate[0]] = False
            json.dump(KEYS, open(f'[DATA_DIR]/{KEY_FILENAME}', 'w'))
        elif candidate[1]:
            return candidate[0]


def generate_query_conditioned_on_content(in_file):
    global all_num, alive
    all_num = sum([1 for _ in open(in_file)])
    print("All samples:", all_num)
    with ThreadPoolExecutor(max_workers=20) as pool:
        for line in tqdm(open(in_file)):
            doc = json.loads(line)
            pool.submit(single_worker, doc)
            alive += 1
            while alive >= 200:
                time.sleep(10)
        while alive >= 0:
            time.sleep(3)


if __name__ == '__main__':
    prompt = "Given the sentence: 'Wilfried \" Willi \" Schneider ( born 13 March 1963 in Mediaș , Transylvania ) is a German skeleton racer who competed from 1992 to 2002 . He won two medals in the men 's skeleton event at the FIBT World Championships with a gold in 1998 and a bronze in 1999 . Schneider also finish ninth in the men 's skeleton event at the 2002 Winter Olympics in Salt Lake City . He won the men 's overall Skeleton World Cup title in 1997 - 8 . After retiring from competition Schneider became a coach , leading the Canadian skeleton team to three medals at the 2006 Winter Olympics in Turin ( a gold for Duff Gibson , a silver for Jeff Pain and a bronze for Melissa Hollingsworth ) , and coaching Jon Montgomery to victory in the 2010 Winter Olympics in Vancouver , British Columbia , Canada . In July 2012 Schneider agreed a two - year contract to coach the Russian skeleton team .', determine the countries of citizenship of Wilfried \" Willi \" Schneider by selecting the correct options from the provided list: ['Mediaș', 'Transylvania', 'German', 'FIBT World Championships', '2002 Winter Olympics', 'Salt Lake City', 'Skeleton World Cup', 'Canadian', '2006 Winter Olympics', 'Turin', '2010 Winter Olympics', 'Vancouver', 'British Columbia', 'Canada', 'Russian'].(If no correct options are found, output 'unknown'). answer format '[answer1,anser2,...]'. Possible answers: "
    start = time.time()
    a = make_chat_request(prompt, key="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")
    print(time.time() - start)
    print(a)
    # print(a['choices'][0]['message']['content'])

    # KEY_FILENAME = 'keys.json'
    # KEYS = json.load(open(f'[KEY_DIR]/{KEY_FILENAME}'))
    # CNTS = dict((k, 0) for k in KEYS)
    # filename = "[FILE_NAME]"
    #
    # alive, finish, all_num = 0, 0, 0
    # generate_query_conditioned_on_content(f"[DATA_DIR]/{filename}")

    # TODO: Test single prompt for debugging
    # print(make_chat_request(
    #     prompt=format_cangtou_poem(
    #         json.loads(open(f"[DATA_DIR]/{filename}").readline())),
    #     key=get_a_valid_key()
    # ))
    # s = "FIBT World Championships"
    # a = tokenizer.convert_tokens_to_ids(s)
    # print(a)
    # b = tokenizer.convert_ids_to_tokens([37, 9865, 51, 2159, 24782])
    # print(b)
    # a = tokenizer.encode_plus(s)
    # print(a)
    # b = tokenizer.decode(a['input_ids'])
    # print(b)
