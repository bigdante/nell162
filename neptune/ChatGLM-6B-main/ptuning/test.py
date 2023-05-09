# import os
# import re
# import json
# from tqdm import tqdm
# # from parse_api import *
# from transformers import (
#     AutoConfig,
#     AutoModel,
#     AutoTokenizer,
# )
# import torch
#
# cuda_device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
# model_name_or_path = "THUDM/chatglm-6b"
# ptuning_checkpoint = './output/adgen-chatglm-6b-pt-one_2048_autokg-64-1e-2/checkpoint-19000'
# config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
# config.pre_seq_len = 64
# config.prefix_projection = False
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
#
# # file = open("./auto_kg/step/train.json")
# # lines = file.readlines()
# # count = 0
# # for line in tqdm(lines):
# #     a = json.loads(line)
# #     c = " "
# #     for i in a['history']:
# #         c += i[0] + " " + i[1] + " "
# #     b = tokenizer.tokenize(c)
# #     d = tokenizer.encode(c)
# #     if len(d) > 2048:
# #         count += 1
# # print(count)
#
# model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True).half()
# prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
# new_prefix_state_dict = {}
# for k, v in prefix_state_dict.items():
#     if k.startswith("transformer.prefix_encoder."):
#         new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
# model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
# model.to(cuda_device)
# model = model.eval()
# while True:
#     userinput = input("--->")
#     try:
#         u = eval(userinput)
#     except:
#         continue
#     ori_input = u['prompt']
#     ori_history = u['history']
#     response, new_history = model.chat(tokenizer, ori_input, history=ori_history)
#     print(response)
#     try:
#         match = re.search(r'【([^】,]+)', response)
#         result = match.group(1)
#         print(result)
#     except:
#         continue
# # if match:
# #     result = match.group(1)
# #     for f in functions:
# #         if f in result:
# #             method_return = get_api(f)
# #             ori_history = new_history
# #             ori_input = ori_input + response + "\n[Return] " + method_return
# a = {"prompt": "next?", "response": "[A] $0\u3010Start_TYPE_Branch\u3011", "history": [["[T] Begin, retrieve sentences.", "[A]\u3010get_sentences()\u3011\n[R] SENTENCE=The Chief Minister of Madhya Pradesh is the chief executive of the central Indian state of Madhya Pradesh . The first non - Congress chief minister was Govind Narayan Singh who defected from the party and lead a Samyukta Vidhayak Dal government from 1967 to 1969 ."], ["next?", "[T] Got sentence, identify ENTITIES in the SENTENCE."], ["next?", "[A]\u3010get_entities()\u3011"], ["next?", "[R] ENTITIES=['Madhya Pradesh', 'Indian', 'Indian National Congress', 'Govind Narayan Singh', 'Samyukta Vidhayak Dal', '1967', '1969']"], ["next?", "[T] There are 7 ENTITIES in the SENTENCE."], ["next?", "[T] ^0 With ENTITIES found, iterate the 0th ENTITY as HEAD."], ["next?", "[A] ^0\u3010choose_a_head()\u3011"], ["next?", "[R] ^0 HEAD=1969"], ["next?", "[A] ^0\u3010Start_HEAD_Branch\u3011Determine TYPES of HEAD according to SENTENCE."], ["next?", "[A] ^0\u3010determine_types()\u3011"], ["next?", "[R] ^0 TYPES=['TIME']"], ["next?", "[T] ^0 There are 1 TYPES of HEAD."], ["next?", "[T] $0 Obtained HEAD TYPES, traverse 0th TYPE."], ["next?", "[A] $0\u3010choose_a_type()\u3011"], ["next?", "[R] $0 TYPE=TIME"]]}
#
# d = ""
# for dd in a['history']:
#     d += "\n" + dd[0] + "\n" + dd[1]
# print(d)
import re
from ast import literal_eval

text = "[Return] TAIL=\"['Senna', 'Senna', 'Emerson Fittipaldi']\""

# Extract the content within the double quotes
content = re.search(r'"([^"]*)"', text)
if content:
    content_str = content.group(1)
    # Convert the extracted content to a list using ast.literal_eval
    result = literal_eval(content_str)
    print(result)
else:
    print("No content found within double quotes.")

