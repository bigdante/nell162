import os
import re
import json
from tqdm import tqdm
# from parse_api import *
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
import torch

cuda_device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
model_name_or_path = "THUDM/chatglm-6b"
ptuning_checkpoint = './output/2048_auto_kg-64-1e-2/checkpoint-2000'
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
config.pre_seq_len = 64
config.prefix_projection = False
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True).half()
prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
model.to(cuda_device)
model = model.eval()
while True:
    userinput = input("--->")
    try:
        u = eval(userinput)
    except:
        continue
    ori_input = u['prompt']
    ori_history = u['history']
    response, new_history = model.chat(tokenizer, ori_input, history=ori_history)
    print(response)
    try:
        match = re.search(r'【([^】,]+)', response)
        result = match.group(1)
        print(result)
    except:
        continue
# if match:
#     result = match.group(1)
#     for f in functions:
#         if f in result:
#             method_return = get_api(f)
#             ori_history = new_history
#             ori_input = ori_input + response + "\n[Return] " + method_return


# file = open("./auto_kg/make_COT_traindata_redocred/step/rounds_head_type_relation_alias.json")
# lines = file.readlines()
# count = 0
# max_len = 0
# for index,line in enumerate(tqdm(lines)):
#     a = json.loads(line)
#     c = " "
#     for i in a['history']:
#         c += i[0] + " " + i[1] + " "
#     b = tokenizer.tokenize(c)
#     d = tokenizer.encode(c)
#     max_len = max(len(d), max_len)
#     if len(d) > 2048:
#         count += 1
#         print(index)
#         break