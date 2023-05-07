# import os
# import re
# from typing import Dict, Tuple, Union, Optional
# from parse_api import *
# import torch
# from torch.nn import Module
# from transformers import AutoModel, AutoConfig, AutoTokenizer
#
#
# def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
#     # transformer.word_embeddings 占用1层
#     # transformer.final_layernorm 和 lm_head 占用1层
#     # transformer.layers 占用 28 层
#     # 总共30层分配到num_gpus张卡上
#     num_trans_layers = 28
#     per_gpu_layers = 30 / num_gpus
#
#     # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
#     # windows下 model.device 会被设置成 transformer.word_embeddings.device
#     # linux下 model.device 会被设置成 lm_head.device
#     # 在调用chat或者stream_chat时,input_ids会被放到model.device上
#     # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
#     # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
#     device_map = {'transformer.word_embeddings': 0,
#                   'transformer.final_layernorm': 0, 'lm_head': 0}
#
#     used = 2
#     gpu_target = 0
#     for i in range(num_trans_layers):
#         if used >= per_gpu_layers:
#             gpu_target += 1
#             used = 0
#         assert gpu_target < num_gpus
#         device_map[f'transformer.layers.{i}'] = gpu_target
#         used += 1
#     device_map['transformer.prefix_encoder.embedding.weight'] = 3
#     return device_map
#
#
# ptuning_checkpoint = './output/adgen-chatglm-6b-pt-one-64-1e-2/checkpoint-20000'
# checkpoint_path = "THUDM/chatglm-6b"
#
#
# def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
#                        device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
#     if num_gpus < 2 and device_map is None:
#         model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
#     else:
#         from accelerate import dispatch_model
#         config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
#         config.pre_seq_len = 64
#         config.prefix_projection = False
#         model = AutoModel.from_pretrained(checkpoint_path, config=config, trust_remote_code=True, **kwargs).half()
#         prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
#         new_prefix_state_dict = {}
#         for k, v in prefix_state_dict.items():
#             if k.startswith("transformer.prefix_encoder."):
#                 new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
#         model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
#         if device_map is None:
#             device_map = auto_configure_device_map(num_gpus)
#
#         model = dispatch_model(model, device_map=device_map)
#         model.transformer.prefix_encoder.embedding.weight.data = model.transformer.prefix_encoder.embedding.weight.data.to(model.device)
#
#     return model
#
#
# tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
# model = load_model_on_gpus(checkpoint_path, 4, auto_configure_device_map(4))
#
# model = model.eval()
# input = {"prompt": "[Thought] With a RELATION chosen, I must attempt to get RELATION_ALIAS_TEMPLATE.",
#          "response": "[Action] \u3010get_relation_alias_template(relation=RELATION)\u3011", "history": [
#         ["[Thought] To be a fact extractor, I need to start by retrieving sentences from Wikipedia.",
#          "[Action] \u3010get_sentences()\u3011\n[Return] SENTENCES=\"Peadar Bracken ( 23 January 1887 \u2013 January 19 , 1961 ) was an Irish freedom fighter during Ireland 's War of Independence . In 1914 he was elected Captain by the men of the Tullamore Company Irish Volunteers , before later being appointed Commandant of the Athlone Brigade area by Padraig Pearse . Bracken , alongside his comrade Seamus Brennan are credited with firing the first ever shots in the 1916 rising in Tullamore . Bracken was under order from Padraig Pearse and took command off him for the duration of the 1916 rising . He commanded the critical GPO garrison outpost at O'Connell Bridge , Dublin during the Easter Rising . Later on in his career , in 1920 , he was a senior officer in the Irish Republican Brotherhood branch in Tullamore , and commanding officer in chief of the Athlone Brigade . Bracken died in January 19 , 1961 .\""],
#         ["[Thought] After obtaining sentences from Wikipedia, it's time to identify all ENTITIES in the SENTENCES.",
#          "[Action] \u3010get_entities(sentences=SENTENCES)\u3011\n[Return] ENTITIES=\"['Peadar Bracken', '23 January 1887', 'January 19 , 1961', 'Ireland', 'War of Independence']\""],
#         ["[Thought] With all ENTITIES found, I must select an entity from ENTITIES as the head.",
#          "[Action] \u3010choose_an_entity(entities=ENTITIES)\u3011\n[Return] HEAD=\"War of Independence\""],
#         ["[Thought] Having selected a HEAD from ENTITIES, I need to determine what TYPES the HEAD may have.",
#          "[Action] \u3010get_types(head=HEAD)\u3011\n[Return] TYPES=\"['MISC']\""],
#         ["[Thought] After obtaining HEAD TYPES, it's time to choose a TYPE from TYPES.", "[Action] \u3010choose_a_type(types=TYPES)\u3011\n[Return] TYPE=\"MISC\""],
#         ["[Thought] With a TYPE selected from TYPES, I must discover what relations HEAD may have according to TYPE.",
#          "[Action] \u3010get_relations(head=HEAD,type=TYPE)\u3011\n[Return] RELATIONS=\"['start time', 'end time', 'location']\""],
#         ["[Thought] Having obtained all RELATIONS, I need to pick a relation from RELATIONS.",
#          "[Action] \u3010choose_a_relation(relations=RELATIONS)\u3011\n[Return] RELATION=\"end time\""]]}
#
# ori_input = "[Thought] To be a fact extractor, I need to start by retrieving sentences from Wikipedia."
# ori_history = input['history']
# response, new_history = model.chat(tokenizer, ori_input, history=ori_history)
# pattern = r'【(.*?)】'
# match = re.search(pattern, response)
# result = match.group(1)
# print(response)
# print(result)
# print("================")
# for f in functions:
#     if f in result:
#         method_return = get_api(f)
#         print(method_return)
#
import subprocess
import os

# 设置 Ninja 的路径
# ninja_path = "/zhangpai22/envs/dragon/bin"
#
# # 获取当前的 PATH 环境变量
# current_path = os.environ.get("PATH", "")
#
# # 将 Ninja 的路径添加到 PATH 环境变量中
# os.environ["PATH"] = f"{current_path}:{ninja_path}"

def a():
    try:
        subprocess.check_output('ninja --version'.split())
    except Exception:
        return False
    else:
        return True
print(a())
print('ninja --version'.split())


# Make alias template for me with the given relation alias: ['place of birth','owned by','birthday'].
# For example: given relation alias ["country of citizenship","subject of (country)","subject of","citizenship","citizen of","national of"], you should give me relation alias templates as following:
# {"country of citizenship": "Given the sentence: '{s[sentence]}', determine the country of citizenship for {s[head]}.","subject of": "Given the sentence: '{s[sentence]}', identify the country for which {s[head]} is a subject.","citizenship": "Given the sentence: '{s[sentence]}', determine the citizenship of {s[head]}.","citizen of": "Given the sentence: '{s[sentence]}', identify the country of which {s[head]} is a citizen.","national of": "Given the sentence: '{s[sentence]}', determine the nationality of {s[head]}.",}
# the keys are relation alias, and the values are templates. You should give me relation alias template according to the relation description.
# {s[head]} must appear in the template, do not have {s[tail]} or {s[object]} in the template.Response Format:
# Your respond must be a dict format, for example:
#     {
#       'alias1':'template1'
#       'alias2':'template2'
#       'alias3':'template3'
#     }
# Ensure the response can be parsed by Python eval().