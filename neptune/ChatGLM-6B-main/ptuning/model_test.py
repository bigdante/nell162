import os
import re
from parse_api import *
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
import torch

cuda_device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
model_name_or_path = "THUDM/chatglm-6b"
ptuning_checkpoint = './output/adgen-chatglm-6b-pt-one-64-1e-2/checkpoint-30000'
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
input = {"prompt": "[Thought] After get all TAILS, I need to choose a tail from TAILS.", "response": "[Action] \u3010choose_a_tail(tails=TAILS)\u3011", "history": [["[Thought] To be a fact extractor, I need to start by retrieving sentences from Wikipedia.", "[Action] \u3010get_sentences()\u3011\n[Return] SENTENCES=\"Wilfried \" Willi \" Schneider ( born 13 March 1963 in Media\u0219 , Transylvania ) is a German skeleton racer who competed from 1992 to 2002 . He won two medals in the men 's skeleton event at the FIBT World Championships with a gold in 1998 and a bronze in 1999 . Schneider also finish ninth in the men 's skeleton event at the 2002 Winter Olympics in Salt Lake City . He won the men 's overall Skeleton World Cup title in 1997 - 8 . After retiring from competition Schneider became a coach , leading the Canadian skeleton team to three medals at the 2006 Winter Olympics in Turin ( a gold for Duff Gibson , a silver for Jeff Pain and a bronze for Melissa Hollingsworth ) , and coaching Jon Montgomery to victory in the 2010 Winter Olympics in Vancouver , British Columbia , Canada . In July 2012 Schneider agreed a two - year contract to coach the Russian skeleton team .\""], ["[Thought] After obtaining sentences from Wikipedia, it's time to identify all ENTITIES in the SENTENCES.", "[Action] \u3010get_entities(sentences=SENTENCES)\u3011\n[Return] ENTITIES=\"['Wilfried \" Willi \" Schneider', '13 March 1963', 'Media\u0219', 'Transylvania', 'German']\""], ["[Thought] With all ENTITIES found, I must select an entity from ENTITIES as the head.", "[Action] \u3010choose_an_entity(entities=ENTITIES)\u3011\n[Return] HEAD=\"2002 Winter Olympics\""], ["[Thought] Having selected a HEAD from ENTITIES, I need to determine what TYPES the HEAD may have.", "[Action] \u3010get_types(head=HEAD)\u3011\n[Return] TYPES=\"['MISC']\""], ["[Thought] After obtaining HEAD TYPES, it's time to choose a TYPE from TYPES.", "[Action] \u3010choose_a_type(types=TYPES)\u3011\n[Return] TYPE=\"MISC\""], ["[Thought] With a TYPE selected from TYPES, I must discover what relations HEAD may have according to TYPE.", "[Action] \u3010get_relations(head=HEAD,type=TYPE)\u3011\n[Return] RELATIONS=\"['start time', 'end time', 'location']\""], ["[Thought] Having obtained all RELATIONS, I need to pick a relation from RELATIONS.", "[Action] \u3010choose_a_relation(relations=RELATIONS)\u3011\n[Return] RELATION=\"location\""], ["[Thought] With a RELATION chosen, I must attempt to get RELATION_ALIAS_TEMPLATE.", "[Action] \u3010get_relation_alias_template(relation=RELATION)\u3011\n[Return] RELATION_ALIAS_TEMPLATE={'locality': \"Given the sentence: '{s[sentence]}', determine the locality of {s[head]} by selecting the correct options from the provided list: {s[tail_choose]}.\", 'located in': \"Given the sentence: '{s[sentence]}', determine where {s[head]} is located by selecting the correct options from the provided list: {s[tail_choose]}.\"}"], ["[Thought] After obtaining RELATION_ALIAS_TEMPLATE, I need to use RELATION_ALIAS_TEMPLATE to get a TAIL.", "[Action] \u3010get_tail(sentence=SENTENCE, relation_alias_template=RELATION_ALIAS_TEMPLATE, head=HEAD)\u3011\n[Return] TAILS=\"['2002', 'Media\u0219']\""]]}
ori_input = input['prompt']
ori_history = input['history']
response, new_history = model.chat(tokenizer, ori_input, history=ori_history)
match = re.search(r'【([^】,]+)', response)
result = match.group(1)
print(response)
print(result)
# if match:
#     result = match.group(1)
#     for f in functions:
#         if f in result:
#             method_return = get_api(f)
#             ori_history = new_history
#             ori_input = ori_input + response + "\n[Return] " + method_return

