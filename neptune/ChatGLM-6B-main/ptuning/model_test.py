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
# ptuning_checkpoint = './output/adgen-chatglm-6b-pt-one_new-64-1e-2/checkpoint-37000'
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

# config.pre_seq_len = 64
# config.prefix_projection = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True).half()
# prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))

# new_prefix_state_dict = {}
# for k, v in prefix_state_dict.items():
#     if k.startswith("transformer.prefix_encoder."):
#         new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
# model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
model.to(cuda_device)
model = model.eval()
system_prompt = "Response Format: \n" \
                "Your respond must be a dict format, for example: \n" \
                "    {\n" \
                "      'alias1':'template1'\n" \
                "      'alias2':'template2'\n" \
                "      'alias3':'template3'\n" \
                "    }" \
                "\n" \
                "Ensure the response can be parsed by Python eval().\n"
example_prompt = "For example: given relation alias [\"country of citizenship\",\"subject of (country)\",\"subject of\",\"citizenship\",\"citizen of\",\"national of\"], you should give me relation alias templates as following:\n" \
                 "{" \
                 "\"country of citizenship\": \"Given the sentence: '{s[sentence]}', determine the country of citizenship for {s[head]}.\"," \
                 "\"subject of\": \"Given the sentence: '{s[sentence]}', identify the country for which {s[head]} is a subject.\"," \
                 "\"citizenship\": \"Given the sentence: '{s[sentence]}', determine the citizenship of {s[head]}.\"," \
                 "\"citizen of\": \"Given the sentence: '{s[sentence]}', identify the country of which {s[head]} is a citizen.\"," \
                 "\"national of\": \"Given the sentence: '{s[sentence]}', determine the nationality of {s[head]}.\"," \
                 "}\n" \
                 "the keys are relation alias, and the values are templates. You should give me relation alias template according to the relation description.\n" \
                 "{s[head]} must appear in the template, do not have {s[tail]} or {s[object]} in the template."
input = "Make alias template for me with the given relation alias: ['country','nation','birthday'].\n" + example_prompt + system_prompt
# print(input)
history = []

input = "Given the sentence: 'My name is Jack. I have lived in Beijing for 20 years, and I like roast duck. I came back to my hometown, the USA last year.'. Determine the country of Jack? Your respond must be a list format, for example:1['answer1','answer2',...],if unable to determine, output:['unknown'] Choose word from sentence."
response, new_history = model.chat(tokenizer, input, history=history)
print(response)
# in ={"prompt": "what about next", "response": "[Action]\u3010exit()\u3011", "history": [["[Thought] To be a fact extractor, I need to start by retrieving sentences from Wikipedia.", "[Action] \u3010get_sentences()\u3011\n[Return] SENTENCES=\"Big Muddy Creek is a tributary of the Missouri River , approximately 191 \u00a0 mi ( 307 \u00a0 km ) long , in the Canadian province of Saskatchewan and the U.S. state of Montana . It flows through the Big Muddy Badlands . It rises in southern Saskatchewan at Willow Bunch Lake , on the plains north of the international border and approximately 20 \u00a0 mi ( 32 \u00a0 km ) northwest of Big Beaver . It flows southeast through the Big Muddy Badlands and through Big Muddy Lake , then south into Sheridan County , Montana , past Redstone , then east , past Plentywood , then south , forming the eastern border of Fort Peck Indian Reservation . It joins the Missouri west of Culbertson . It was explored in 1805 by the Lewis and Clark Expedition , who called it Martha 's River in their journals and noticed on their return voyage in 1806 that it had changed its mouth on the Missouri . Along with the Milk River and the Poplar River , it is one of three waterways in Canada that drain into the Gulf of Mexico .\""], ["what about next", "[Thought] After obtaining sentences from Wikipedia, it's time to identify all ENTITIES in the SENTENCES."], ["what about next", "[Action] \u3010get_entities(sentences=SENTENCES)\u3011\n[Return] ENTITIES=\"['Big Muddy Creek', 'Missouri River', '191 \\xa0 mi', 'Canadian', 'Saskatchewan']\""], ["what about next", "[Thought] With all ENTITIES found, I must select an entity from ENTITIES as the head."], ["what about next", "[Action] \u3010choose_an_entity(entities=ENTITIES)\u3011\n[Return] HEAD=\"Culbertson\""], ["what about next", "[Thought] Having selected a HEAD from ENTITIES, I need to determine what TYPES the HEAD may have."], ["what about next", "[Action] \u3010get_types(head=HEAD)\u3011\n[Return] TYPES=\"['LOC']\""], ["what about next", "[Thought] After obtaining HEAD TYPES, it's time to choose a TYPE from TYPES."], ["what about next", "[Action] \u3010choose_a_type(types=TYPES)\u3011\n[Return] TYPE=\"LOC\""], ["what about next", "[Thought] With a TYPE selected from TYPES, I must discover what relations HEAD may have according to TYPE."], ["what about next", "[Action] \u3010get_relations(head=HEAD,type=TYPE)\u3011\n[Return] RELATIONS=\"['country', 'located in the administrative territorial entity', 'contains administrative territorial entity']\""], ["what about next", "[Thought] Having obtained all RELATIONS, I need to pick a relation from RELATIONS."], ["what about next", "[Action] \u3010choose_a_relation(relations=RELATIONS)\u3011\n[Return] RELATION=\"located in the administrative territorial entity\""], ["what about next", "[Thought] With a RELATION chosen, I must attempt to get RELATION_ALIAS_TEMPLATE."], ["what about next", "[Action] \u3010get_relation_alias_template(relation=RELATION)\u3011\n[Return] RELATION_ALIAS_TEMPLATE={'included in the administrative region of': \"From the sentence: '{s[sentence]}', identify the administrative regions that include {s[head]} by choosing the correct options from the given list: {s[tail_choose]}.\", 'belongs to the administration of': \"Considering the sentence: '{s[sentence]}', identify the administrative entities to which {s[head]} belongs by choosing the appropriate options from the list supplied: {s[tail_choose]}.\"}"], ["what about next", "[Thought] After obtaining RELATION_ALIAS_TEMPLATE, I need to use RELATION_ALIAS_TEMPLATE to get a TAIL."], ["what about next", "[Action] \u3010get_tail(sentence=SENTENCE, relation_alias_template=RELATION_ALIAS_TEMPLATE, head=HEAD)\u3011\n[Return] TAILS=\"['U.S.', 'Big Muddy Creek']\""], ["what about next", "[Thought] After get all TAILS, I need to choose a tail from TAILS."], ["what about next", "[Action] \u3010choose_a_tail(tails=TAILS)\u3011\n[Return] TAIL=\"U.S.\""], ["what about next", "[Thought] With a TAIL acquired, I can search for more information about HEAD and TAIL using a search engine."], ["what about next", "[Action] \u3010search_engine(head=HEAD, tail=TAIL)\u3011\n[Return] CONTEXT=\"William Wirt Culbertson (September 22, 1835 \u2013 October 31, 1911) was a U.S. Representative from Kentucky. Nebraska Highway 17 is a highway in southwestern Nebraska. It has a southern terminus at the Kansas border south of Culbertson and a northern terminus in Culbertson at an intersection with U.S. Highway 6 and U.S. Highway 34. The highway extends into Kansas at its southern terminus via K-117. William Constantine Culbertson (November 25, 1825 \u2013 May 24, 1906) was a Republican member of the U.S. House of Representatives from Pennsylvania.\""], ["what about next", "[Thought] After obtaining the CONTEXT, I can utilize the information to verify the fact."], ["what about next", "[Action] \u3010verify(context=CONTEXT, head=HEAD, tail=TAIL)\u3011\n[Return] \"yes\""], ["what about next", "[Thought] After verifying the fact, I should exit this system."]]}

# file = open("./auto_kg/one_thought/train.json")
# lines = file.readlines()
# count = 0
# for line in tqdm(lines):
#     a = json.loads(line)
#     # a = {"prompt": "what about next", "response": "[Action]\u3010exit()\u3011", "history": [["[Thought] To be a fact extractor, I need to start by retrieving sentences from Wikipedia.", "[Action] \u3010get_sentences()\u3011\n[Return] SENTENCES=\"Wilfried \" Willi \" Schneider ( born 13 March 1963 in Media\u0219 , Transylvania ) is a German skeleton racer who competed from 1992 to 2002 . He won two medals in the men 's skeleton event at the FIBT World Championships with a gold in 1998 and a bronze in 1999 . Schneider also finish ninth in the men 's skeleton event at the 2002 Winter Olympics in Salt Lake City . He won the men 's overall Skeleton World Cup title in 1997 - 8 . After retiring from competition Schneider became a coach , leading the Canadian skeleton team to three medals at the 2006 Winter Olympics in Turin ( a gold for Duff Gibson , a silver for Jeff Pain and a bronze for Melissa Hollingsworth ) , and coaching Jon Montgomery to victory in the 2010 Winter Olympics in Vancouver , British Columbia , Canada . In July 2012 Schneider agreed a two - year contract to coach the Russian skeleton team .\""], ["what about next", "[Thought] After obtaining sentences from Wikipedia, it's time to identify all ENTITIES in the SENTENCES."], ["what about next", "[Action] \u3010get_entities(sentences=SENTENCES)\u3011\n[Return] ENTITIES=\"['Wilfried \" Willi \" Schneider', '13 March 1963', 'Media\u0219', 'Transylvania', 'German']\""], ["what about next", "[Thought] With all ENTITIES found, I must select an entity from ENTITIES as the head."], ["what about next", "[Action] \u3010choose_an_entity(entities=ENTITIES)\u3011\n[Return] HEAD=\"2002 Winter Olympics\""], ["what about next", "[Thought] Having selected a HEAD from ENTITIES, I need to determine what TYPES the HEAD may have."], ["what about next", "[Action] \u3010get_types(head=HEAD)\u3011\n[Return] TYPES=\"['MISC']\""], ["what about next", "[Thought] After obtaining HEAD TYPES, it's time to choose a TYPE from TYPES."], ["what about next", "[Action] \u3010choose_a_type(types=TYPES)\u3011\n[Return] TYPE=\"MISC\""], ["what about next", "[Thought] With a TYPE selected from TYPES, I must discover what relations HEAD may have according to TYPE."], ["what about next", "[Action] \u3010get_relations(head=HEAD,type=TYPE)\u3011\n[Return] RELATIONS=\"['start time', 'end time', 'location']\""], ["what about next", "[Thought] Having obtained all RELATIONS, I need to pick a relation from RELATIONS."], ["what about next", "[Action] \u3010choose_a_relation(relations=RELATIONS)\u3011\n[Return] RELATION=\"end time\""], ["what about next", "[Thought] With a RELATION chosen, I must attempt to get RELATION_ALIAS_TEMPLATE."], ["what about next", "[Action] \u3010get_relation_alias_template(relation=RELATION)\u3011\n[Return] RELATION_ALIAS_TEMPLATE={'fall date': \"Given the sentence: '{s[sentence]}', identify the date when {s[head]} fell or ended by selecting the correct option from the provided list: {s[tail_choose]}.\", 'end date': \"Given the sentence: '{s[sentence]}', identify the date when {s[head]} ended or terminated by selecting the correct option from the provided list: {s[tail_choose]}.\"}"], ["what about next", "[Thought] After obtaining RELATION_ALIAS_TEMPLATE, I need to use RELATION_ALIAS_TEMPLATE to get a TAIL."], ["what about next", "[Action] \u3010get_tail(sentence=SENTENCE, relation_alias_template=RELATION_ALIAS_TEMPLATE, head=HEAD)\u3011\n[Return] TAILS=\"['2002', 'German']\""], ["what about next", "[Thought] After get all TAILS, I need to choose a tail from TAILS."], ["what about next", "[Action] \u3010choose_a_tail(tails=TAILS)\u3011\n[Return] TAIL=\"2002\""], ["what about next", "[Thought] With a TAIL acquired, I can search for more information about HEAD and TAIL using a search engine."], ["what about next", "[Action] \u3010search_engine(head=HEAD, tail=TAIL)\u3011\n[Return] CONTEXT=\"2002 Winter Olympics 2002 Winter Olympics Stefan Georgiev () (born 20 May 1977) is an alpine skier from Bulgaria. He competed at the 1998 Winter Olympics, 2002 Winter Olympics, 2006 Winter Olympics and the 2010 Winter Olympics. His best result was a 16th place in the combined in 2002."], ["what about next", "\"\n[Thought] After obtaining the CONTEXT, I can utilize the information to verify the fact."], ["what about next", "[Thought] After obtaining the CONTEXT, I can utilize the information to verify the fact."], ["what about next", "[Action] \u3010verify(context=CONTEXT, head=HEAD, tail=TAIL)\u3011\n[Return] \"yes\""], ["what about next", "[Thought] After verifying the fact, I should exit this system."]]}
#     c =" "
#     for i in a['history']:
#         c+= i[0]+" "+i[1]+" "
#     b = tokenizer.tokenize(c)
#     d = tokenizer.encode(c)
#     if len(d)>2048:
#         count += 1
# print(count)

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
# if match:
#     result = match.group(1)
#     for f in functions:
#         if f in result:
#             method_return = get_api(f)
#             ori_history = new_history
#             ori_input = ori_input + response + "\n[Return] " + method_return
