import os
import re
import gradio as gr
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
import torch
from parse_api import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

cuda_device = torch.device("cuda:3") if torch.cuda.is_available() else "cpu"
model_name_or_path = "THUDM/chatglm-6b"
ptuning_checkpoint = './output/adgen-chatglm-6b-ft-1e-2/checkpoint-19400'
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
history_output = " "


def handle_input(user_input):
    global history_output
    ori_input = user_input
    ori_history = []
    while True:
        response, new_history = model.chat(tokenizer, ori_input, history=ori_history)
        history_output += '/n' + str(new_history)
        return response, history_output
        # print(response)
        # match = re.search(r'【([^【,]+)', response)
        # if match:
        #     result = match.group(1)
        #     if "exit()" in result:
        #         history_output.append("All done")
        #         return response, "\n".join(history_output)
        #     flag = 0
        #     for f in functions:
        #         if f in result:
        #             method_return = get_api(f)
        #             ori_history = new_history
        #             ori_input = ori_input + response + "\n[Return] " + method_return
        #             flag = 1
        #             history_output.append(response)
        #             return response, "\n".join(history_output)
        #     if not flag:
        #         history_output.append("no method match")
        #         return "no method match", "\n".join(history_output)
        # else:
        #     history_output += "\nno method offered"
        #     return "no method offered", "\n".join(history_output)


iface = gr.Interface(
    fn=handle_input,
    inputs=gr.components.Textbox(lines=5, label="Input"),
    outputs=[gr.components.Textbox(label="Current Output"),
             gr.components.Textbox(label="History Output")],
    title="AUTO KG",
)
iface.launch(share=True, debug=True)
