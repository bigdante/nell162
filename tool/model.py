import os
from typing import Dict, Union, Optional, List
import torch
from torch.nn import Module
from transformers import AutoModel, AutoConfig, AutoTokenizer

ptuning_checkpoint = 'neptune/ChatGLM-6B-main/ptuning/output/adgen-chatglm-6b-pt-one_2048-64-1e-2/checkpoint-37000'
checkpoint_path = "THUDM/chatglm-6b"


def auto_configure_device_map(gpus: List[int]) -> Dict[str, int]:
    num_gpus = len(gpus)
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus
    device_map = {'transformer.word_embeddings': gpus[0],
                  'transformer.final_layernorm': gpus[0], 'lm_head': gpus[0]}
    used = 2
    gpu_target_index = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target_index += 1
            used = 0
        assert gpu_target_index < num_gpus
        device_map[f'transformer.layers.{i}'] = gpus[gpu_target_index]
        used += 1
    device_map['transformer.prefix_encoder.embedding.weight'] = gpus[-1]
    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2, device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model
        config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        config.pre_seq_len = 64
        config.prefix_projection = False
        model = AutoModel.from_pretrained(checkpoint_path, config=config, trust_remote_code=True, **kwargs).half()
        prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)
        model.transformer.prefix_encoder.embedding.weight.data = model.transformer.prefix_encoder.embedding.weight.data.to(model.device)
    return model


class ModelAndTokenizerSingleton:
    _instance = None
    _model = None
    _tokenizer = None

    @staticmethod
    def getInstance():
        if ModelAndTokenizerSingleton._instance is None:
            ModelAndTokenizerSingleton._instance = ModelAndTokenizerSingleton()
        return ModelAndTokenizerSingleton._instance

    def load_model(self):
        if self._model is None:
            print("Loading model...")
            model = load_model_on_gpus(checkpoint_path, 4, auto_configure_device_map([1, 2, 3, 4]))
            model = model.eval()
            self._model = model
        return self._model

    def load_tokenizer(self):
        if self._tokenizer is None:
            print("Loading tokenizer...")
            # Load your tokenizer here
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            self._tokenizer = tokenizer
        return self._tokenizer


singleton_instance = ModelAndTokenizerSingleton.getInstance()
model = singleton_instance.load_model()
tokenizer = singleton_instance.load_tokenizer()
