import ipdb
import torch.distributed as dist
import copy
import json
import torch
import random
import argparse
import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="neptune-small", help="T5 model name to use for fine-tuning")
    parser.add_argument("--local_rank", default=None, help="local rank")
    parser.add_argument("--epochs", default=0, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="Training and evaluation batch size")
    parser.add_argument("--adapter", action="store_true", help="Use adapter if set to True, default is False")
    parser.add_argument("--experiment", default="country", help="experiment_name and adapter_name")
    parser.add_argument("--train", action="store_true", help="for training")
    parser.add_argument("--train_mode", default="vf", help="for training mode, vf or gen")
    parser.add_argument("--train_data_path", default="./for_vf_train_data/", help="train for_vf_train_data path")
    parser.add_argument("--inference_data_path", default="", help="inference for_vf_train_data path")
    parser.add_argument("--inference", action="store_true", help="only inference")
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--warmup_proportion", default=0.1, help="to warm up to lr")
    parser.add_argument("--top_p", default=0.9, help="get top_p answer")
    parser.add_argument("--max_length", default=512, type=int, help="max_length")
    parser.add_argument("--gen_save_path", default="", help="save_gen_data for next vf step")
    parser.add_argument("--vf_save_path", default="", help="save_vf_data")
    parser.add_argument("--save_ckpt_path", default="", help="save_ckpt_path")
    parser.add_argument("--load_ckpt_path", default=None, help="load_ckpt_path")
    args = parser.parse_args()
    args.local_rank = os.environ.get('LOCAL_RANK', None)
    args.global_rank = int(os.environ['RANK']) if args.local_rank is not None else None
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def init_distributed_training(args):
    if args.local_rank is not None:
        global_rank = int(os.environ['RANK'])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(global_rank)
        device = torch.device(f"cuda:{global_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl')


def get_data(args):
    data = []
    with open(args.train_data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print("load make_COT_traindata_redocred done")
    if args.global_rank == 0:
        args.logger.info(f"loading train make_COT_traindata_redocred from {args.train_data_path}")
    if args.train_mode == "vf":
        train_data = [(s['input'], s['label']) for s in data]
        counter = Counter(item[1] for item in train_data)
        label_weight = counter["No"] / counter["Yes"]
    args.label_weight = label_weight
    random.shuffle(train_data)
    args.data_length = len(data)
    return train_data, label_weight


class CustomDataset(Dataset):
    def __init__(self, args, data=None, tokenizer=None, max_input_length=None, label_weigh=1):
        self.data = data
        self.label_weigh = label_weigh
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        input_tokens = self.tokenizer.encode_plus(input_text, max_length=self.max_input_length, truncation=True, padding='max_length', return_tensors='pt')
        label = self.tokenizer.convert_tokens_to_ids(target_text)
        label_weight = self.label_weigh if target_text != "No" else 1.0
        return {
            'input_text': input_text,
            'input_ids': input_tokens['input_ids'].squeeze(),
            'attention_mask': input_tokens['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long).unsqueeze(0),
            'label_weight': torch.tensor(label_weight)
        }



def get_data_loader(args, tokenizer):
    if args.train:
        data, label_weigh = get_data(args)
        dataset = CustomDataset(args, data=data, tokenizer=tokenizer, max_input_length=args.max_length, label_weigh=label_weigh)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        train_sampler = DistributedSampler(train_dataset) if args.local_rank is not None else None
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader


def get_entity_sentence_id(vertex):
    sentence_id = []
    for entity in vertex:
        sentence_id.append(entity['sent_id'])
    return list(set(sentence_id))


def get_entity_choose(item, vertex_i, tail_constrain):
    tail_choose = {}
    tail_choose_ids = {}
    entity_list = copy.deepcopy(item['vertexSet'])
    for index, i in enumerate(entity_list):
        if i == vertex_i:
            continue
        tail_types = str(sorted(list(set([tail['type'] for tail in i]))))
        if tail_types in tail_constrain:
            entity_name = [tail['name'] for tail in i]
            sentence_ids = [tail['sent_id'] for tail in i]
            entity_name = list(set(entity_name))
            sentence_ids = list(set(sentence_ids))
            tail_choose[index] = max(entity_name, key=len)
            tail_choose_ids[index] = sentence_ids
        else:
            continue
    if tail_choose:
        tail_choose[len(entity_list)] = "None"
    sentence_id = sorted(list(set([x for lst in tail_choose_ids.values() for x in lst])))
    return tail_choose, sentence_id
