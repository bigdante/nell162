from sklearn.metrics import precision_recall_fscore_support
import logging
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from transformers import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP


def get_looger(args):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if args.train:
        mode = "train"
    else:
        mode = "inference"
    if args.save_ckpt_path:
        file_handler = logging.FileHandler(f'{args.save_ckpt_path}/{timestamp}_{args.train_mode}_{mode}_{args.experiment}.txt', mode='w', encoding='utf-8', delay=True)
    else:
        file_handler = logging.FileHandler(f'{args.load_ckpt_path}/{timestamp}_{args.train_mode}_{mode}_{args.experiment}.txt', mode='w', encoding='utf-8', delay=True)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_model(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    model.to(args.device)
    if args.local_rank is not None:
        model = DDP(model, device_ids=[args.global_rank], output_device=args.global_rank)
    return model, optimizer, tokenizer


def run_vf_valid_test(model=None, args=None, dataloader=None, tokenizer=None, mode="valid"):
    if args.local_rank is not None:
        model = DDP(model, device_ids=[args.global_rank], output_device=args.global_rank)
    device = args.device
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            _, batch_preds = torch.max(outputs.logits, dim=1)
            preds.extend(batch_preds.tolist())
            labels.extend(batch_labels.tolist())
        precision, recall, f1, _ = precision_recall_fscore_support(preds, labels, average='binary')
        print(f"valid | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
