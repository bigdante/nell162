import json
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import os


def read_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print("load train_auto_glm_data done")
    return data


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['input']
        label = 0 if self.data[idx]['label'] == 'No' else 1
        encoding = self.tokenizer.encode_plus(
            text,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train(model, dataloader, criterion, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.logits, batch_labels)
            total_loss += loss.item()
            _, batch_preds = torch.max(outputs.logits, dim=1)
            preds.extend(batch_preds.tolist())
            labels.extend(batch_labels.tolist())

    return total_loss / len(dataloader), preds, labels


def main(rank, world_size, cuda_devices):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(cuda_devices[rank])
    device = torch.device(f"cuda:{cuda_devices[rank]}")
    data = read_data('./for_vf_train_data/country of citizenship.json')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cuda_devices[rank]], output_device=cuda_devices[rank])

    # 多卡训练
    batch_size = 32
    train_dataset = MyDataset(train_data, tokenizer)
    val_dataset = MyDataset(val_data, tokenizer)
    test_dataset = MyDataset(test_data, tokenizer)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler)

    # 计算类权重
    # print("===============")
    num_yes = sum(1 for d in train_data if d['label'] == 'Yes')
    num_no = sum(1 for d in train_data if d['label'] == 'No')
    # print(num_no, num_yes, num_no + num_yes)
    class_weights = torch.tensor([num_yes / (num_yes + num_no), num_no / (num_yes + num_no)], dtype=torch.float).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 20
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_f1 = 0.0
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device, scheduler)
        val_loss, val_preds, val_labels = evaluate(model, val_dataloader, criterion, device)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, pos_label=1, average='binary')
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), "best_checkpoint.pt")

    # Load the best model and evaluate on the test set
    best_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    best_model = nn.DataParallel(best_model, device_ids=cuda_devices).to(device)
    best_model.load_state_dict(torch.load("best_checkpoint.pt"))

    test_loss, test_preds, test_labels = evaluate(best_model, test_dataloader, criterion, device)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, pos_label=1, average='binary')

    print("Test results:")
    print(f"Test Loss: {test_loss:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f}")


if __name__ == '__main__':

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    cuda_devices = [6,7]
    world_size = len(cuda_devices)
    mp.spawn(main, args=(world_size, cuda_devices), nprocs=world_size, join=True)
