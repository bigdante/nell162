import json
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification


def read_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print("load train_auto_glm_data done")
    return data[:10000]


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['query']['prompt']
        # text = self.train_auto_glm_data[idx]['input']
        label = 0 if self.data[idx]['label'] == 'fp' else 1
        # label = 0 if self.train_auto_glm_data[idx]['label'] == 'No' else 1
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


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
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


def main():
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    data = read_data('./for_inference_data/country of citizenship.json')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MyDataset(data, tokenizer)
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    model = nn.DataParallel(model, device_ids=[7]).to(device)
    model.load_state_dict(torch.load("best_checkpoint.pt"))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_preds, test_labels = evaluate(model, dataloader, criterion, device)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, pos_label=1, average='binary')
    print(f"Test Loss: {test_loss:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
