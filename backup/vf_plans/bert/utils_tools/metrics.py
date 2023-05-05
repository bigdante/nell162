import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import time

def cm(predictions, labels):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=["No", "Yes"]).ravel()
    return tn, fp, fn, tp


def compute_metrics_vf(labels, preds):
    yes_labels = [label for label in labels if label == "Yes"]
    yes_preds = [pred for i, pred in enumerate(preds) if labels[i] == "Yes"]
    if len(yes_labels) > 0:
        acc = accuracy_score(yes_labels, yes_preds)
    else:
        acc = np.nan
    # acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, pos_label='Yes', zero_division=0)
    recall = recall_score(labels, preds, pos_label='Yes', zero_division=0)
    precision = precision_score(labels, preds, pos_label='Yes', zero_division=0)
    tn, fp, fn, tp = cm(labels, preds)
    return acc, f1, recall, precision, tp, fp, tn, fn


def compute_metrics_gen(y_true, y_pred, args):
    start = time.time()
    y_true_filter = [label for label in y_true if label != (14794,)]
    y_pred_filter = [pred for i, pred in enumerate(y_pred) if y_true[i] != (14794,)]
    if not y_true_filter:
        return 0, 0, 0, 0
    mlb = MultiLabelBinarizer()
    y_true_mlb = mlb.fit_transform([{label} for label in y_true_filter])
    y_pred_mlb = mlb.transform([{label} for label in y_pred_filter])
    acc = accuracy_score(y_true_mlb, y_pred_mlb)
    precision = precision_score(y_true_mlb, y_pred_mlb, average='micro')
    recall = recall_score(y_true_mlb, y_pred_mlb, average='micro')
    f1 = f1_score(y_true_mlb, y_pred_mlb, average='micro')
    args.logger.info(f"metrics compute done, take time {time.time() - start:.2f}")
    return acc, f1, recall, precision
