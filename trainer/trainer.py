from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
# from seqeval.metrics import precision_score, recall_score, f1_score

def flat_accuracy(preds, labels):
    """
    Tính độ chính xác cho bài toán phân loại.
    Args:
        preds (numpy.ndarray): Mảng dự đoán (logits hoặc xác suất).
        labels (numpy.ndarray): Mảng nhãn thực tế.
    Returns:
        float: Độ chính xác.
    """
    # Chọn nhãn dự đoán (class với xác suất cao nhất)
    preds_flat = np.argmax(preds, axis=1)
    labels_flat = labels
    return np.sum(preds_flat == labels_flat) / len(labels_flat)

def compute_metrics(preds, labels, label_map=None):
    """
    Tính toán Precision, Recall, và F1 cho bài toán phân loại.
    Args:
        preds (numpy.ndarray): Mảng dự đoán (logits hoặc xác suất).
        labels (numpy.ndarray): Mảng nhãn thực tế.
        label_map (dict): (Tùy chọn) ánh xạ từ chỉ số nhãn sang tên nhãn.
    Returns:
        dict: Precision, Recall, F1-score.
    """
    # Chọn nhãn dự đoán (class với xác suất cao nhất)
    preds_flat = np.argmax(preds, axis=1)  # Chọn lớp có xác suất cao nhất
    labels_flat = labels

    # Nếu có ánh xạ nhãn, chuyển đổi
    if label_map is not None:
        pred_tags = [label_map[p] for p in preds_flat]
        label_tags = [label_map[l] for l in labels_flat]
    else:
        pred_tags = preds_flat
        label_tags = labels_flat

    # Kiểm tra nếu pred_tags và label_tags có kiểu dữ liệu phù hợp
    pred_tags = np.array(pred_tags)
    label_tags = np.array(label_tags)

    # Tính toán Precision, Recall, F1
    precision = precision_score(label_tags, pred_tags, average='weighted')
    recall = recall_score(label_tags, pred_tags, average='weighted')
    f1 = f1_score(label_tags, pred_tags, average='weighted')

    # Tính toán độ chính xác
    accuracy = np.sum(preds_flat == labels_flat) / len(labels_flat)

    return {"Accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def compute_metrics_macro(preds, labels, label_map=None):
    """
    Tính toán Precision, Recall, và F1 cho bài toán phân loại.
    Args:
        preds (numpy.ndarray): Mảng dự đoán (logits hoặc xác suất).
        labels (numpy.ndarray): Mảng nhãn thực tế.
        label_map (dict): (Tùy chọn) ánh xạ từ chỉ số nhãn sang tên nhãn.
    Returns:
        dict: Precision, Recall, F1-score.
    """
    # Chọn nhãn dự đoán (class với xác suất cao nhất)
    preds_flat = np.argmax(preds, axis=1)  # Chọn lớp có xác suất cao nhất
    labels_flat = labels

    # Nếu có ánh xạ nhãn, chuyển đổi
    if label_map is not None:
        pred_tags = [label_map[p] for p in preds_flat]
        label_tags = [label_map[l] for l in labels_flat]
    else:
        pred_tags = preds_flat
        label_tags = labels_flat

    # Kiểm tra nếu pred_tags và label_tags có kiểu dữ liệu phù hợp
    pred_tags = np.array(pred_tags)
    label_tags = np.array(label_tags)

    # Tính toán Precision, Recall, F1
    precision = precision_score(label_tags, pred_tags, average='macro')
    recall = recall_score(label_tags, pred_tags, average='macro')
    f1 = f1_score(label_tags, pred_tags, average='macro')

    # Tính toán độ chính xác
    accuracy = np.sum(preds_flat == labels_flat) / len(labels_flat)

    return {"Accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def _f1(preds, labels):
    preds_flat = np.argmax(preds, axis=1)
    labels_flat = labels
    preds_flat = np.array(preds_flat)
    labels_flat = np.array(labels_flat)

    f1 = f1_score(labels_flat, preds_flat, average='weighted')

    return f1

class trainer():
    def train(self, model, dataloader, epoch, epochs, writer, criterion, optimizer, device, max_grad_norm, scheduler=None, grad_clip=True):
        progress_bar = tqdm(dataloader, colour = '#800080', ncols = 120)
        total_loss = 0
        total_samples = 0
        b_errors = 0
        model.train()
        for iter, batch in enumerate(progress_bar):
            inputs, labels = batch['input_ids'].to(device), batch['targets'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss+=loss.item()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_samples+=inputs.size(0)
            progress_bar.set_description(f"TRAIN | Epoch: {epoch+1}/{epochs} | Iter: {iter+1}/{len(dataloader)} | Error: {b_errors}/{len(dataloader)} | Loss: {(total_loss/total_samples):.4f}")
        writer.add_scalar('Train/Loss', total_loss/total_samples, epoch+1)
    
    def validation(self, model, dataloader, criterion, device):
        model.eval()
        eval_loss, nb_eval_steps = 0, 0
        all_preds = []
        all_labels = []
        for batch in dataloader:
            inputs, labels = batch['input_ids'].to(device), batch['targets'].to(device)
            with torch.no_grad():
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.to('cpu').numpy())
            eval_loss += loss.mean().item()

            nb_eval_steps += 1

        f1 = _f1(all_preds, all_labels)
        return eval_loss/nb_eval_steps, f1
    
    def evaluate_model(self, model , dataloader, label_map, device):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch['input_ids'].to(device), batch['targets'].to(device)
                outputs = model(inputs)
                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        metrics = compute_metrics(all_preds, all_labels, label_map)
        metrics_macro = compute_metrics_macro(all_preds, all_labels, label_map)
        return metrics, metrics_macro
    

    