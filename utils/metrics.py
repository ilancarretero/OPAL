"""
quant_metrics_utils.py

Utility functions to perform model inference and compute quantitative classification metrics.
"""
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize


def evaluate_quant_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a dataset and collect true labels, predicted labels, and predicted probabilities.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for the dataset to evaluate.
        device: Device to perform computation on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple of numpy arrays: (y_true, y_pred, y_probabilities).
    """
    model.eval()
    y_true_list, y_pred_list, y_prob_list = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)

            # Compute softmax probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            y_prob_list.extend(probs.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)
    return y_true, y_pred, y_prob


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification performance metrics from predictions.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        y_prob: Array of predicted probabilities for each class.

    Returns:
        Dictionary of metrics: ACC, SEN, SPE, PPV, NPV, F1, AUC.
    """
    # Standard metrics
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, average='macro', zero_division=0)
    ppv = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # AUC requires one-hot encoding of true labels
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)
    try:
        auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')

    # Confusion matrix based specificity and NPV
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    specificity_list, npv_list = [], []
    TN_FN_sum = cm.sum()
    for idx, cls in enumerate(classes):
        TP = cm[idx, idx]
        FP = cm[:, idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        TN = TN_FN_sum - (TP + FP + FN)

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        specificity_list.append(specificity)
        npv_list.append(npv)

    spe = float(np.mean(specificity_list))
    npv_val = float(np.mean(npv_list))

    return {
        'ACC': float(acc),
        'SEN': float(sen),
        'SPE': spe,
        'PPV': float(ppv),
        'NPV': npv_val,
        'F1': float(f1),
        'AUC': float(auc)
    }


def run_inference_and_save_metrics(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    quant_metrics_dir: str
) -> None:
    """
    Evaluate model on train, validation, and test splits, compute metrics, and save to CSV.

    Args:
        model: Trained PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        device: Device to perform computation on.
        quant_metrics_dir: Directory path to save the quantitative metrics CSV.
    """
    metrics_results = {}
    splits = {'Train': train_loader, 'Validation': val_loader, 'Test': test_loader}

    # Iterate through data splits
    for split_name, loader in splits.items():
        y_true, y_pred, y_prob = evaluate_quant_metrics(model, loader, device)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics_results[split_name] = metrics

    # Convert results to DataFrame and save
    os.makedirs(quant_metrics_dir, exist_ok=True)
    df = pd.DataFrame(metrics_results).T.round(4)
    csv_path = os.path.join(quant_metrics_dir, 'quantitative_metrics.csv')
    df.to_csv(csv_path, index=True)

    print(f"Quantitative metrics saved to: {csv_path}")
    print(df)
