import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def compute_accuracy(preds, targets):
    """
    Compute top-1 accuracy.
    Args:
        preds: Predicted logits or probabilities (Tensor)
        targets: Ground truth labels (Tensor)
    Returns:
        accuracy: Float
    """
    pred_labels = torch.argmax(preds, dim=1)
    correct = (pred_labels == targets).sum().item()
    return correct / len(targets)

def compute_confusion(preds, targets):
    """
    Compute confusion matrix.
    Returns:
        np.ndarray
    """
    pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
    true_labels = targets.cpu().numpy()
    return confusion_matrix(true_labels, pred_labels)

def get_classification_report(preds, targets, target_names=None):
    """
    Returns classification report string.
    """
    pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
    true_labels = targets.cpu().numpy()
    return classification_report(true_labels, pred_labels, target_names=target_names)
