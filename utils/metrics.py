import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def compute_accuracy(y_true, y_pred):
    """
    Computes the classification accuracy.
    """
    return accuracy_score(y_true, y_pred)

def compute_confusion_matrix(y_true, y_pred, labels=None):
    """
    Computes the confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)

def top_k_accuracy(probs, targets, k=5):
    """
    Computes the Top-K accuracy.
    """
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum([t in top_k for t, top_k in zip(targets, top_k_preds)])
    return correct / len(targets)
