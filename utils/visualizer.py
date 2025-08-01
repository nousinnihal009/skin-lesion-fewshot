import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_tsne(embeddings, labels, save_path=None):
    """
    Plot t-SNE of embeddings.
    """
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    reduced = tsne.fit_transform(embeddings.cpu().numpy())
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels.cpu().numpy(), cmap='tab10', s=12)
    plt.colorbar(scatter, ticks=range(len(torch.unique(labels))))
    plt.title("t-SNE of Embeddings")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(preds, targets, class_names, save_path=None):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(targets.cpu(), preds.cpu(), labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    if save_path:
        plt.savefig(save_path)
    plt.close()
