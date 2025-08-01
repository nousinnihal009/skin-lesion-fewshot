import torch
import torch.nn.functional as F


def compute_prototypes(embeddings, labels):
    """
    Compute class prototypes by averaging support embeddings for each class.
    """
    classes = torch.unique(labels)
    prototypes = torch.stack([embeddings[labels == cls].mean(0) for cls in classes])
    return prototypes


def compute_loss(prototypes, query_embeddings, query_labels):
    """
    Compute negative log-likelihood loss based on distances to prototypes.
    """
    distances = euclidean_distance(query_embeddings, prototypes)
    log_probs = F.log_softmax(-distances, dim=1)
    loss = F.nll_loss(log_probs, query_labels)
    return loss, log_probs


def predict_class(prototypes, query_embeddings):
    """
    Predict class by computing distances to prototypes.
    """
    distances = euclidean_distance(query_embeddings, prototypes)
    return torch.argmin(distances, dim=1)


def euclidean_distance(a, b):
    """
    Compute pairwise Euclidean distance between two sets of vectors.
    """
    n, m = a.size(0), b.size(0)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return torch.pow(a - b, 2).sum(2)


def accuracy(logits, labels):
    """
    Compute classification accuracy.
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()
