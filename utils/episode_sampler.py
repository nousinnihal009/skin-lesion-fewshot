import random
from collections import defaultdict

def create_episode(data, n_way=5, k_shot=1, q_queries=5):
    """
    Creates a single few-shot learning episode.

    Args:
        data (dict): Dictionary with class labels as keys and list of image paths as values.
        n_way (int): Number of classes per episode.
        k_shot (int): Number of support images per class.
        q_queries (int): Number of query images per class.

    Returns:
        dict: Episode with support and query sets.
    """
    selected_classes = random.sample(list(data.keys()), n_way)

    support_set = []
    query_set = []

    for cls in selected_classes:
        samples = random.sample(data[cls], k_shot + q_queries)
        support_set.extend([(s, cls) for s in samples[:k_shot]])
        query_set.extend([(q, cls) for q in samples[k_shot:]])

    return {
        "support": support_set,
        "query": query_set,
        "classes": selected_classes
    }

def group_images_by_class(image_paths, labels):
    """
    Organizes images by their class labels.

    Args:
        image_paths (list): List of image paths.
        labels (list): Corresponding class labels.

    Returns:
        dict: Dictionary of class -> list of images
    """
    grouped = defaultdict(list)
    for img, label in zip(image_paths, labels):
        grouped[label].append(img)
    return grouped
