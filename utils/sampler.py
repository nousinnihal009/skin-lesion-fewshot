import random
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from utils.loader import ISICDataset


def split_by_class(dataset):
    """
    Splits dataset into a dictionary where keys are class labels and values are list of indices.
    """
    class_to_indices = defaultdict(list)
    for idx, (_, _, class_name) in enumerate(dataset):
        class_to_indices[class_name].append(idx)
    return class_to_indices


def create_episode(class_to_indices, n_way, k_shot, query):
    """
    Samples a single episode: n_way classes, each with k_shot support and query examples.
    """
    selected_classes = random.sample(list(class_to_indices.keys()), n_way)
    support_images, support_labels = [], []
    query_images, query_labels = [], []

    label_map = {cls: i for i, cls in enumerate(selected_classes)}

    for cls in selected_classes:
        indices = random.sample(class_to_indices[cls], k_shot + query)
        support_indices = indices[:k_shot]
        query_indices = indices[k_shot:]

        support_images.extend(support_indices)
        support_labels.extend([label_map[cls]] * k_shot)

        query_images.extend(query_indices)
        query_labels.extend([label_map[cls]] * query)

    return support_images, support_labels, query_images, query_labels


class FewShotDataset(torch.utils.data.Dataset):
    """
    Wrapper for few-shot episodic training data generation.
    """
    def __init__(self, dataset, episodes, n_way, k_shot, query):
        self.dataset = dataset
        self.episodes = episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.query = query
        self.class_to_indices = split_by_class(dataset)

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        support_ids, support_labels, query_ids, query_labels = create_episode(
            self.class_to_indices, self.n_way, self.k_shot, self.query
        )

        support_images = torch.stack([self.dataset[i][0] for i in support_ids])
        query_images = torch.stack([self.dataset[i][0] for i in query_ids])
        return support_images, query_images, torch.tensor(support_labels), torch.tensor(query_labels)


def get_few_shot_dataloader(train_path, val_path, n_way, k_shot, query, episodes, num_workers=4):
    """
    Returns DataLoaders for episodic training and validation.
    """
    train_set = ISICDataset(train_path)
    val_set = ISICDataset(val_path)

    train_sampler = FewShotDataset(train_set, episodes, n_way, k_shot, query)
    val_sampler = FewShotDataset(val_set, episodes // 2, n_way, k_shot, query)

    train_loader = DataLoader(train_sampler, batch_size=1, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_sampler, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
