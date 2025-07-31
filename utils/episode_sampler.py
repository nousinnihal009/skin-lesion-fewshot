import random
from collections import defaultdict
from torch.utils.data import Dataset

class FewShotEpisodeSampler(Dataset):
    def __init__(self, data, n_way, k_shot, q_queries, episodes_per_epoch):
        """
        Args:
            data: List of (image, label) tuples
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            q_queries: Number of query examples per class
            episodes_per_epoch: How many episodes per epoch
        """
        self.data = data
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.episodes_per_epoch = episodes_per_epoch

        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(data):
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, index):
        selected_classes = random.sample(self.labels, self.n_way)

        support_set = []
        query_set = []

        for cls in selected_classes:
            indices = random.sample(self.label_to_indices[cls], self.k_shot + self.q_queries)
            support_idxs = indices[:self.k_shot]
            query_idxs = indices[self.k_shot:]

            support_set.extend(support_idxs)
            query_set.extend(query_idxs)

        return support_set, query_set
