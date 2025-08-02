import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
from utils.sampler import FewShotSampler
from utils.augmentation import get_training_augmentation, get_validation_augmentation
import json
from utils.augmentation import get_validation_augmentation

class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


def load_data_from_directory(data_dir):
    image_paths, labels = [], []
    label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(data_dir)))}

    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(label_map[label])
    return image_paths, labels


def get_few_shot_dataloader(
    train_dir, val_dir, n_way, k_shot, query, episodes, num_workers=4, image_size=224
):
    train_paths, train_labels = load_data_from_directory(train_dir)
    val_paths, val_labels = load_data_from_directory(val_dir)

    train_transform = get_training_augmentation(image_size=image_size)
    val_transform = get_validation_augmentation(image_size=image_size)

    train_dataset = SkinLesionDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = SkinLesionDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=FewShotSampler(train_labels, n_way, k_shot, query, episodes),
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=FewShotSampler(val_labels, n_way, k_shot, query, episodes),
        num_workers=num_workers
    )

    return train_loader, val_loader

def load_episode_from_json(json_path, image_size=224):
    with open(json_path, 'r') as f:
        episode_data = json.load(f)

    transform = get_validation_augmentation(image_size=image_size)
    episodes = []

    for ep in episode_data['episodes']:
        support_images, support_labels = [], []
        query_images, query_labels = [], []

        label_map = {cls: idx for idx, cls in enumerate(ep['classes'])}

        for cls in ep['support']:
            for img_path in ep['support'][cls]:
                img = np.array(Image.open(img_path).convert("RGB"))
                img = transform(image=img)["image"]
                support_images.append(img)
                support_labels.append(label_map[cls])

        for cls in ep['query']:
            for img_path in ep['query'][cls]:
                img = np.array(Image.open(img_path).convert("RGB"))
                img = transform(image=img)["image"]
                query_images.append(img)
                query_labels.append(label_map[cls])

        episode = {
            'support_images': torch.stack(support_images),
            'query_images': torch.stack(query_images),
            'support_labels': torch.tensor(support_labels),
            'query_labels': torch.tensor(query_labels),
        }

        episodes.append(episode)

    return episodes
