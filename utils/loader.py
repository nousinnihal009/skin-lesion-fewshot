import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class ISICDataset(Dataset):
    """
    Custom Dataset for ISIC images used in Few-Shot Learning.
    Automatically handles label encoding and transforms.
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.image_paths = []
        self.labels = []

        for class_name in sorted(os.listdir(image_dir)):
            class_path = os.path.join(image_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_name)

        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.encoded_labels[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label), self.labels[idx]  # include human-readable class
