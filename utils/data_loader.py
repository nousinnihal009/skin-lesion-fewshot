import os
import json
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class FewShotISICDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        """
        Args:
            data_dict (dict): Dictionary of image path -> label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = list(data_dict.items())
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_json_split(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def build_label_index_mapping(labels):
    """
    Creates a label to index mapping.
    """
    unique_labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique_labels)}

def prepare_data_dict(image_dir, label_dict):
    """
    Args:
        image_dir (str): Directory where class folders exist.
        label_dict (dict): Mapping of image filename to label.

    Returns:
        dict: Mapping of full image path to label.
    """
    data_dict = {}
    for fname, label in label_dict.items():
        full_path = os.path.join(image_dir, label, fname)
        if os.path.exists(full_path):
            data_dict[full_path] = label
    return data_dict

def get_default_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
