import os
import glob
from PIL import Image
import numpy as np

def load_image(path):
    """Loads an image and returns it as a normalized NumPy array."""
    image = Image.open(path).convert('RGB')
    return np.asarray(image).astype(np.float32) / 255.0

def load_dataset(folder_path, class_limit=None):
    """
    Loads dataset from the folder path assuming subfolders are class names.
    """
    dataset = {}
    classes = sorted(os.listdir(folder_path))[:class_limit] if class_limit else os.listdir(folder_path)

    for cls in classes:
        class_path = os.path.join(folder_path, cls)
        image_paths = glob.glob(os.path.join(class_path, '*.jpg'))
        dataset[cls] = image_paths

    return dataset
