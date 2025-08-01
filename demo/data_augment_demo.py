# demo/data_augment_demo.py

import os
import torch
from torchvision import transforms
from PIL import Image
from utils.augmentation import get_advanced_augmentations
from utils.visualizer import visualize_augmentation

def load_images_from_folder(folder, limit=None):
    """
    Loads all image file paths from a folder.
    """
    supported = ['.png', '.jpg', '.jpeg', '.bmp']
    image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in supported]
    return image_paths[:limit] if limit else image_paths

def preview_augmentations(image_paths, augment_fn, save_dir="results/aug_preview"):
    """
    Applies augmentations and saves original vs augmented comparisons.
    """
    os.makedirs(save_dir, exist_ok=True)
    to_tensor = transforms.ToTensor()

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            image_tensor = to_tensor(image).unsqueeze(0)  # Shape: (1, C, H, W)

            augmented = augment_fn(image_tensor.clone())[0].detach()

            save_path = os.path.join(save_dir, f"aug_{os.path.basename(path)}")
            visualize_augmentation(image_tensor[0], augmented, save_path=save_path)
            print(f"[✓] Saved: {save_path}")
        except Exception as e:
            print(f"[!] Failed for {path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="demo/sample_images", help="Path to sample image folder")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of images")
    args = parser.parse_args()

    print("[INFO] Loading images...")
    image_paths = load_images_from_folder(args.image_dir, limit=args.limit)

    print("[INFO] Applying augmentations...")
    augmentation_fn = get_advanced_augmentations()

    preview_augmentations(image_paths, augmentation_fn)
