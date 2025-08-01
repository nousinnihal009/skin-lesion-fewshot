import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.augmentation import get_training_augmentation

def show_augmented_samples(image_path, n_samples=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    aug = get_training_augmentation(image_size=image.shape[0])
    plt.figure(figsize=(15, 5))

    for i in range(n_samples):
        augmented = aug(image=image)['image']
        img_np = augmented.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        plt.subplot(1, n_samples, i + 1)
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f"Aug {i + 1}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image for augmentation demo")
    args = parser.parse_args()
    show_augmented_samples(args.image_path)
