import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random
import torch


class AugmentationPipeline:
    """
    A modular image augmentation pipeline using Albumentations.
    Supports both light and strong augmentations for few-shot learning tasks.
    """

    def __init__(self, mode='light', image_size=224):
        assert mode in ['light', 'strong'], "Mode must be 'light' or 'strong'"
        self.mode = mode
        self.image_size = image_size
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        if self.mode == 'light':
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
        else:  # strong
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.GaussianBlur(blur_limit=3, p=0.3)
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.ISONoise(p=0.3)
                ], p=0.3),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])

    def __call__(self, image):
        """
        Apply the augmentation pipeline to a PIL image or NumPy array.
        Returns a tensor suitable for PyTorch models.
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        elif isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        augmented = self.pipeline(image=image)
        return augmented['image']


def get_augmentation(mode='light', image_size=224):
    """
    Returns a callable augmentation function based on the specified mode.
    Can be plugged directly into datasets or DataLoaders.
    """
    return AugmentationPipeline(mode=mode, image_size=image_size)
