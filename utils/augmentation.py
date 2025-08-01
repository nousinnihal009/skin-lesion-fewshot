import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                               rotate_limit=15, p=0.5, border_mode=0),
        ], p=1.0),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.MotionBlur(p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3)
        ], p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.4),
        A.CLAHE(p=0.2),
        A.CoarseDropout(max_holes=1, max_height=image_size//10,
                        max_width=image_size//10, min_holes=1, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_validation_augmentation(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_weak_augmentation(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
