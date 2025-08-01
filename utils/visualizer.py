# utils/visualizer.py

import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch

def draw_prediction(img: Image.Image, text: str, font_path=None):
    """
    Draws a label text on top of an image and returns the modified image.
    """
    draw = ImageDraw.Draw(img)
    font_size = max(14, img.size[0] // 20)

    try:
        font = ImageFont.truetype(font_path or "arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    text_width, text_height = draw.textsize(text, font=font)
    x, y = 10, 10
    draw.rectangle([x - 5, y - 5, x + text_width + 5, y + text_height + 5], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 0), font=font)

    return img


def visualize_prediction(image_path, predicted_class, save_dir, font_path=None):
    """
    Overlays prediction label on image and saves to `save_dir`.
    """
    os.makedirs(save_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    img = draw_prediction(img, f"Prediction: {predicted_class}", font_path)

    save_path = os.path.join(save_dir, f"pred_{os.path.basename(image_path)}")
    img.save(save_path)
    return save_path


def visualize_augmentation(original_tensor, augmented_tensor, save_path="aug_comparison.png", font_path=None):
    """
    Combines original and augmented images side-by-side for comparison.
    Saves the result to `save_path`.
    """
    to_pil = transforms.ToPILImage()
    original_img = to_pil(original_tensor.cpu().squeeze())
    augmented_img = to_pil(augmented_tensor.cpu().squeeze())

    original_img = draw_prediction(original_img, "Original", font_path)
    augmented_img = draw_prediction(augmented_img, "Augmented", font_path)

    combined_width = original_img.width + augmented_img.width
    combined_height = max(original_img.height, augmented_img.height)

    combined = Image.new("RGB", (combined_width, combined_height))
    combined.paste(original_img, (0, 0))
    combined.paste(augmented_img, (original_img.width, 0))

    combined.save(save_path)
    return save_path
