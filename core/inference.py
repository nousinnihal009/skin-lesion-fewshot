# core/inference.py

import torch
import yaml
from torchvision import transforms
from PIL import Image
from utils.proto_net import PrototypicalNetwork
from utils.helpers import prepare_single_episode
import os

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def infer(model, image_path, support_set, device):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    query_image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    support_images, support_labels = prepare_single_episode(support_set, transform, device)

    with torch.no_grad():
        output = model(support_images, support_labels, query_image)
        pred = torch.argmax(output, dim=1).item()

    return pred

def main():
    config = load_config("config/config.yaml")
    device = torch.device(config['experiment']['device'])

    model = PrototypicalNetwork().to(device)
    model.load_state_dict(torch.load(config['train']['save_path'], map_location=device))

    image_path = config['demo']['sample_image']
    support_set = [
        ("data/class1/img1.jpg", 0),
        ("data/class1/img2.jpg", 0),
        ("data/class2/img1.jpg", 1),
        ("data/class2/img2.jpg", 1),
        # Add more support samples as needed
    ]

    pred_class = infer(model, image_path, support_set, device)
    print(f"Predicted class index: {pred_class}")

if __name__ == "__main__":
    main()
