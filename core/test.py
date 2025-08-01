import argparse
import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models.protonet import ProtoNet
from utils.helpers import compute_prototypes, predict_class
from utils.augmentation import get_validation_augmentation
from utils.config_parser import load_config


def load_test_data(root_dir, n_way, k_shot, query, image_size=224):
    transform = get_validation_augmentation(image_size=image_size)
    support_set, query_set = [], []
    support_labels, query_labels = [], []

    class_folders = sorted(os.listdir(root_dir))[:n_way]
    label_map = {cls: idx for idx, cls in enumerate(class_folders)}

    for cls in class_folders:
        cls_path = os.path.join(root_dir, cls)
        all_images = sorted([
            os.path.join(cls_path, img) for img in os.listdir(cls_path)
            if img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        selected = all_images[:k_shot + query]
        support_imgs = selected[:k_shot]
        query_imgs = selected[k_shot:k_shot + query]

        for img_path in support_imgs:
            img = np.array(Image.open(img_path).convert("RGB"))
            tensor_img = transform(image=img)["image"]
            support_set.append(tensor_img)
            support_labels.append(label_map[cls])

        for img_path in query_imgs:
            img = np.array(Image.open(img_path).convert("RGB"))
            tensor_img = transform(image=img)["image"]
            query_set.append(tensor_img)
            query_labels.append(label_map[cls])

    support_set = torch.stack(support_set)
    query_set = torch.stack(query_set)
    support_labels = torch.tensor(support_labels)
    query_labels = torch.tensor(query_labels)

    return support_set, query_set, support_labels, query_labels


def test(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading model...")
    model = ProtoNet().to(device)
    model.load_state_dict(torch.load(config['test']['checkpoint_path'], map_location=device))
    model.eval()

    print("[INFO] Preparing test set...")
    support_set, query_set, support_labels, query_labels = load_test_data(
        config['data']['test_path'],
        n_way=config['fewshot']['n_way'],
        k_shot=config['fewshot']['k_shot'],
        query=config['fewshot']['query'],
        image_size=config['data'].get('image_size', 224)
    )

    support_set = support_set.to(device)
    query_set = query_set.to(device)
    support_labels = support_labels.to(device)
    query_labels = query_labels.to(device)

    print("[INFO] Computing prototypes and evaluating...")
    with torch.no_grad():
        support_embeddings = model(support_set)
        query_embeddings = model(query_set)

        prototypes = compute_prototypes(support_embeddings, support_labels)
        predicted = predict_class(prototypes, query_embeddings)

        acc = accuracy_score(query_labels.cpu(), predicted.cpu()) * 100
        f1 = f1_score(query_labels.cpu(), predicted.cpu(), average='macro') * 100

    print(f"[RESULT] Accuracy: {acc:.2f}% | F1 Score: {f1:.2f}%")
    print("[INFO] Classification Report:")
    print(classification_report(query_labels.cpu(), predicted.cpu(), digits=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    test(args.config)
