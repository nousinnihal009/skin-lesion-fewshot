import argparse
import torch
import os
from torchvision import transforms
from PIL import Image
from utils.helpers import compute_prototypes, predict_class
from models.protonet import ProtoNet
from utils.augmentation import get_validation_augmentation
from glob import glob
import numpy as np
from tqdm import tqdm


def load_few_shot_test_set(root_dir, n_way, k_shot, query, image_size=224):
    transform = get_validation_augmentation(image_size=image_size)
    support_set, query_set = [], []
    support_labels, query_labels = [], []

    class_folders = sorted(os.listdir(root_dir))[:n_way]

    label_map = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

    for cls in class_folders:
        cls_path = os.path.join(root_dir, cls)
        all_images = glob(os.path.join(cls_path, "*.jpg"))
        selected_images = all_images[:k_shot + query]
        support_imgs = selected_images[:k_shot]
        query_imgs = selected_images[k_shot:k_shot + query]

        for img_path in support_imgs:
            img = np.array(Image.open(img_path).convert("RGB"))
            img_tensor = transform(image=img)["image"]
            support_set.append(img_tensor)
            support_labels.append(label_map[cls])

        for img_path in query_imgs:
            img = np.array(Image.open(img_path).convert("RGB"))
            img_tensor = transform(image=img)["image"]
            query_set.append(img_tensor)
            query_labels.append(label_map[cls])

    support_set = torch.stack(support_set)
    query_set = torch.stack(query_set)
    support_labels = torch.tensor(support_labels)
    query_labels = torch.tensor(query_labels)

    return support_set, query_set, support_labels, query_labels


def run_demo(model_path, test_path, n_way=3, k_shot=5, query=5, image_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading few-shot test set...")
    support_set, query_set, support_labels, query_labels = load_few_shot_test_set(
        test_path, n_way=n_way, k_shot=k_shot, query=query, image_size=image_size
    )

    print("[INFO] Loading model...")
    model = ProtoNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("[INFO] Computing prototypes...")
    support_embeddings = model(support_set.to(device))
    prototypes = compute_prototypes(support_embeddings, support_labels.to(device))

    print("[INFO] Running inference on query samples...")
    query_embeddings = model(query_set.to(device))
    predicted_labels = predict_class(prototypes, query_embeddings)

    correct = (predicted_labels == query_labels.to(device)).sum().item()
    total = len(query_labels)
    accuracy = correct / total * 100

    print(f"[RESULT] Accuracy on few-shot test set: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/protonet_best.pth", help="Path to trained model")
    parser.add_argument("--test_path", type=str, default="data/test", help="Path to few-shot test data")
    parser.add_argument("--n_way", type=int, default=3, help="Number of classes (N-way)")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of shots per class (K-shot)")
    parser.add_argument("--query", type=int, default=5, help="Number of query samples per class")
    parser.add_argument("--image_size", type=int, default=224, help="Size to resize images to")
    args = parser.parse_args()

    run_demo(args.model_path, args.test_path, args.n_way, args.k_shot, args.query, args.image_size)
