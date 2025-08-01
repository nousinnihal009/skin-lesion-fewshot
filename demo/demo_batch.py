import os
import yaml
import torch
import logging
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.protonet import ProtoNet
from utils.helpers import compute_prototypes
from utils.config_parser import load_config
from utils.augmentation import get_test_transforms
from utils.visualizer import visualize_prediction
from utils.logger import setup_logging

def load_images_from_folder(folder_path, image_size):
    transform = get_test_transforms(image_size)
    images, paths = [], []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            image = Image.open(path).convert("RGB")
            images.append(transform(image))
            paths.append(path)
    return torch.stack(images), paths

def run_demo(config_path="demo/demo_config.yaml"):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Logging
    if config["logging"]["verbose"]:
        setup_logging(config["logging"]["log_path"])
    logging.info("[START] Running Batch Demo")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ProtoNet().to(device)
    model.load_state_dict(torch.load(config["model"]["checkpoint_path"], map_location=device))
    model.eval()

    # Load images
    images, paths = load_images_from_folder(config["data"]["input_dir"], config["data"]["image_size"])
    images = images.to(device)

    # Dummy support set (you can adapt this with actual few-shot labels)
    # In a real few-shot demo, you'd need few support samples for each class
    support = images[:5]
    support_labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long).to(device)
    query = images[5:]
    query_paths = paths[5:]

    # Inference
    with torch.no_grad():
        support_embeddings = model(support)
        query_embeddings = model(query)
        prototypes = compute_prototypes(support_embeddings, support_labels)
        logits = -torch.cdist(query_embeddings, prototypes)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    # Save results
    if config["output"]["save_predictions"]:
        output_path = config["output"]["predictions_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame({
            "Image": [os.path.basename(p) for p in query_paths],
            "Predicted_Class": predictions
        })
        df.to_csv(output_path, index=False)
        logging.info(f"[SAVED] Predictions written to {output_path}")

    # Save visualizations
    if config["output"]["save_visualizations"]:
        vis_dir = config["output"]["visualizations_dir"]
        os.makedirs(vis_dir, exist_ok=True)
        for path, pred in zip(query_paths, predictions):
            visualize_prediction(path, pred, vis_dir)

    logging.info("[DONE] Demo completed.")

if __name__ == "__main__":
    run_demo()
