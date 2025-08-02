import argparse
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models.protonet import ProtoNet
from utils.helpers import compute_prototypes, predict_class
from utils.config_parser import load_config
from utils.loader import load_episode_from_json
from utils.profiling import log_gpu_memory


def evaluate_episode(model, device, episode):
    support_set = episode['support_images'].to(device)
    query_set = episode['query_images'].to(device)
    support_labels = episode['support_labels'].to(device)
    query_labels = episode['query_labels'].to(device)

    with torch.no_grad():
        support_embeddings = model(support_set)
        query_embeddings = model(query_set)

        prototypes = compute_prototypes(support_embeddings, support_labels)
        predictions = predict_class(prototypes, query_embeddings)

    acc = accuracy_score(query_labels.cpu(), predictions.cpu()) * 100
    f1 = f1_score(query_labels.cpu(), predictions.cpu(), average='macro') * 100

    return acc, f1, query_labels.cpu().numpy(), predictions.cpu().numpy()


def test(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ProtoNet().to(device)
    checkpoint = torch.load(config['test']['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load test episodes from JSON
    print("[INFO] Loading test episodes...")
    test_episodes = load_episode_from_json(
        config['test']['episode_json'],
        image_size=config['data'].get('image_size', 224)
    )

    all_acc, all_f1, all_y_true, all_y_pred = [], [], [], []

    print(f"[INFO] Evaluating {len(test_episodes)} test episodes...")
    for idx, episode in enumerate(tqdm(test_episodes, desc="Testing Episodes")):
        acc, f1, y_true, y_pred = evaluate_episode(model, device, episode)
        all_acc.append(acc)
        all_f1.append(f1)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        log_gpu_memory()
        if config['test'].get('verbose', False):
            print(f"Episode {idx+1}: Acc = {acc:.2f}%, F1 = {f1:.2f}%")

    # Final aggregated metrics
    mean_acc = np.mean(all_acc)
    std_acc = np.std(all_acc)
    mean_f1 = np.mean(all_f1)

    print("\n[RESULTS] Final Test Performance:")
    print(f"Avg Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Avg F1 Score: {mean_f1:.2f}%")
    print("\n[INFO] Classification Report:")
    print(classification_report(all_y_true, all_y_pred, digits=4))

    # Save results
    if 'results' in config and 'dir' in config['results']:
        os.makedirs(config['results']['dir'], exist_ok=True)
        output_path = os.path.join(config['results']['dir'], 'test_metrics.json')
        with open(output_path, 'w') as f:
            json.dump({
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_f1": mean_f1,
                "classification_report": classification_report(
                    all_y_true, all_y_pred, digits=4, output_dict=True
                )
            }, f, indent=4)
        print(f"[SAVED] Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    test(args.config)
