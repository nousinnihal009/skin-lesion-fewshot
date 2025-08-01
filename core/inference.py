# core/inference.py

import os
import torch
import argparse
import logging
from utils.data_loader import load_few_shot_test_set
from utils.helpers import compute_prototypes, predict_class
from models.protonet import ProtoNet
from utils.metrics import compute_classification_metrics

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model = ProtoNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logging.info(f"Loaded model from {model_path}")
    return model

def run_inference(model, support_set, support_labels, query_set, query_labels, device):
    with torch.no_grad():
        support_embeddings = model(support_set.to(device))
        query_embeddings = model(query_set.to(device))

        prototypes = compute_prototypes(support_embeddings, support_labels)
        predicted_labels = predict_class(prototypes, query_embeddings)

        acc, prec, rec, f1 = compute_classification_metrics(
            predicted_labels.cpu(), query_labels.cpu()
        )

        logging.info(f"[RESULT] Accuracy: {acc:.2f}%")
        logging.info(f"[RESULT] Precision: {prec:.2f}%")
        logging.info(f"[RESULT] Recall: {rec:.2f}%")
        logging.info(f"[RESULT] F1 Score: {f1:.2f}%")

    return predicted_labels

def main(args):
    setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    support_set, query_set, support_labels, query_labels = load_few_shot_test_set(
        args.test_path, args.n_way, args.k_shot, args.query
    )

    model = load_model(args.model_path, device)
    run_inference(model, support_set, support_labels, query_set, query_labels, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Few-Shot Skin Lesion Classification")
    parser.add_argument("--model_path", type=str, default="models/protonet_best.pth", help="Path to trained model checkpoint")
    parser.add_argument("--test_path", type=str, default="data/test", help="Path to few-shot formatted test dataset")
    parser.add_argument("--n_way", type=int, default=5, help="Number of classes for N-way classification")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of support samples per class (K-shot)")
    parser.add_argument("--query", type=int, default=5, help="Number of query samples per class")
    
    args = parser.parse_args()
    main(args)
