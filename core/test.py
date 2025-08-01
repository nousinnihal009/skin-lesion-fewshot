import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm

from utils.data_loader import load_few_shot_test_set
from models.protonet import ProtoNet
from utils.helpers import compute_prototypes, predict_class
from utils.logger import get_logger


def evaluate_few_shot_task(model, support_set, query_set, support_labels, query_labels, device):
    model.eval()
    
    with torch.no_grad():
        support_embeddings = model(support_set.to(device))
        query_embeddings = model(query_set.to(device))
        prototypes = compute_prototypes(support_embeddings, support_labels)
        predictions = predict_class(prototypes, query_embeddings)

    correct = (predictions == query_labels.to(device)).sum().item()
    total = len(query_labels)
    acc = correct / total * 100

    report = classification_report(query_labels.cpu(), predictions.cpu(), output_dict=True)
    return acc, report


def main(args):
    logger = get_logger("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading test set...")
    support_set, query_set, support_labels, query_labels = load_few_shot_test_set(
        args.test_path, n_way=args.n_way, k_shot=args.k_shot, query=args.query
    )

    logger.info("Loading model...")
    model = ProtoNet(backbone=args.backbone).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    logger.info("Evaluating few-shot task...")
    acc, report = evaluate_few_shot_task(model, support_set, query_set, support_labels, query_labels, device)

    logger.info(f"[RESULT] Few-shot classification accuracy: {acc:.2f}%")
    logger.info(f"[METRICS] Classification Report:\n{classification_report(query_labels.cpu(), predict_class(compute_prototypes(model(support_set.to(device)), support_labels), model(query_set.to(device))).cpu())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot Skin Lesion Classification Evaluation")
    parser.add_argument("--model_path", type=str, default="models/protonet_best.pth")
    parser.add_argument("--test_path", type=str, default="data/test")
    parser.add_argument("--n_way", type=int, default=3)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50", "convnet"])
    args = parser.parse_args()

    main(args)
