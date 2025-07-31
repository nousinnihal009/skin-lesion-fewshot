import argparse
import torch
from utils.data_loader import load_few_shot_test_set
from models.protonet import ProtoNet
from utils.helpers import compute_prototypes, predict_class

def run_demo(model_path, test_path, n_way=3, k_shot=5, query=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading few-shot test set...")
    support_set, query_set, support_labels, query_labels = load_few_shot_test_set(
        test_path, n_way=n_way, k_shot=k_shot, query=query
    )

    print("[INFO] Loading model...")
    model = ProtoNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("[INFO] Computing prototypes...")
    support_embeddings = model(support_set.to(device))
    prototypes = compute_prototypes(support_embeddings, support_labels)

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
    args = parser.parse_args()

    run_demo(args.model_path, args.test_path, args.n_way, args.k_shot, args.query)
