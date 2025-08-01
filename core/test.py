# core/test.py

import torch
import yaml
from torch.utils.data import DataLoader
from utils.fewshot_dataset import FewShotDataset
from utils.proto_net import PrototypicalNetwork
from utils.sampler import EpisodicSampler
from utils.metrics import compute_metrics, log_results
from tqdm import tqdm

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for support_images, support_labels, query_images, query_labels in tqdm(dataloader, desc="Testing"):
            support_images = support_images.to(device)
            query_images = query_images.to(device)
            support_labels = support_labels.to(device)

            output = model(support_images, support_labels, query_images)
            pred = torch.argmax(output, dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(query_labels.numpy())

    return all_preds, all_labels

def main():
    config = load_config("config/config.yaml")
    device = torch.device(config['experiment']['device'])

    dataset = FewShotDataset(config['dataset'], split="test")
    sampler = EpisodicSampler(dataset, config['fewshot'])
    dataloader = DataLoader(dataset, batch_size=config['test']['batch_size'], sampler=sampler)

    model = PrototypicalNetwork().to(device)
    model.load_state_dict(torch.load(config['train']['save_path'], map_location=device))

    preds, labels = evaluate(model, dataloader, device)
    results = compute_metrics(labels, preds)
    log_results(results, "results/test_metrics.json")

    print("Test Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
