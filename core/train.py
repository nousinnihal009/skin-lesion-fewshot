# core/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import random
import numpy as np
from utils.fewshot_dataset import FewShotDataset
from utils.proto_net import PrototypicalNetwork
from utils.sampler import EpisodicSampler
from utils.metrics import compute_metrics, log_results
from utils.visualizer import plot_loss_curve
from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_one_episode(model, data, optimizer, criterion, device):
    support_images, support_labels, query_images, query_labels = data
    support_images = support_images.to(device)
    query_images = query_images.to(device)
    support_labels = support_labels.to(device)
    query_labels = query_labels.to(device)

    model.train()
    optimizer.zero_grad()
    output = model(support_images, support_labels, query_images)
    loss = criterion(output, query_labels)
    loss.backward()
    optimizer.step()

    pred = torch.argmax(output, dim=1)
    acc = (pred == query_labels).float().mean().item()

    return loss.item(), acc

def main():
    config = load_config("config/config.yaml")
    set_seed(config['experiment']['seed'])

    device = torch.device(config['experiment']['device'])
    dataset = FewShotDataset(config['dataset'])
    sampler = EpisodicSampler(dataset, config['fewshot'])
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], sampler=sampler)

    model = PrototypicalNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    all_losses = []
    for epoch in range(config['train']['epochs']):
        epoch_loss = 0
        epoch_acc = 0
        loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}")
        for episode_data in loop:
            loss, acc = train_one_episode(model, episode_data, optimizer, criterion, device)
            epoch_loss += loss
            epoch_acc += acc
            loop.set_postfix(loss=loss, acc=acc)

        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_acc / len(dataloader)
        all_losses.append(avg_loss)

        print(f"Epoch {epoch+1} Summary: Loss={avg_loss:.4f} | Accuracy={avg_acc:.4f}")

    os.makedirs(os.path.dirname(config['train']['save_path']), exist_ok=True)
    torch.save(model.state_dict(), config['train']['save_path'])
    plot_loss_curve(all_losses, save_path="results/loss_curve.png")

if __name__ == "__main__":
    main()
