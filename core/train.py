import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import get_few_shot_dataloader
from utils.helpers import compute_prototypes, compute_loss, accuracy
from models.protonet import ProtoNet
from utils.early_stopper import EarlyStopper
from utils.metrics import FewShotMetrics
from utils.config_parser import load_config
from utils.profiling import profile_section, log_gpu_memory


def train(config_path):
    # Load config
    config = load_config(config_path)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(log_dir=config['train']['log_dir'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load few-shot dataloaders
    train_loader, val_loader = get_few_shot_dataloader(
        config['data']['train_path'],
        config['data']['val_path'],
        n_way=config['fewshot']['n_way'],
        k_shot=config['fewshot']['k_shot'],
        query=config['fewshot']['query'],
        episodes=config['train']['episodes'],
        num_workers=config['train']['num_workers']
    )

    # Initialize model, optimizer, and utilities
    model = ProtoNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    stopper = EarlyStopper(patience=config['train']['patience'])
    metrics = FewShotMetrics()

    # Training loop
    for episode in tqdm(range(config['train']['episodes']), desc="Training Episodes"):
        model.train()
        support_set, query_set, support_labels, query_labels = next(iter(train_loader))
        support_set, query_set = support_set.to(device), query_set.to(device)
        support_labels, query_labels = support_labels.to(device), query_labels.to(device)

        optimizer.zero_grad()
        with profile_section("Forward Pass"):
            support_embeddings = model(support_set)
            query_embeddings = model(query_set)

        log_gpu_memory()

        prototypes = compute_prototypes(support_embeddings, support_labels)
        loss, logits = compute_loss(prototypes, query_embeddings, query_labels)
        acc = accuracy(logits, query_labels)

        loss.backward()
        optimizer.step()

        writer.add_scalar("Train/Loss", loss.item(), episode)
        writer.add_scalar("Train/Accuracy", acc, episode)
        metrics.update_train(loss.item(), acc)

        # Validation
        if episode % config['train']['val_interval'] == 0:
            model.eval()
            val_loss_list, val_acc_list = [], []

            with torch.no_grad():
                for val_support, val_query, val_s_labels, val_q_labels in val_loader:
                    val_support, val_query = val_support.to(device), val_query.to(device)
                    val_s_labels, val_q_labels = val_s_labels.to(device), val_q_labels.to(device)

                    with profile_section("Validation Forward Pass"):
                        val_support_embeddings = model(val_support)
                        val_query_embeddings = model(val_query)

                    val_prototypes = compute_prototypes(val_support_embeddings, val_s_labels)
                    val_loss, val_logits = compute_loss(val_prototypes, val_query_embeddings, val_q_labels)
                    val_acc = accuracy(val_logits, val_q_labels)

                    val_loss_list.append(val_loss.item())
                    val_acc_list.append(val_acc)

                val_loss_mean = np.mean(val_loss_list)
                val_acc_mean = np.mean(val_acc_list)
                writer.add_scalar("Val/Loss", val_loss_mean, episode)
                writer.add_scalar("Val/Accuracy", val_acc_mean, episode)
                metrics.update_val(val_loss_mean, val_acc_mean)

                logging.info(f"Episode {episode}: Val Loss = {val_loss_mean:.4f}, Val Acc = {val_acc_mean:.2f}%")

                if stopper.early_stop(val_loss_mean):
                    logging.info("[STOP] Early stopping triggered.")
                    break

    # Save model
    model_path = os.path.join(config['train']['checkpoint_dir'], 'protonet_best.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"[DONE] Model saved to {model_path}")
    writer.close()

    # Save training metrics
    metrics.save(os.path.join(config['results']['dir'], 'train_metrics.json'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    train(args.config)
