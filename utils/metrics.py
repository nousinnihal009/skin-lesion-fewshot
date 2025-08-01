import json
import os
import logging
from collections import defaultdict
from typing import Dict, Any, Optional


class FewShotMetrics:
    """
    Modular and extensible metrics tracker for few-shot learning tasks.
    Tracks losses, accuracies, and supports JSON serialization and logging.
    """

    def __init__(self) -> None:
        self.metrics = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list)
        }

    def update_train(self, loss: float, acc: float) -> None:
        self.metrics["train"]["loss"].append(loss)
        self.metrics["train"]["accuracy"].append(acc)

    def update_val(self, loss: float, acc: float) -> None:
        self.metrics["val"]["loss"].append(loss)
        self.metrics["val"]["accuracy"].append(acc)

    def update_test(self, loss: float, acc: float) -> None:
        self.metrics["test"]["loss"].append(loss)
        self.metrics["test"]["accuracy"].append(acc)

    def get_latest(self, phase: str) -> Dict[str, float]:
        """
        Returns the latest recorded loss and accuracy for the given phase.
        """
        if phase not in self.metrics:
            raise ValueError(f"Invalid phase '{phase}'")
        loss = self.metrics[phase]["loss"][-1] if self.metrics[phase]["loss"] else None
        acc = self.metrics[phase]["accuracy"][-1] if self.metrics[phase]["accuracy"] else None
        return {"loss": loss, "accuracy": acc}

    def get_summary(self, phase: str) -> Dict[str, float]:
        """
        Returns the mean loss and accuracy for the given phase.
        """
        if phase not in self.metrics:
            raise ValueError(f"Invalid phase '{phase}'")
        loss_list = self.metrics[phase]["loss"]
        acc_list = self.metrics[phase]["accuracy"]

        summary = {
            "mean_loss": sum(loss_list) / len(loss_list) if loss_list else 0.0,
            "mean_accuracy": sum(acc_list) / len(acc_list) if acc_list else 0.0
        }
        return summary

    def save(self, path: str) -> None:
        """
        Saves metrics to a JSON file at the specified path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        logging.info(f"[METRICS] Saved metrics to {path}")

    def load(self, path: str) -> None:
        """
        Loads metrics from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metrics file not found at: {path}")
        with open(path, "r") as f:
            self.metrics = json.load(f)
        logging.info(f"[METRICS] Loaded metrics from {path}")

    def reset(self, phase: Optional[str] = None) -> None:
        """
        Resets metrics for a given phase or all phases.
        """
        if phase:
            if phase in self.metrics:
                self.metrics[phase] = defaultdict(list)
                logging.info(f"[METRICS] Reset metrics for phase: {phase}")
            else:
                raise ValueError(f"Invalid phase '{phase}'")
        else:
            for key in self.metrics:
                self.metrics[key] = defaultdict(list)
            logging.info(f"[METRICS] Reset metrics for all phases")

    def log_latest(self, phase: str, episode: int) -> None:
        """
        Logs the latest metrics using Python's logging system.
        """
        latest = self.get_latest(phase)
        if latest["loss"] is not None and latest["accuracy"] is not None:
            logging.info(f"[{phase.upper()}] Episode {episode}: Loss = {latest['loss']:.4f}, "
                         f"Accuracy = {latest['accuracy']:.2f}%")

    def export_summary_dict(self) -> Dict[str, Any]:
        """
        Returns a clean summary dict of all metrics (for use in external reporting or APIs).
        """
        return {
            phase: self.get_summary(phase)
            for phase in self.metrics
        }
