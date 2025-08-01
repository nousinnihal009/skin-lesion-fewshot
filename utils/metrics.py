import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

class FewShotMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(labels.cpu().numpy())

    def compute(self):
        if not self.predictions:
            return {}

        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        return {
            "accuracy": np.mean(preds == targets) * 100,
            "f1_score": f1_score(targets, preds, average='macro') * 100,
            "precision": precision_score(targets, preds, average='macro') * 100,
            "recall": recall_score(targets, preds, average='macro') * 100,
            "confusion_matrix": confusion_matrix(targets, preds)
        }

    def report(self):
        report_dict = self.compute()
        print("[METRICS] Evaluation Report:")
        for key, value in report_dict.items():
            if key == "confusion_matrix":
                print(f"{key}:\n{value}")
            else:
                print(f"{key}: {value:.2f}")
        return report_dict

    def classification_report(self):
        return classification_report(self.targets, self.predictions, digits=4)

