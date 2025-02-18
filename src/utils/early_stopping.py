import torch
import os
from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    monitor: str
    model_path: str
    mode: str = "max"
    best_score: float | None = None
    counter: int = 0
    best_test_metric: float | None = None

    def __post_init__(self):
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))

    def __call__(self, val_metric, test_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_metric, model)
            return False

        if self.mode == "max":
            improvement = score > self.best_score
        else:  # mode == 'min'
            improvement = score < self.best_score

        if improvement:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(test_metric, model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    def save_checkpoint(self, test_metric, model):
        """Save model checkpoint."""
        torch.save(model.state_dict(), self.model_path)
        self.best_test_metric = test_metric
