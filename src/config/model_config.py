from typing import Literal


class Config:
    # Random seed for reproducibility
    seed: int = 42

    # General training parameters
    epochs: int = 200

    # Early stopping settings
    monitor: Literal["accuracy", "loss"] = "loss"  # Monitor metric for early stopping
    mode: Literal["min", "max"] = "min"  # 'min' for loss, 'max' for accuracy
    patience: int = 20
    checkpoint_path: str = "checkpoints/best_model.pt"

    # Model selection
    model_type: Literal["gcn", "sage"] = "gcn"

    # Model hyperparameters
    hidden_channels: int = 64
    dropout_rate: float = 0.5

    # GraphSAGE specific
    sage_aggregator: Literal["mean", "max", "sum"] = "mean"

    # Optimizer parameters
    learning_rate: float = 0.01
    weight_decay: float = 5e-4

    # Dataset parameters
    dataset_name: str = "Cora"

