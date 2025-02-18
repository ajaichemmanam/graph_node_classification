from typing import Literal


class ModelConfig:
    # Random seed for reproducibility
    seed: int = 42

    # General training parameters
    epochs: int = 200
    patience: int = 20
    checkpoint_path: str = "checkpoints/best_model.pt"

    # Model selection
    model_type: Literal["gcn", "sage"] = "sage"

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
