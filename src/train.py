import torch
from models import MODEL_REGISTRY
from data.dataset import load_data
from utils.trainer import Trainer
from utils.early_stopping import EarlyStopping
from utils.seed import set_seed
from config.model_config import ModelConfig


def create_model(model_type, num_features, num_classes):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Model type {model_type} not supported. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]

    if model_type == "sage":
        return model_class(
            num_features=num_features,
            hidden_channels=ModelConfig.hidden_channels,
            num_classes=num_classes,
            dropout_rate=ModelConfig.dropout_rate,
            aggregator=ModelConfig.sage_aggregator,
        )
    else:  # GCN
        return model_class(
            num_features=num_features,
            hidden_channels=ModelConfig.hidden_channels,
            num_classes=num_classes,
            dropout_rate=ModelConfig.dropout_rate,
        )


def main():
    # Set random seed for reproducibility
    set_seed(ModelConfig.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset, data = load_data(root="data", name=ModelConfig.dataset_name, device=device)

    # Initialize model
    model = create_model(
        model_type=ModelConfig.model_type,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
    ).to(device)

    print(
        f"Training {ModelConfig.model_type.upper()} model on {ModelConfig.dataset_name} dataset"
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ModelConfig.learning_rate,
        weight_decay=ModelConfig.weight_decay,
    )

    # Initialize trainer and early stopping
    trainer = Trainer(model, optimizer, device)
    early_stopping = EarlyStopping(
        patience=ModelConfig.patience, model_path=ModelConfig.checkpoint_path
    )

    # Training loop
    for epoch in range(1, ModelConfig.epochs + 1):
        loss = trainer.train_step(data)
        train_acc, val_acc, test_acc = trainer.test(data)

        # Early stopping check with model saving
        if early_stopping(val_acc, test_acc, model):
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, "
                f"Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )

    print(f"Final Test Accuracy: {early_stopping.best_test_acc:.4f}")


if __name__ == "__main__":
    main()
