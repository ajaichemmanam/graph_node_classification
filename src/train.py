import torch
from models import MODEL_REGISTRY
from data.dataset import load_data
from utils.trainer import Trainer
from utils.early_stopping import EarlyStopping
from utils.seed import set_seed
from config.model_config import Config


def create_model(model_type, num_features, num_classes):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Model type {model_type} not supported. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]

    if model_type == "sage":
        return model_class(
            num_features=num_features,
            hidden_channels=Config.hidden_channels,
            num_classes=num_classes,
            dropout_rate=Config.dropout_rate,
            aggregator=Config.sage_aggregator,
        )
    elif model_type == "gcn":
        return model_class(
            num_features=num_features,
            hidden_channels=Config.hidden_channels,
            num_classes=num_classes,
            dropout_rate=Config.dropout_rate,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    # Set random seed for reproducibility
    set_seed(Config.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset, data = load_data(root="data", name=Config.dataset_name, device=device)

    print(f"Dataset: {dataset}")
    print(f"Number of Classes: {dataset.num_classes}")
    print(f"Number of Node Features: {dataset.num_node_features}")

    # Initialize model
    model = create_model(
        model_type=Config.model_type,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
    ).to(device)

    print(
        f"Training {Config.model_type.upper()} model on {Config.dataset_name} dataset"
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    # Initialize trainer and early stopping
    trainer = Trainer(model, optimizer, device)
    early_stopping = EarlyStopping(
        patience=Config.patience,
        model_path=Config.checkpoint_path,
        monitor=Config.monitor,
        mode=Config.mode,
    )

    # Training loop
    for epoch in range(1, Config.epochs + 1):
        loss = trainer.train_step(data)
        train_acc, val_acc, test_acc = trainer.test(data)

        # Early stopping check based on configured metric
        monitor_value = val_acc if Config.monitor == "accuracy" else loss
        test_metric = test_acc if Config.monitor == "accuracy" else loss

        if early_stopping(monitor_value, test_metric, model):
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            metric_name = "Accuracy" if Config.monitor == "accuracy" else "Loss"
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, "
                f"Val: {val_acc:.4f}, Test: {test_acc:.4f}, "
                f"Best Val {metric_name}: {early_stopping.best_score:.4f}"
            )

    final_metric = early_stopping.best_test_metric
    metric_name = "Accuracy" if Config.monitor == "accuracy" else "Loss"
    print(f"Final Test {metric_name}: {final_metric:.4f}")


if __name__ == "__main__":
    main()
