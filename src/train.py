import torch

from config.model_config import Config
from data.dataset import load_data
from utils.early_stopping import EarlyStopping
from utils.model_utils import create_model
from utils.seed import set_seed
from utils.trainer import Trainer


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
        device=device,
    )

    print(f"Training {Config.model_type.upper()} model")

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
        train_acc, train_loss = trainer.evaluate(data, data.train_mask)
        val_acc, val_loss = trainer.evaluate(data, data.val_mask)

        # Early stopping check based on configured metric
        monitor_value = val_acc if Config.monitor == "accuracy" else val_loss
        test_metric = val_acc if Config.monitor == "accuracy" else val_loss

        if early_stopping(monitor_value, test_metric, model):
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            metric_name = "Accuracy" if Config.monitor == "accuracy" else "Loss"
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, "
                f"Val: {val_acc:.4f}, Best Val {metric_name}: {early_stopping.best_score:.4f}"
            )

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(Config.checkpoint_path, weights_only=True))
    test_acc, test_loss = trainer.evaluate(data, data.test_mask)
    print(f"Final Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
