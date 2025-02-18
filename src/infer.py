import torch
from models import MODEL_REGISTRY
from data.dataset import load_data
from config.model_config import Config


def load_model(model_path, model_type, num_features, num_classes, device):
    """Load a trained model from checkpoint."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Model type {model_type} not supported. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]

    if model_type == "sage":
        model = model_class(
            num_features=num_features,
            hidden_channels=Config.hidden_channels,
            num_classes=num_classes,
            dropout_rate=Config.dropout_rate,
            aggregator=Config.sage_aggregator,
        )
    else:  # GCN
        model = model_class(
            num_features=num_features,
            hidden_channels=Config.hidden_channels,
            num_classes=num_classes,
            dropout_rate=Config.dropout_rate,
        )

    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(device)


@torch.no_grad()
def inference(model, data):
    """Perform inference using the trained model."""
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    return pred


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset, data = load_data(root="data", name=Config.dataset_name, device=device)

    # Load trained model
    model = load_model(
        model_path="checkpoints/best_model.pt",
        model_type=Config.model_type,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        device=device,
    )

    # Perform inference
    predictions = inference(model, data)
    print(f"Predictions: {predictions}")
    label = data.y
    print(f"True labels: {label}")
    accuracy = (predictions == label).float().mean()
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
