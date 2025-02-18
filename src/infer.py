import torch

from config.model_config import Config
from data.dataset import load_data
from utils.model_utils import load_model


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
        model_path=Config.checkpoint_path,
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
