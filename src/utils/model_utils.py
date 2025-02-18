import torch

from config.model_config import Config
from models import MODEL_REGISTRY


def create_model(model_type, num_features, num_classes, device=None):
    """Create a new model instance.

    Args:
        model_type (str): Type of model ('gcn' or 'sage')
        num_features (int): Number of input features
        num_classes (int): Number of output classes
        device (torch.device, optional): Device to put the model on

    Returns:
        torch.nn.Module: The created model
    """
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
    elif model_type == "gcn":
        model = model_class(
            num_features=num_features,
            hidden_channels=Config.hidden_channels,
            num_classes=num_classes,
            dropout_rate=Config.dropout_rate,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if device is not None:
        model = model.to(device)

    return model


def load_model(model_path, model_type, num_features, num_classes, device=None):
    """Load a model from checkpoint.

    Args:
        model_path (str): Path to the model checkpoint
        model_type (str): Type of model ('gcn' or 'sage')
        num_features (int): Number of input features
        num_classes (int): Number of output classes
        device (torch.device, optional): Device to put the model on

    Returns:
        torch.nn.Module: The loaded model
    """
    model = create_model(model_type, num_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    if device is not None:
        model = model.to(device)

    return model
