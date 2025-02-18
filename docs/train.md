
# Training Process

## Configuration

### Model Parameters
- Hidden channels: 64
- Dropout rate: 0.5
- Learning rate: 0.01
- Weight decay: 5e-4

### Training Settings
- Maximum epochs: 200
- Early stopping:
  - Patience: 20
  - Monitor: "loss" (can be "loss" or "accuracy")
  - Mode: "min" (for loss) or "max" (for accuracy)
- Random seed: 42
- Model checkpoint path: checkpoints/best_model.pt

## Training Pipeline

### 1. Initialization
```python
# Set random seed for reproducibility
set_seed(42)

# Initialize model and move to device (CPU/GPU)
model = create_model(
    model_type=Config.model_type,
    num_features=dataset.num_features,
    num_classes=dataset.num_classes,
    device=device
)

# Setup optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=Config.learning_rate,
    weight_decay=Config.weight_decay
)
```

### 2. Training Loop
- For each epoch:
  1. Forward pass and loss calculation using NLL loss
  2. Backward pass and parameter updates
  3. Evaluation on train and validation sets
  4. Early stopping check based on configured metric (loss or accuracy)
  5. Save best model if monitored metric improves

### 3. Early Stopping
- Monitors validation metric (configurable: loss or accuracy)
- Saves best model state to configured checkpoint path
- Stops training if no improvement for specified patience period
- Maintains record of best metric
