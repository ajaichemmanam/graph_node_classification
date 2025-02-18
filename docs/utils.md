# Utilities Documentation

## 1. Early Stopping (`early_stopping.py`)

Prevents overfitting by monitoring validation performance.

### Usage

```python
early_stopping = EarlyStopping(
    patience=20,
    model_path='checkpoints/best_model.pt',
    monitor='loss',  # or 'accuracy'
    mode='min'  # 'min' for loss, 'max' for accuracy
)
if early_stopping(val_metric, test_metric, model):
    print("Early stopping triggered")
```

### Features

- Tracks best validation metric (loss or accuracy)
- Stores corresponding test metric
- Implements patience mechanism
- Saves best model checkpoint
- Returns stop signal when triggered

## 2. Model Utils (`model_utils.py`)

Handles model creation and loading operations.

### create_model()

Creates a new model instance with specified configuration.

```python
model = create_model(
    model_type='gcn',  # or 'sage'
    num_features=dataset.num_features,
    num_classes=dataset.num_classes,
    device=device  # optional
)
```

#### Parameters

- `model_type`: Type of model ('gcn' or 'sage')
- `num_features`: Number of input features
- `num_classes`: Number of output classes
- `device`: Device to put the model on (optional)

#### Features

- Supports GCN and GraphSAGE architectures
- Configurable hidden channels and dropout rate
- Automatic device placement
- GraphSAGE-specific aggregator selection

### load_model()

Loads a trained model from a checkpoint.

```python
model = load_model(
    model_path='checkpoints/best_model.pt',
    model_type='gcn',  # or 'sage'
    num_features=dataset.num_features,
    num_classes=dataset.num_classes,
    device=device  # optional
)
```

#### Parameters

- `model_path`: Path to the model checkpoint
- `model_type`: Type of model ('gcn' or 'sage')
- `num_features`: Number of input features
- `num_classes`: Number of output classes
- `device`: Device to put the model on (optional)

#### Features

- Loads model weights from checkpoint
- Recreates model architecture
- Automatic device placement

## 3. Trainer (`trainer.py`)

Handles model training and evaluation.

### Key Methods

- `train_step()`: Single training iteration
- `test()`: Evaluation on all data splits

### Implementation Details

- Uses cross-entropy loss
- Handles gradient computation
- Manages device placement
- Calculates accuracy metrics

## 4. Seed Setting (`seed.py`)

Ensures reproducibility across runs.

### Components Set

- Python's random
- NumPy random
- PyTorch random
- CUDA random (if available)
- CUDNN settings

### Usage

```python
set_seed(42)  # Set before any random operations
```

## 5. Model Registry

Maintains available models and handles model creation.

### Supported Models

- GCN
- GraphSAGE

### Registration

```python
MODEL_REGISTRY = {
    'gcn': GCN,
    'sage': GraphSAGE
}
```
