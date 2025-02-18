# Model Loading and Inference

### Command

```bash
python src/infer.py
```

### Loading Saved Model

```python
model = load_model(
    model_path=Config.checkpoint_path,
    model_type=Config.model_type,
    num_features=dataset.num_features,
    num_classes=dataset.num_classes,
    device=device
)
```

### Inference Process

```python
@torch.no_grad()
def inference(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    return pred
```
