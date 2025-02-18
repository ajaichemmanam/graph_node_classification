import torch
import os

class EarlyStopping:
    def __init__(self, patience=20, model_path='checkpoints/best_model.pt'):
        self.patience = patience
        self.counter = 0
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.model_path = model_path
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def __call__(self, val_acc, test_acc, model):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_test_acc = test_acc
            self.counter = 0
            # Save the model
            torch.save(model.state_dict(), self.model_path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
