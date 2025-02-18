import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data)
        # Apply log_softmax before nll_loss
        log_prob = F.log_softmax(out, dim=1)
        loss = F.nll_loss(log_prob[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        out = self.model(data)
        pred = out.argmax(dim=1)

        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        return train_acc, val_acc, test_acc
