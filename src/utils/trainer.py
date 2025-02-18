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
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, mask):
        self.model.eval()
        out = self.model(data)
        pred = out.argmax(dim=1)
        accuracy = (pred[mask] == data.y[mask]).float().mean()
        loss = F.nll_loss(out[mask], data.y[mask])
        return accuracy.item(), loss.item()
