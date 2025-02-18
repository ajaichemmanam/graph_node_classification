import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import Linear
import torch.nn as nn

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout_rate=0.5, aggregator='mean'):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels, aggr=aggregator)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr=aggregator)
        self.out = Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.out(x)
        return x