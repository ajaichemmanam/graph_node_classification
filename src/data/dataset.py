from torch_geometric.datasets import Planetoid


def load_data(root="data", name="Cora", device="cuda"):
    dataset = Planetoid(root=root, name=name)
    data = dataset[0].to(device)
    return dataset, data
