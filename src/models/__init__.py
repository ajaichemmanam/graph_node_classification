from .gcn import GCN
from .sage import GraphSAGE

MODEL_REGISTRY = {"gcn": GCN, "sage": GraphSAGE}
