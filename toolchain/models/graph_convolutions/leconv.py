# labels: test_group::turnkey name::leconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The local extremum graph neural network operator from the
`"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations"
<https://arxiv.org/abs/1911.07979>`_ paper
"""

import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import LEConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2


model = LEConv(dataset.num_features, dataset.num_classes)
x = torch.ones(data.num_nodes, data.num_features, dtype=torch.float)
edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "edge_index": edge_index,
}

# Call model
model(**inputs)
