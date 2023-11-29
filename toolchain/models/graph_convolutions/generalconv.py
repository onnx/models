# labels: test_group::turnkey name::generalconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
A general GNN layer adapted from the `"Design Space for Graph Neural Networks"
<https://arxiv.org/abs/2011.08843>`_ paper.
"""

import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GeneralConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2


model = GeneralConv(dataset.num_features, dataset.num_classes)
x = torch.ones(data.num_nodes, data.num_features, dtype=torch.float)
edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "edge_index": edge_index,
}

# Call model
model(**inputs)
