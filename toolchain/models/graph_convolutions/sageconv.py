# labels: test_group::turnkey name::sageconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The GraphSAGE operator from the `"Inductive Representation Learning on Large Graphs"
<https://arxiv.org/abs/1706.02216>`_ paper
"""

import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2


model = SAGEConv(dataset.num_features, dataset.num_classes)
x = torch.ones(data.num_nodes, data.num_features, dtype=torch.float)
edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "edge_index": edge_index,
}

# Call model
model(**inputs)
