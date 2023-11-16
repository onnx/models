# labels: test_group::turnkey name::nnconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The continuous kernel-based convolutional operator from the
`"Neural Message Passing for Quantum Chemistry"
<https://arxiv.org/abs/1704.01212>`_ paper.
This convolution is also known as the edge-conditioned convolution from the
`"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs"
<https://arxiv.org/abs/1704.02901>`_ paper
"""

import torch
import torch.nn as nn

from torch_geometric.datasets import TUDataset
from torch_geometric.nn import NNConv

dataset = TUDataset(root=".", name="ER_MD", use_edge_attr=True)
data = dataset[0]
edge_index_rows = 2


nn1 = nn.Sequential(nn.Linear(6, 25), nn.ReLU(), nn.Linear(25, data.num_features * 32))
model = NNConv(data.num_features, 32, nn1, aggr="mean")
x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
inputs = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
forward_input = (x, edge_index, edge_attr)
input_dict = {"forward": forward_input}
module = torch.jit.trace_module(model, input_dict)


# Call model
model(**inputs)
