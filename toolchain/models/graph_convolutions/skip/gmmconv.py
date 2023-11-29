# labels: test_group::turnkey name::gmmconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The gaussian mixture model convolutional operator from the
`"Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
<https://arxiv.org/abs/1611.08402>`_ paper
"""

import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GMMConv

dataset = TUDataset(root=".", name="ER_MD", use_edge_attr=True)
data = dataset[0]
edge_index_rows = 2


model = GMMConv(data.num_features, out_channels=2, dim=6, kernel_size=2)
x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
inputs = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
forward_input = (x, edge_index, edge_attr)
input_dict = {"forward": forward_input}
module = torch.jit.trace_module(model, input_dict)


# Call model
model(**inputs)
