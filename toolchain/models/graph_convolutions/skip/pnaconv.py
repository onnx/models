# labels: test_group::turnkey name::pnaconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The Principal Neighbourhood Aggregation graph convolution operator from the
`"Principal Neighbourhood Aggregation for Graph Nets"
<https://arxiv.org/abs/2004.05718>`_ paper
"""
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import PNAConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2
deg = torch.ones(1, data.num_nodes, dtype=torch.int)


# PNAConv(in_channels, out_channels, aggregators, scalers, deg)
model = PNAConv(dataset.num_features, dataset.num_classes, ["sum"], ["identity"], deg)
x = torch.ones(data.num_nodes, data.num_features, dtype=torch.float)
edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "edge_index": edge_index,
}
forward_input = (x, edge_index)
input_dict = {"forward": forward_input}
module = torch.jit.trace_module(model, input_dict)


# Call model
model(**inputs)
