# labels: test_group::turnkey name::gatedgraphconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The gated graph convolution operator from the `"Gated Graph Sequence Neural Networks"
<https://arxiv.org/abs/1511.05493>`_ paper
"""
from turnkeyml.parser import parse
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GatedGraphConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2

# Parsing command-line arguments
num_layers, out_channels = parse(["num_layers", "out_channels"])


model = GatedGraphConv(out_channels, num_layers)
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
