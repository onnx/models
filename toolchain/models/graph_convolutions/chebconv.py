# labels: test_group::turnkey name::chebconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The chebyshev spectral graph convolutional operator from the `"Convolutional
Neural Networks on Graphs with Fast Localized Spectral Filtering"
<https://arxiv.org/abs/1606.09375>`_ paper
"""
from turnkeyml.parser import parse
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import ChebConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2

# Parsing command-line arguments
k = parse(["k"])[0]

model = ChebConv(
    dataset.num_features, dataset.num_classes, k
)  # k - Chebyshev filter size
x = torch.ones(data.num_nodes, data.num_features, dtype=torch.float)
edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "edge_index": edge_index,
}

# Call model
model(**inputs)
