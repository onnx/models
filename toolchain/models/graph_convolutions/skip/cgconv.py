# labels: test_group::turnkey name::cgconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The crystal graph convolutional operator from the `"Crystal Graph Convolutional Neural Networks
for an Accurate and Interpretable Prediction of Material Properties"
<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_paper
"""
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import CGConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2


model = CGConv(dataset.num_features)
inputs = {
    "x": torch.ones(data.num_nodes, data.num_features, dtype=torch.float),
    "edge_index": torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long),
}

# Call model
model(**inputs)
