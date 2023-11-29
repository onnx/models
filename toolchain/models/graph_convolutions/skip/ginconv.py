# labels: test_group::turnkey name::ginconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The graph isomorphism operator from the “How Powerful are Graph Neural Networks?”
https://arxiv.org/abs/1810.00826
"""
import torch
import torch.nn as nn

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2


model = GINConv(
    nn.Sequential(
        nn.Linear(dataset.num_features, dataset.num_classes),
        nn.ReLU(),
        nn.Linear(dataset.num_classes, dataset.num_classes),
    )
)

inputs = {
    "x": torch.ones(data.num_nodes, data.num_features, dtype=torch.float),
    "edge_index": torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long),
}


# Call model
model(**inputs)
