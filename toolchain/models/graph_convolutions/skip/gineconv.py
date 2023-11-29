# labels: test_group::turnkey name::gineconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The modified :class:`GINConv` operator from the
`"Strategies for Pre-training Graph Neural Networks"
<https://arxiv.org/abs/1905.12265>`_paper
"""

import torch
import torch.nn as nn

from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINEConv

dataset = TUDataset(root=".", name="ER_MD", use_edge_attr=True)
data = dataset[0]
edge_index_rows = 2


model = GINEConv(
    nn.Sequential(
        nn.Linear(dataset.num_features, dataset.num_classes),
        nn.ReLU(),
        nn.Linear(dataset.num_classes, dataset.num_classes),
    )
)
x = torch.zeros(18, 10, dtype=torch.float)
edge_index = data.edge_index
edge_attr = torch.zeros(306, 10, dtype=torch.long)
inputs = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
forward_input = (x, edge_index, edge_attr)
input_dict = {"forward": forward_input}
module = torch.jit.trace_module(model, input_dict)


# Call model
model(**inputs)
