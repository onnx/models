# labels: test_group::turnkey name::lgconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The Light Graph Convolution (LGC) operator from the
`"LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
<https://arxiv.org/abs/2002.02126>`_ paper
"""

import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import LGConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2


model = LGConv()
inputs = {
    "x": torch.ones(data.num_nodes, data.num_features, dtype=torch.float),
    "edge_index": torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long),
}

# Call model
model(**inputs)
