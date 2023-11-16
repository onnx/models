# labels: test_group::turnkey name::appnp author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The approximate personalized propagation of neural predictions layer from the
`"Predict then Propagate: Graph Neural Networks meet Personalized PageRank"
<https://arxiv.org/abs/1810.05997>`_ paper
"""

from turnkeyml.parser import parse
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import APPNP

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2

# Parsing command-line arguments
k, alpha = parse(["k", "alpha"])


model = APPNP(k, alpha)  # k - number of iterations / alpha - teleport probability
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
