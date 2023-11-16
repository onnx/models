# labels: test_group::turnkey name::signedconv author::graph_convolutions task::Graph_Machine_Learning license::mit
"""
The signed graph convolutional operator from the `"Signed Graph Convolutional Network"
<https://arxiv.org/abs/1808.06354>`_ paper
"""
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SignedConv

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
edge_index_rows = 2

model = SignedConv(dataset.num_features, dataset.num_classes, True)
# first_aggr - Denotes which aggregation formula to use.
x = torch.ones(data.num_nodes, data.num_features, dtype=torch.float)
p_edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
n_edge_index = torch.ones(edge_index_rows, data.num_nodes, dtype=torch.long)
inputs = {
    "x": x,
    "p_edge_index": p_edge_index,
    "n_edge_index": n_edge_index,
}
forward_input = (x, p_edge_index, n_edge_index)
input_dict = {"forward": forward_input}
module = torch.jit.trace_module(model, input_dict)


# Call model
model(**inputs)
