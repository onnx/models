# labels: test_group::turnkey name::shufflenet_v2_x1_5 author::torch_hub task::Computer_Vision license::bsd-3-clause
"""
https://github.com/pytorch/hub/blob/master/pytorch_vision_shufflenet_v2.md
"""

from turnkeyml.parser import parse
import torch
from torchvision.models import ShuffleNet_V2_X1_5_Weights

torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, num_channels, width, height = parse(
    ["pretrained", "batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = torch.hub.load(
    "pytorch/vision:v0.13.1",
    "shufflenet_v2_x1_5",
    weights=ShuffleNet_V2_X1_5_Weights.DEFAULT if pretrained else None,
)
model.eval()
inputs = {"x": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
