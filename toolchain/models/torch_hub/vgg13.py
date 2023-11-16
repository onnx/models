# labels: test_group::turnkey name::vgg13 author::torch_hub task::Computer_Vision license::bsd-3-clause
"""
https://github.com/pytorch/hub/blob/master/pytorch_vision_vgg.md
"""

from turnkeyml.parser import parse
import torch
from torchvision.models import VGG13_Weights

torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, num_channels, width, height = parse(
    ["pretrained", "batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = torch.hub.load(
    "pytorch/vision:v0.13.1",
    "vgg13",
    weights=VGG13_Weights.DEFAULT if pretrained else None,
)
model.eval()
inputs = {"x": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
