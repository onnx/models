# labels: test_group::turnkey name::retinanet_resnet50_fpn_v2 author::torchvision task::Computer_Vision license::bsd-3-clause
"""
https://pytorch.org/vision/stable/models/retinanet.html
"""

from turnkeyml.parser import parse
import torch
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)


torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, num_channels, width, height = parse(
    ["pretrained", "batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = retinanet_resnet50_fpn_v2(
    weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
)
model.eval()
inputs = {"images": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
