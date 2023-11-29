# labels: test_group::turnkey name::fcos_resnet50_fpn author::torchvision task::Computer_Vision license::bsd-3-clause
"""
https://pytorch.org/vision/stable/models/fcos.html
"""

from turnkeyml.parser import parse
import torch
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights


torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, num_channels, width, height = parse(
    ["pretrained", "batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = fcos_resnet50_fpn(
    weights=FCOS_ResNet50_FPN_Weights.DEFAULT if pretrained else None
)
model.eval()
inputs = {"images": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
