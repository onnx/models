# labels: test_group::turnkey name::efficientnet_b5 author::torch_hub task::Computer_Vision license::bsd-3-clause
# Skip reason: Fails during the analysis stage of turnkey
from turnkeyml.parser import parse
import torch
from torchvision.models import EfficientNet_B5_Weights

torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, num_channels, width, height = parse(
    ["pretrained", "batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = torch.hub.load(
    "pytorch/vision:v0.13.1",
    "efficientnet_b5",
    weights=EfficientNet_B5_Weights.DEFAULT if pretrained else None,
)
model.eval()
inputs = {"x": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
