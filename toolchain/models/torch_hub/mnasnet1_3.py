# labels: test_group::turnkey name::mnasnet1_3 author::torch_hub task::Computer_Vision license::bsd-3-clause
from turnkeyml.parser import parse
import torch
from torchvision.models import MNASNet1_3_Weights

torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, num_channels, width, height = parse(
    ["pretrained", "batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = torch.hub.load(
    "pytorch/vision:v0.13.1",
    "mnasnet1_3",
    weights=MNASNet1_3_Weights.DEFAULT if pretrained else None,
)
model.eval()
inputs = {"x": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
