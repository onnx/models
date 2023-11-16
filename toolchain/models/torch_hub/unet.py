# labels: test_group::turnkey name::unet author::torch_hub task::Computer_Vision license::bsd-3-clause
"""https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/"""
from turnkeyml.parser import parse
import torch

# Parsing command-line arguments
batch_size, height, num_channels, width = parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
model = torch.hub.load(
    "mateuszbuda/brain-segmentation-pytorch",
    "unet",
    in_channels=3,
    out_channels=1,
    init_features=32,
    pretrained=True,
)

inputs = {"x": torch.ones(batch_size, num_channels, height, width, dtype=torch.float)}


# Call model
model(**inputs)
