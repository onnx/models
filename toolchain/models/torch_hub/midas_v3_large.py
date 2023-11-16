# labels: test_group::turnkey name::midas_v3_large author::torch_hub task::Computer_Vision license::bsd-3-clause
"""https://pytorch.org/hub/intelisl_midas_v2/"""
from turnkeyml.parser import parse
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, height, num_channels, width = parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")

inputs = {"x": torch.ones(batch_size, num_channels, height, width, dtype=torch.float)}


# Call model
model(**inputs)
