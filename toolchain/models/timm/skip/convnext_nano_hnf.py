# labels: name::convnext_nano_hnf author::timm task::Computer_Vision license::apache-2.0
import torch
import timm
from turnkeyml.parser import parse

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, pretrained = parse(["batch_size", "pretrained"])

# Creating model and set it to evaluation mode
model = timm.create_model("convnext_nano_hnf", pretrained = pretrained)
model.eval()

# Creating inputs
input_size = model.default_cfg["input_size"]
batched_input_size = (batch_size,) + input_size
inputs = torch.rand(batched_input_size)

# Calling model
model(inputs)
