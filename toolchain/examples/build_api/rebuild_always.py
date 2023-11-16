"""
    This example is built to demonstrate build_models()'s rebuild = "always" setting.

    build_models() will always rebuild the model, even when a build of that model is
    found in the build cache, when the `rebuild` argument is set to
    "always".

    You can demonstrate the functionality for rebuild="always" by running this
    script twice and seeing that the model still gets rebuilt even when the model
    is cached and there are no changes to the model.
"""

import torch
from turnkeyml import build_model

torch.manual_seed(0)


# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate model and generate inputs
input_size = 10
output_size = 5
pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

# Build/Rebuild model
build_model(pytorch_model, inputs, rebuild="always")

print("Example rebuild_always.py finished")
